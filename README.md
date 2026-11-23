# Trabalho N3 · Projeto Integrador de MLOps — Bank Marketing

Católica de Santa Catarina · Análise Preditiva · Prof. M. Sc. Rodrigo Hagstrom

**Integrantes:** Matheus Sady · Rafael Venturi · Rafael de Lara

**Entregáveis:** relatório técnico em [`reports/relatorio_n3.pdf`](reports/relatorio_n3.pdf) · evidências dos ciclos em
[`evidencias/ciclos/`](evidencias/ciclos) · capturas de tela em [`evidencias/prints/`](evidencias/prints) · histórico do
MLflow em `state/mlflow.db` (abrir com `python scripts/abrir_mlflow.py`).

Pipeline MLOps único e agendado que, a cada ciclo, incorpora um novo lote de dados, elege features por votação entre
três métodos, coloca dois algoritmos para competir, registra tudo no MLflow e decide automaticamente se o modelo
vigente deve ser substituído.

## Problema e dados

| Item | Definição |
|---|---|
| Problema | Classificação binária: o cliente contratará o depósito a prazo oferecido pela campanha de telemarketing? |
| Fonte | [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing) — `bank-additional-full.csv` (41.188 contatos, mai/2008 a nov/2010) |
| Licença | CC BY 4.0 (Moro, Cortez & Rita, 2014) |
| Unidade de observação | Um contato da campanha com um cliente |
| Alvo | `y` (`yes`/`no`); taxa de `yes` varia de ~5% a ~49% ao longo do tempo |
| Atributo removido | `duration` — só é conhecida após a ligação terminar (vazamento do alvo) |
| Métrica principal | **ROC-AUC** (maximizar). A prevalência muda muito entre lotes; a AUC independe do limiar e mede a capacidade de *ordenar* clientes, que é o uso real (priorizar ligações). F1 macro, acurácia balanceada, precisão e recall são secundárias. |

### Chegada de dados e particionamento

`scripts/preparar_dados.py` gera, uma única vez, a partir do arquivo original (SHA-256 conferido):

- `data/holdout/teste.csv` — 10% estratificado de todo o período, **reservado**: nunca participa de decisões, só é relatado;
- `data/raw/lotes/lote_001.csv … lote_005.csv` — o restante em **ordem cronológica**: `lote_001` com os 70% mais antigos e
  quatro lotes menores com o período seguinte. Cada ciclo incorpora o próximo lote ainda não processado
  (`state/estado_ingestao.json`), sem duplicar registros (`registro_id`).

Dentro de cada ciclo, o pool incorporado é dividido em **treino (80%) e validação (20%) pelo hash do `registro_id`**.
A atribuição é fixa entre ciclos, então o modelo vigente nunca treinou com registros da validação atual e a comparação
vigente × candidato é justa. Um corte temporal (últimos 20%) foi testado e descartado: cada fatia recente é um regime
econômico diferente e a validação mediria só essa ruptura (AUC ≈ 0,5 para qualquer modelo).

## Fluxo de um ciclo

```
schedule (cron) ─► 01 ETL + validação + drift ─► 02 três métodos + votação ─► 03 Modelo A × Modelo B
               ─► 04 eleição + regra de promoção ─► 05 previsões + teste integrado ─► commit do estado + artifact
```

| Notebook | Responsabilidade | Saídas no ciclo (`evidencias/ciclos/<cycle_id>/`) |
|---|---|---|
| `01_etl` | `cycle_id`, `data_version`, contrato de dados, limpeza, atributos, partição, drift | `contexto.json`, `qualidade_dados.json`, `diagnostico.json`, `drift.*` |
| `02_votacao_features` | SelectKBest (informação mútua), importância de Random Forest, RFE (Regressão Logística) e votação | `votacao.csv`, `ranking_*.csv`, `features_eleitas.json` |
| `03_treino_competicao` | Modelo A `LogisticRegression` × Modelo B `HistGradientBoostingClassifier`, mesmas features e protocolo | `comparacao_modelos.csv` |
| `04_selecao_promocao` | vencedor do ciclo, reavaliação do vigente, regra de promoção, manifesto, histórico | `decisao.json`, `resumo.md` |
| `05_previsoes` | carrega o vigente pelo manifesto, prevê o teste reservado, compara ciclos, verifica o ciclo | `previsoes.csv` |

Os notebooks documentam e orquestram; a lógica reutilizável fica em `src/`:
`ingestao.py`, `preprocessamento.py`, `features.py`, `modelagem.py`, `promocao.py`, `drift.py`, `rastreamento.py`, `ciclo.py`.

### Regras definidas antes do treino (`config/config.yaml`)

- **Votação:** cada método aprova seu top-10; é eleito o atributo com **≥ 2 de 3 votos**; se forem eleitos menos de 6,
  completa-se pelos não eleitos com mais votos e, em empate, menor posição média nos três rankings.
- **Promoção:** o vencedor do ciclo substitui o vigente somente se `AUC_vencedor ≥ AUC_vigente + 0,01` **e**
  `AUC_vencedor ≥ 0,60` **e** a validação de dados passou. Caso contrário o vigente é mantido. No primeiro ciclo basta o piso.
- **Drift (critério 10):** PSI ≥ 0,20 em 3 ou mais atributos ⇒ alerta.

### Rastreabilidade e persistência

- Todos os runs do MLflow levam as tags `cycle_id`, `data_version` e `etapa`, agrupados sob um run pai `ciclo-<cycle_id>`.
- O MLflow usa `state/mlflow.db` (SQLite) e `state/mlartifacts/`. Como o runner do GitHub Actions é descartado, o workflow
  **faz commit** de `state/`, `models/` e `evidencias/` ao fim de cada ciclo bem-sucedido. O próximo ciclo parte do
  estado versionado — nunca de arquivos de um runner anterior.
- Cada execução também publica um *artifact* baixável (retenção de 30 dias) só com o pacote daquele ciclo: notebooks
  executados, evidências, manifesto do vigente e histórico de ciclos. Ele é publicado mesmo quando o ciclo falha, para
  permitir investigar o erro.
- `models/modelo_vigente.json` registra o vigente (ciclo, `data_version`, algoritmo, `run_id`, features, métrica, caminho)
  e a decisão do último ciclo; cada promoção gera um novo arquivo em `models/historico/` (nada é sobrescrito).
- `state/historico_ciclos.csv` resume todos os ciclos em uma linha cada.

### Critério 10 — monitoramento de drift com alerta

- **Problema:** os lotes chegam de períodos diferentes (crise de 2008, mudança de juros, campanhas anteriores); a
  distribuição dos clientes muda e o modelo pode degradar sem que ninguém perceba.
- **Solução:** o notebook 01 calcula o PSI de cada atributo do lote novo contra os dados já incorporados.
- **Integração:** o resultado vai para o MLflow (`drift_alerta`, `drift_psi_maximo`), para o resumo do job no GitHub e, em
  caso de alerta, o workflow **abre uma issue** automaticamente com a tabela de PSI.
- **Evidência:** `evidencias/ciclos/<cycle_id>/drift.md` e as issues "Alerta de drift" do repositório.

## Como executar

### Instalação

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Um ciclo local (mesma sequência do workflow)

```bash
python scripts/executar_ciclo.py
```

Cada execução incorpora o próximo lote. Para voltar ao início (apaga MLflow, modelos e evidências; mantém os dados):

```bash
python scripts/resetar_estado.py --confirmar
```

### Consultar o MLflow

```bash
python scripts/abrir_mlflow.py
```

Abra http://127.0.0.1:5000 → experimento `n3-bank-marketing`. Filtro útil: `tags.cycle_id = '<cycle_id>'`.

### Reproduzir as previsões do vigente

```python
import json, joblib, pandas as pd
manifesto = json.load(open("models/modelo_vigente.json"))
modelo = joblib.load(manifesto["caminho_modelo"])  # pré-processamento + features + estimador
teste = pd.read_csv("data/processed/teste.csv")      # gerado pelo notebook 01
modelo.predict_proba(teste)[:, 1]
```

### Execução agendada (GitHub Actions)

`.github/workflows/pipeline_mlops.yml` é o **único** workflow, com dois gatilhos:

- `schedule` — cron a cada 20 minutos (minutos 07, 27 e 47);
- `workflow_dispatch` — depuração (`origem = manual`) e **sequência automática** (`origem = inicio` / `encadeado`).

**Por que existe a sequência automática.** Durante o trabalho, o serviço de agendamento do GitHub não criou nenhuma
execução para a conta deste repositório: os disparos manuais funcionavam, mas nenhum horário do cron foi atendido — nem
num repositório de teste mínimo (workflow que só imprime uma mensagem, cron a cada 5 minutos), e sem incidente na página de
status do GitHub. Como solução equivalente, o job `agendar-proximo` do mesmo workflow espera um intervalo ao fim de cada
ciclo bem-sucedido e dispara o seguinte, decrementando um contador. A sequência é aberta uma única vez:

```bash
gh workflow run pipeline_mlops.yml -f origem=inicio -f ciclos_restantes=6 -f intervalo_minutos=5
```

Daí em diante não há intervenção humana; uma falha interrompe a sequência. O cron continua no YAML e, quando o GitHub
atende algum horário, o ciclo aparece com gatilho `schedule`.

**Situação na entrega:** a sequência executou 7 ciclos (18:15–18:58 UTC de 23/11/2026, todos com sucesso) e o workflow
foi desligado em seguida para não gerar ciclos sem dados novos. Para religar:
`gh workflow enable pipeline_mlops.yml`. A linha do tempo do agendamento está em `evidencias/agendamento/`. Requisitos no repositório: *Settings → Actions → General → Workflow permissions → Read and write*. O
agendamento só roda na branch padrão. O gatilho e a origem de cada ciclo ficam registrados em
`evidencias/ciclos/<cycle_id>/execucao.json`.

### Preparar os dados do zero (opcional)

Os lotes já estão versionados. Para regerá-los a partir da UCI:

```bash
python scripts/preparar_dados.py
```

### Testes

```bash
python -m pytest -q
```

### Relatório técnico

Gerado a partir das evidências reais dos ciclos (tabelas de votos, comparação, decisões, drift e links das execuções):

```bash
python scripts/gerar_relatorio.py --somente-automaticos --pdf
```

Saída em `reports/relatorio_n3.html` e `reports/relatorio_n3.pdf` (o PDF usa o Chrome/Chromium em modo headless).

## Mapa dos critérios de avaliação

| # | Critério | Onde verificar |
|---|---|---|
| 1 | Problema, dados e avaliação | esta seção "Problema e dados", `config/config.yaml`, notebook 01 |
| 2 | ETL e engenharia de atributos | notebook 01, `src/preprocessamento.py`, `diagnostico.json` |
| 3 | Três métodos de seleção | notebook 02, `src/features.py`, `ranking_*.csv` |
| 4 | Votação e eleição | `votacao.csv`, `features_eleitas.json`, run `votacao_features` no MLflow |
| 5 | Dois algoritmos | notebook 03, runs `modelo_a-*` e `modelo_b-*` com o mesmo `cycle_id` |
| 6 | MLflow e rastreabilidade | `state/mlflow.db`, `scripts/abrir_mlflow.py`, `state/historico_ciclos.csv` |
| 7 | Workflow único e agendamento | `.github/workflows/pipeline_mlops.yml`, aba Actions, `execucao.json` |
| 8 | Eleição, promoção e previsões | notebooks 04 e 05, `models/modelo_vigente.json`, `decisao.json`, `previsoes.csv` |
| 9 | Repositório, relatório e evidências | este README, `reports/`, `evidencias/` |
| 10 | Recurso adicional | monitoramento de drift com alerta (seção acima) |

## Estrutura

```
.github/workflows/pipeline_mlops.yml   workflow único (schedule + workflow_dispatch)
config/config.yaml                     dados, partição, votação, algoritmos, métrica e regra de promoção
data/raw/lotes/                        fila de lotes (simula a chegada de dados novos)
data/holdout/teste.csv                 teste reservado
data/processed/                        dados do ciclo corrente (não versionado; regenerado)
notebooks/01..05                       etapas do pipeline
src/                                   funções reutilizáveis
scripts/                               preparar dados, executar ciclo, abrir MLflow, resetar estado
models/modelo_vigente.json             manifesto do modelo vigente
models/historico/                      todos os modelos já promovidos
state/                                 MLflow (SQLite + artefatos), estado de ingestão, histórico de ciclos
evidencias/ciclos/<cycle_id>/          notebooks executados e saídas de cada ciclo
reports/                               relatório técnico (PDF)
tests/                                 testes das regras de votação, promoção, drift e partição
```
