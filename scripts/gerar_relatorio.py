"""Gera o relatório técnico a partir das evidências reais dos ciclos.

Uso:
    python scripts/gerar_relatorio.py                     # todos os ciclos
    python scripts/gerar_relatorio.py --somente-automaticos # só ciclos disparados sem intervenção
                                                            # (schedule ou sequência automática)
    python scripts/gerar_relatorio.py --pdf               # também gera o PDF (Chrome/Chromium headless)

Saídas: reports/relatorio_n3.html e, com --pdf, reports/relatorio_n3.pdf.
Fontes: state/historico_ciclos.csv, evidencias/ciclos/<cycle_id>/*, models/modelo_vigente.json, state/mlflow.db.
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import re
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src.config import caminho, carregar_config  # noqa: E402

INTEGRANTES = ["Matheus Sady", "Rafael Venturi", "Rafael de Lara"]
REPOSITORIO = "https://github.com/Bornety0/projeto-n3-mlops"
# 'inicio' abre a sequência automática; os ciclos seguintes chegam como 'encadeado'.
ORIGENS_AUTOMATICAS = {"schedule", "inicio", "encadeado"}
NAVEGADORES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "google-chrome", "chromium", "chromium-browser",
]


def ler_json(arquivo: Path) -> dict:
    return json.loads(arquivo.read_text(encoding="utf-8")) if arquivo.exists() else {}


def esc(valor) -> str:
    return html.escape("" if valor is None or (isinstance(valor, float) and pd.isna(valor)) else str(valor))


def tabela_html(df: pd.DataFrame, indice: bool = False, colunas_html: tuple[str, ...] = ()) -> str:
    """Escapa todas as células, exceto as colunas que já trazem HTML montado aqui (links)."""
    if not colunas_html:
        return df.to_html(index=indice, border=0, classes="tabela", na_rep="-", escape=True)
    seguro = df.copy()
    for coluna in seguro.columns:
        if coluna not in colunas_html:
            seguro[coluna] = seguro[coluna].map(esc)
    return seguro.to_html(index=indice, border=0, classes="tabela", na_rep="-", escape=False)


def grafico_base64(fig) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return f'<img src="data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}">'


def carregar_ciclos(cfg: dict, somente_automaticos: bool) -> pd.DataFrame:
    arquivo = caminho(cfg, cfg["saida"]["historico_ciclos"])
    if not arquivo.exists():
        raise SystemExit("Nenhum ciclo executado ainda (state/historico_ciclos.csv não existe).")
    historico = pd.read_csv(arquivo)
    pasta = caminho(cfg, cfg["saida"]["diretorio_evidencias"])
    execucoes = [ler_json(pasta / c / "execucao.json") for c in historico["cycle_id"]]
    historico["gatilho"] = [e.get("gatilho", "local") for e in execucoes]
    historico["origem"] = [e.get("origem", e.get("gatilho", "local")) for e in execucoes]
    historico["run_url"] = [e.get("run_url", "") for e in execucoes]
    if somente_automaticos:
        historico = historico[historico["origem"].isin(ORIGENS_AUTOMATICAS)]
    if historico.empty:
        raise SystemExit("Nenhum ciclo com o filtro escolhido.")
    historico.insert(0, "ciclo", range(1, len(historico) + 1))
    return historico.reset_index(drop=True)


def contar_runs_mlflow(cfg: dict) -> int:
    banco = caminho(cfg, cfg["mlflow"]["tracking_db"])
    if not banco.exists():
        return 0
    with sqlite3.connect(banco) as conexao:
        return conexao.execute("SELECT COUNT(*) FROM runs WHERE lifecycle_stage = 'active'").fetchone()[0]


def resumo_lotes(cfg: dict) -> pd.DataFrame:
    linhas = []
    for arquivo in sorted(caminho(cfg, cfg["dados"]["diretorio_lotes"]).glob("lote_*.csv")):
        df = pd.read_csv(arquivo, usecols=["registro_id", "y"])
        linhas.append({"lote": arquivo.stem, "registros": len(df), "ids": f"{df.registro_id.min()}-{df.registro_id.max()}", "taxa de yes": f"{(df.y == 'yes').mean():.1%}"})
    teste = pd.read_csv(caminho(cfg, cfg["dados"]["arquivo_teste"]), usecols=["y"])
    linhas.append({"lote": "teste (reservado)", "registros": len(teste), "ids": "amostra estratificada", "taxa de yes": f"{(teste.y == 'yes').mean():.1%}"})
    return pd.DataFrame(linhas)


def secao_votacao(cfg: dict, ciclos: pd.DataFrame) -> str:
    pasta = caminho(cfg, cfg["saida"]["diretorio_evidencias"])
    ultimo = ciclos.iloc[-1]["cycle_id"]
    votos = pd.read_csv(pasta / ultimo / "votacao.csv")
    colunas = ["feature", "voto_selectkbest", "voto_random_forest", "voto_rfe", "total_votos", "posicao_media", "eleita", "criterio"]
    votos = votos[colunas].rename(columns={"voto_selectkbest": "SelectKBest", "voto_random_forest": "Random Forest", "voto_rfe": "RFE"})

    evolucao = pd.DataFrame(index=sorted(cfg["features"]["numericas"] + cfg["features"]["categoricas"]))
    for _, ciclo in ciclos.iterrows():
        eleitas = set(str(ciclo["features_eleitas"]).split("|"))
        evolucao[f"C{ciclo['ciclo']}"] = ["●" if f in eleitas else "" for f in evolucao.index]
    evolucao = evolucao[(evolucao != "").any(axis=1)]
    evolucao.index.name = "feature"
    return (
        f"<h3>Tabela de votos do ciclo {len(ciclos)} (<code>{esc(ultimo)}</code>)</h3>"
        + tabela_html(votos)
        + "<h3>Features eleitas em cada ciclo</h3><p>● = eleita no ciclo. A votação é refeita a cada ciclo com os dados disponíveis.</p>"
        + tabela_html(evolucao.reset_index())
    )


def grafico_metricas(ciclos: pd.DataFrame, metrica: str) -> str:
    fig, eixo = plt.subplots(figsize=(8, 3.6))
    eixo.plot(ciclos["ciclo"], ciclos[f"{metrica}_modelo_a"], marker="o", label="Modelo A (validação)")
    eixo.plot(ciclos["ciclo"], ciclos[f"{metrica}_modelo_b"], marker="o", label="Modelo B (validação)")
    teste = pd.to_numeric(ciclos[f"{metrica}_teste_vigente"], errors="coerce")
    eixo.plot(ciclos["ciclo"], teste, marker="s", linestyle="--", color="black", label="Vigente após o ciclo (teste reservado)")
    for _, linha in ciclos[ciclos["decisao"] == "promover"].iterrows():
        eixo.axvline(linha["ciclo"], color="green", alpha=0.15, linewidth=8)
    eixo.set_xlabel("ciclo (faixa verde = promoção)")
    eixo.set_ylabel(metrica)
    eixo.set_xticks(ciclos["ciclo"])
    eixo.grid(alpha=0.3)
    eixo.legend(fontsize=8, loc="lower right")
    return grafico_base64(fig)


LEGENDAS_PRINTS = {
    "01_actions_execucoes": "Execuções do workflow: o início da sequência por Bornety0 e os ciclos seguintes disparados pelo próprio pipeline (github-actions[bot]).",
    "02_actions_filtro_schedule_vazio": "Filtro event:schedule no Actions: nenhuma execução criada pelo agendamento nativo, apesar do cron configurado.",
    "03_execucao_ciclo4_promocao": "Ciclo com promoção: job ciclo seguido de agendar-proximo, e o pacote do ciclo publicado como artifact.",
    "04_issues_drift": "Issues abertas automaticamente pelo monitoramento de drift (critério 10).",
    "05_github_status": "Histórico de incidentes do GitHub: nenhum incidente do Actions registrado no dia da execução.",
    "06_mlflow_runs": "MLflow: um run pai por ciclo (cycle_id), agrupando as etapas registradas pelos notebooks.",
    "07_mlflow_comparacao_modelos": "MLflow: Modelo A × Modelo B do mesmo ciclo, com as mesmas features, hiperparâmetros e métricas lado a lado.",
}


def secao_prints() -> str:
    """Prints em evidencias/prints/ entram embutidos, para o PDF ficar autocontido."""
    arquivos = sorted((RAIZ / "evidencias" / "prints").glob("*.png"))
    if not arquivos:
        return ""
    blocos = ["<h3>Capturas de tela</h3>"]
    for arquivo in arquivos:
        imagem = base64.b64encode(arquivo.read_bytes()).decode()
        legenda = LEGENDAS_PRINTS.get(arquivo.stem, arquivo.stem)
        blocos.append(
            f'<figure class="print"><img src="data:image/png;base64,{imagem}">'
            f"<figcaption>{esc(legenda)} <code>{esc(arquivo.name)}</code></figcaption></figure>"
        )
    return "\n".join(blocos)


def markdown_simples(texto: str) -> str:
    """Converte o subconjunto de Markdown usado em reports/conclusao.md: ### títulos, listas com '- ',
    parágrafos, **negrito** e `código`."""

    def inline(trecho: str) -> str:
        trecho = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", esc(trecho))
        return re.sub(r"`(.+?)`", r"<code>\1</code>", trecho)

    blocos = []
    for bloco in re.split(r"\n\s*\n", texto.strip()):
        linhas = [linha.strip() for linha in bloco.splitlines() if linha.strip()]
        if linhas[0].startswith("### "):
            blocos.append(f"<h3>{inline(linhas[0][4:])}</h3>")
            linhas = linhas[1:]
        if not linhas:
            continue
        if all(linha.startswith("- ") for linha in linhas):
            blocos.append("<ul>" + "".join(f"<li>{inline(l[2:])}</li>" for l in linhas) + "</ul>")
        else:
            blocos.append(f"<p>{inline(' '.join(linhas))}</p>")
    return "\n".join(blocos)


def secao_conclusao() -> str:
    """Texto das conclusões escrito pelo grupo em reports/conclusao.md (editável sem mexer no código)."""
    arquivo = RAIZ / "reports" / "conclusao.md"
    return markdown_simples(arquivo.read_text(encoding="utf-8")) if arquivo.exists() else ""


def secao_drift(cfg: dict, ciclos: pd.DataFrame) -> str:
    pasta = caminho(cfg, cfg["saida"]["diretorio_evidencias"])
    linhas = []
    for _, ciclo in ciclos.iterrows():
        drift = ler_json(pasta / ciclo["cycle_id"] / "drift.json")
        if not drift.get("avaliado"):
            linhas.append({"ciclo": ciclo["ciclo"], "lote novo": ciclo["lotes_novos"], "avaliado": "não", "atributos em alerta": "-", "PSI máximo": "-", "taxa de yes (ref → novo)": "-", "alerta": "-"})
            continue
        linhas.append({
            "ciclo": ciclo["ciclo"],
            "lote novo": ciclo["lotes_novos"],
            "avaliado": "sim",
            "atributos em alerta": ", ".join(drift["features_em_alerta"]) or "-",
            "PSI máximo": f"{drift['psi_maximo']:.3f}",
            "taxa de yes (ref → novo)": f"{drift['taxa_positiva_referencia']:.1%} → {drift['taxa_positiva_novo']:.1%}",
            "alerta": "SIM (issue aberta)" if drift["alerta"] else "não",
        })
    return tabela_html(pd.DataFrame(linhas))


def montar_html(cfg: dict, ciclos: pd.DataFrame) -> str:
    metrica = cfg["avaliacao"]["metrica_principal"]
    manifesto = ler_json(caminho(cfg, cfg["promocao"]["manifesto"]))
    pasta = caminho(cfg, cfg["saida"]["diretorio_evidencias"])
    ultimo = ciclos.iloc[-1]["cycle_id"]
    diagnostico = ler_json(pasta / ultimo / "diagnostico.json")
    votacao, promocao, drift_cfg, part = cfg["votacao"], cfg["promocao"], cfg["drift"], cfg["particionamento"]

    competicao = ciclos[["ciclo", "data_version", "lotes_novos", "n_treino", "n_validacao", f"{metrica}_modelo_a", f"{metrica}_modelo_b", "vencedor"]]
    decisoes = ciclos[["ciclo", "vencedor", f"{metrica}_modelo_a", f"{metrica}_modelo_b", f"{metrica}_vigente_anterior", "decisao", "vigente_apos", f"{metrica}_teste_vigente"]]
    execucoes = ciclos[["ciclo", "cycle_id", "gatilho", "origem", "data_version", "decisao"]].copy()
    execucoes["execução no GitHub"] = [f'<a href="{esc(u)}">{esc(u.rsplit("/", 1)[-1])}</a>' if u else "-" for u in ciclos["run_url"]]
    n_agendados = int((ciclos["gatilho"] == "schedule").sum())
    n_automaticos = int(ciclos["origem"].isin(ORIGENS_AUTOMATICAS).sum())
    promocoes = int((ciclos["decisao"] == "promover").sum())
    manutencoes = int((ciclos["decisao"] == "manter_vigente").sum())
    alertas = int(ciclos["drift_alerta"].astype(str).str.lower().eq("true").sum())

    tratamentos = pd.DataFrame([
        ["remover duplicatas idênticas em todas as colunas originais", f"{diagnostico.get('depois', {}).get('limpeza_pool', {}).get('duplicatas_removidas', '-')} registros no último ciclo; comparar após remover duration descartaria contatos legítimos"],
        ["remover duration", "só é conhecida após a ligação terminar: vazamento do alvo"],
        ["manter 'unknown' como categoria", "a ausência de informação é informativa (ex.: default)"],
        ["pdays = 999 → contatado_antes + pdays_dias (ausente)", "999 é um código de 'nunca contatado', não uma quantidade de dias"],
        ["campaign_log = log1p(campaign)", "cauda longa de contatos; reduz o peso de outliers sem descartá-los"],
        ["qtd_desconhecidos, faixa_etaria", "resumem incompletude do cadastro e perfil etário"],
        ["imputação, padronização e one-hot dentro do Pipeline", "ajustados somente no treino de cada ciclo (sem vazamento)"],
    ], columns=["tratamento", "justificativa"])

    teste_final = manifesto.get("ultimo_ciclo", {}).get("teste_vigente", {}).get(metrica, "-")
    conclusao = secao_conclusao() or (
        f"<p>O pipeline executou {len(ciclos)} ciclo(s) de forma automática, refazendo a votação e a competição a cada"
        f" ciclo e aplicando a regra de promoção sem intervenção: {promocoes} promoção(ões) e {manutencoes}"
        f" manutenção(ões). O modelo vigente final é <b>{esc(manifesto.get('algoritmo'))}</b>, com {metrica} de"
        f" {esc(teste_final)} no teste reservado.</p>"
    )

    unknown = diagnostico.get("antes", {}).get("unknown", {})
    texto_unknown = ", ".join(f"{k}: {v}" for k, v in sorted(unknown.items(), key=lambda kv: -kv[1]))

    return f"""<!doctype html>
<html lang="pt-BR"><head><meta charset="utf-8"><title>Relatório Técnico - Trabalho N3</title>
<style>
  body {{ font-family: -apple-system, "Segoe UI", Roboto, Arial, sans-serif; color: #222; max-width: 960px; margin: 32px auto; padding: 0 24px; line-height: 1.45; font-size: 13px; }}
  h1 {{ color: #7a1b35; font-size: 24px; margin-bottom: 4px; }}
  h2 {{ color: #7a1b35; border-bottom: 2px solid #7a1b35; padding-bottom: 4px; margin-top: 32px; font-size: 18px; }}
  h3 {{ font-size: 14px; margin-top: 20px; }}
  .tabela {{ border-collapse: collapse; width: 100%; margin: 8px 0 16px; font-size: 11.5px; }}
  .tabela th {{ background: #f3eef0; text-align: left; padding: 5px 6px; }}
  .tabela td {{ border-top: 1px solid #e6e0e2; padding: 4px 6px; vertical-align: top; }}
  code {{ background: #f5f5f5; padding: 1px 4px; border-radius: 3px; font-size: 11px; }}
  .fluxo {{ display: flex; flex-wrap: wrap; gap: 6px; margin: 12px 0; }}
  .fluxo div {{ background: #f3eef0; border-radius: 6px; padding: 6px 10px; font-size: 11.5px; }}
  .destaque {{ background: #f7f3f4; border-left: 4px solid #7a1b35; padding: 8px 12px; margin: 12px 0; }}
  img {{ width: 640px; max-width: 100%; display: block; margin: 8px auto 16px; }}
  figure.print {{ margin: 12px 0 20px; page-break-inside: avoid; }}
  figure.print img {{ width: 100%; border: 1px solid #ddd; margin: 0 0 4px; }}
  figcaption {{ font-size: 11px; color: #555; }}
  @media print {{ h2 {{ page-break-after: avoid; }} .tabela {{ page-break-inside: auto; }} }}
</style></head><body>

<h1>Relatório Técnico · Trabalho N3 · Projeto Integrador de MLOps</h1>
<p>Católica de Santa Catarina · Análise Preditiva · Prof. M. Sc. Rodrigo Hagstrom<br>
<b>Integrantes:</b> {esc(" · ".join(INTEGRANTES))}<br>
<b>Repositório:</b> <a href="{REPOSITORIO}">{REPOSITORIO}</a><br>
<b>Ciclos considerados:</b> {len(ciclos)} ({n_automaticos} automáticos, dos quais {n_agendados} por <code>schedule</code>) · <b>runs no MLflow:</b> {contar_runs_mlflow(cfg)}</p>

<h2>1. Problema e dados</h2>
<p>Classificação binária: prever se o cliente contratará o depósito a prazo (<code>y</code>) oferecido por campanhas de
telemarketing de um banco português. Fonte: {esc(cfg['dados']['fonte'])}, licença {esc(cfg['dados']['licenca'])}.
Unidade de observação: um contato da campanha. 41.188 registros de mai/2008 a nov/2010, 20 atributos (dados do cliente,
do contato, de campanhas anteriores e indicadores macroeconômicos).</p>
<p><b>Chegada de novos registros:</b> o arquivo original (SHA-256 conferido) foi dividido em um teste reservado e numa fila
de lotes em ordem cronológica. Cada ciclo incorpora o próximo lote ainda não processado, sem duplicar <code>registro_id</code>.</p>
{tabela_html(resumo_lotes(cfg))}
<p><b>Limitações dos dados:</b> a taxa de conversão muda muito ao longo do tempo; o início da série quase não varia nos
indicadores econômicos e não tem campanhas anteriores (<code>poutcome</code> sempre <i>nonexistent</i>), o que limita o
sinal disponível no primeiro ciclo; <code>duration</code> foi removida por ser conhecida só após a ligação.</p>

<h2>2. Protocolo de avaliação (definido antes do treino)</h2>
<ul>
<li><b>Métrica principal:</b> <code>{esc(metrica)}</code> ({esc(cfg['avaliacao']['direcao'])}). A prevalência de <i>yes</i> varia de
~5% a ~49% entre lotes; a ROC-AUC independe do limiar e da prevalência e mede a capacidade de <b>ordenar</b> clientes por
chance de conversão, que é o uso real (priorizar ligações). Secundárias: {esc(", ".join(cfg['avaliacao']['metricas_secundarias']))}.</li>
<li><b>Teste reservado:</b> {part['fracao_teste']:.0%} estratificado de todo o período; nunca usado em decisões, apenas relatado.</li>
<li><b>Treino × validação:</b> {1 - part['fracao_validacao']:.0%} / {part['fracao_validacao']:.0%} do pool pelo hash do
<code>registro_id</code>. Cada registro fica sempre do mesmo lado em todos os ciclos, então o modelo vigente nunca treinou
com a validação atual e a comparação vigente × candidato é justa. Um corte temporal (últimos 20%) foi avaliado e descartado:
cada fatia recente é um regime econômico diferente e a validação mediria apenas a ruptura (AUC ≈ 0,5 para qualquer modelo).</li>
<li><b>Lote inicial:</b> {part['fracao_lote_inicial']:.0%} mais antigos do histórico; com 50% o primeiro ciclo não tinha sinal útil.</li>
</ul>

<h2>3. ETL e engenharia de atributos</h2>
<p>O notebook 01 valida o contrato de dados (colunas, alvo binário, ids únicos, faixas de idade e de contatos, limite de
ausentes); qualquer reprovação interrompe o ciclo antes do treino. Diagnóstico do último ciclo: não há <code>NaN</code>,
mas há valores <i>unknown</i> ({esc(texto_unknown)}).</p>
{tabela_html(tratamentos)}

<h2>4. Três métodos de seleção e votação de features</h2>
<p>Métodos executados no mesmo notebook (02), ajustados apenas no treino do ciclo: <b>SelectKBest</b> com informação mútua,
<b>importância de Random Forest</b> e <b>RFE</b> com Regressão Logística. A importância das colunas one-hot é agregada ao
atributo original, então a votação elege atributos interpretáveis.</p>
<div class="destaque"><b>Regra de votação:</b> cada método aprova seu top-{votacao['top_k']}; é eleito o atributo com
≥ {votacao['votos_minimos']} de 3 votos; se forem eleitos menos de {votacao['min_features']}, completa-se pelos não eleitos com
mais votos e, em empate, menor posição média nos três rankings.</div>
{secao_votacao(cfg, ciclos)}

<h2>5. Competição entre os dois algoritmos</h2>
<p><b>Modelo A:</b> {esc(cfg['modelos']['modelo_a']['classe'])} · <b>Modelo B:</b> {esc(cfg['modelos']['modelo_b']['classe'])}.
Mesmas features eleitas, mesmo treino e validação, mesma seed ({cfg['projeto']['seed']}), <code>class_weight="balanced"</code>.
Cada candidato é um run no MLflow com parâmetros, métricas, features e modelo serializado, marcados com o mesmo <code>cycle_id</code>.</p>
{tabela_html(competicao)}
{grafico_metricas(ciclos, metrica)}

<h2>6. Eleição do vencedor e decisão de promoção</h2>
<div class="destaque"><b>Regra de promoção:</b> o vencedor do ciclo substitui o vigente somente se
<code>{metrica}_vencedor ≥ {metrica}_vigente + {promocao['melhoria_minima']}</code> e <code>{metrica}_vencedor ≥ {promocao['metrica_minima']}</code>
e a validação de dados passou. O vigente é reavaliado na mesma validação do ciclo. No primeiro ciclo basta o piso.</div>
<p>Resultado: <b>{promocoes}</b> promoção(ões) e <b>{manutencoes}</b> manutenção(ões) do vigente.</p>
{tabela_html(decisoes)}
<h3>Manifesto do modelo vigente (<code>models/modelo_vigente.json</code>)</h3>
<p>Ciclo <code>{esc(manifesto.get('cycle_id'))}</code> · algoritmo <b>{esc(manifesto.get('algoritmo'))}</b> · data_version
<code>{esc(manifesto.get('data_version'))}</code> · run_id <code>{esc(manifesto.get('run_id'))}</code><br>
Features: {esc(", ".join(manifesto.get('features', [])))}<br>
Arquivo: <code>{esc(manifesto.get('caminho_modelo'))}</code> (pré-processamento + features + estimador). Modelos anteriores
preservados em <code>models/historico/</code>.</p>

<h2>7. Arquitetura do fluxo único</h2>
<div class="fluxo"><div>schedule (cron)</div><div>01 dados novos, contrato, ETL, drift</div><div>02 três métodos + votação</div>
<div>03 Modelo A × Modelo B</div><div>04 eleição + promoção + manifesto</div><div>05 previsões + teste integrado</div>
<div>commit do estado + artifact</div></div>
<p>Um único workflow (<code>.github/workflows/pipeline_mlops.yml</code>) com <code>schedule</code> e <code>workflow_dispatch</code>
executa os testes das regras e, em sequência, os notebooks 01→05 num único job; qualquer falha interrompe o fluxo antes da
promoção. O estado do ciclo só é efetivado no notebook 04. Como o runner é temporário, o MLflow usa SQLite + artefatos em
<code>state/</code>, e o workflow faz commit de <code>state/</code>, <code>models/</code> e <code>evidencias/</code> ao fim de cada ciclo;
o ciclo seguinte parte do estado versionado. Cada execução também publica o pacote do ciclo como <i>artifact</i>.</p>
<h3>Agendamento: cron nativo e contingência automática</h3>
<p>O workflow declara o gatilho <code>schedule</code> (cron). Durante a execução do trabalho, o serviço de agendamento do
GitHub não criou nenhuma execução para a conta do repositório: os disparos manuais funcionavam, mas nenhum horário do
cron foi atendido — nem num repositório de teste mínimo (<code>teste-agendamento-n3</code>), com um workflow que só imprime
uma mensagem e cron a cada 5 minutos. A página de status do GitHub não indicava incidente. Como solução equivalente, o
mesmo workflow ganhou o job <code>agendar-proximo</code>: ao fim de cada ciclo bem-sucedido ele espera um intervalo e
dispara o ciclo seguinte (<code>origem = encadeado</code>), decrementando um contador até zero. A sequência é aberta uma
única vez (<code>origem = inicio</code>); daí em diante não há intervenção humana, e uma falha interrompe a sequência. O cron
permanece no YAML: execuções por <code>schedule</code>, quando ocorrem, aparecem na tabela abaixo com esse gatilho.</p>

<h2>8. Evidências de execução</h2>
{tabela_html(execucoes, colunas_html=("execução no GitHub",))}
<p>Cada ciclo guarda em <code>evidencias/ciclos/&lt;cycle_id&gt;/</code> os notebooks executados, <code>execucao.json</code>
(gatilho e link da execução), contrato de dados, diagnóstico, votação, comparação, decisão, drift e previsões.
O MLflow é consultado com <code>python scripts/abrir_mlflow.py</code>. A linha do tempo do agendamento, com as consultas
à API do GitHub, está em <code>evidencias/agendamento/</code>.</p>
{secao_prints()}

<h2>9. Recurso adicional (critério 10): monitoramento de drift com alerta</h2>
<ul>
<li><b>Problema identificado:</b> os lotes vêm de períodos diferentes (crise de 2008, mudança de juros, campanhas anteriores);
a distribuição dos clientes muda e o modelo pode degradar sem que ninguém perceba.</li>
<li><b>Justificativa:</b> a regra de promoção compara modelos, mas não avisa quando os <i>dados</i> mudam; o drift é o sinal
antecipado para investigar o modelo e o processo de negócio.</li>
<li><b>Solução:</b> PSI de cada atributo do lote novo contra os dados já incorporados (faixas pelos quantis da referência para
numéricos, frequências para categóricos). PSI ≥ {drift_cfg['psi_alerta']} em {drift_cfg['min_features_alerta']} ou mais atributos ⇒ alerta.</li>
<li><b>Integração:</b> notebook 01 (todo ciclo); métricas <code>drift_alerta</code> e <code>drift_psi_maximo</code> no MLflow; tabela no
resumo do job do GitHub Actions; <b>issue aberta automaticamente</b> quando há alerta.</li>
<li><b>Resultado demonstrado:</b> {alertas} ciclo(s) com alerta.</li>
</ul>
{secao_drift(cfg, ciclos)}

<h2>10. Limitações e conclusões</h2>
<h3>Limitações</h3>
<ul>
<li>A chegada de dados é simulada por uma fila de lotes pré-definidos a partir de um dataset histórico.</li>
<li>O teste reservado cobre todo o período; enquanto os lotes recentes não são incorporados, ele mede extrapolação para
regimes ainda não vistos — por isso a AUC no teste pode cair num ciclo e subir quando o regime entra no treino.</li>
<li>O limiar de classificação é fixo em 0,5 e as probabilidades não são calibradas; F1 e recall dependem disso.</li>
<li>A validação por hash mistura períodos já incorporados; ela mede generalização dentro do histórico disponível, não
previsão do futuro imediato.</li>
<li>O estado em SQLite versionado no Git é adequado ao escopo acadêmico; em produção seria substituído por um servidor
MLflow com armazenamento de objetos.</li>
</ul>
{conclusao}
</body></html>"""


def gerar_pdf(origem: Path, destino: Path) -> bool:
    for candidato in NAVEGADORES:
        executavel = candidato if Path(candidato).exists() else shutil.which(candidato)
        if executavel:
            subprocess.run(
                [executavel, "--headless", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={destino}", origem.as_uri()],
                check=True, capture_output=True,
            )
            return destino.exists()
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--somente-automaticos", action="store_true")
    parser.add_argument("--pdf", action="store_true")
    args = parser.parse_args()

    cfg = carregar_config(RAIZ)
    ciclos = carregar_ciclos(cfg, args.somente_automaticos)
    saida = RAIZ / "reports" / "relatorio_n3.html"
    saida.write_text(montar_html(cfg, ciclos), encoding="utf-8")
    print(f"HTML: {saida.relative_to(RAIZ)} ({len(ciclos)} ciclos)")
    if args.pdf:
        pdf = saida.with_suffix(".pdf")
        print(f"PDF: {pdf.relative_to(RAIZ)}" if gerar_pdf(saida, pdf) else "Chrome/Chromium não encontrado: abra o HTML e imprima em PDF.")


if __name__ == "__main__":
    main()
