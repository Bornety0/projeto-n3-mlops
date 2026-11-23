# Roteiro do relatório técnico (exportar como `reports/relatorio_n3.pdf`)

Estrutura exigida pela seção 7 do enunciado. O relatório é gerado a partir das evidências reais dos ciclos:

```bash
git pull
python scripts/gerar_relatorio.py --somente-agendados --pdf
```

O script preenche as seções 1-10 abaixo com tabelas e gráficos. Antes de entregar, o grupo revisa o texto, acrescenta
prints do MLflow e do Actions na seção de evidências e escreve as próprias conclusões.

1. **Identificação** — integrantes (Matheus Sady, Rafael Venturi, Rafael de Lara), disciplina, link do repositório.
2. **Problema e dados** — fonte/licença, unidade de observação, alvo, atributos, limitações (prevalência variável, período
   inicial com pouco sinal, remoção de `duration`). Métrica (ROC-AUC) e justificativa; partição (teste reservado
   estratificado, validação por hash do `registro_id`, lotes cronológicos) e por que o corte temporal foi descartado.
3. **ETL** — diagnóstico antes/depois (`diagnostico.json`), tratamentos e justificativas (tabela do notebook 01),
   contrato de dados e o que acontece se ele falhar.
4. **Seleção de features** — os três métodos, regra de votação definida antes, **tabela de votos** de pelo menos um ciclo
   (`votacao.csv`) e como o conjunto eleito mudou entre ciclos (notebook 05).
5. **Competição dos modelos** — Modelo A × Modelo B por ciclo (`comparacao_modelos.csv` / `historico_ciclos.csv`).
6. **Decisão de promoção** — regra, decisões de cada ciclo (promover / manter) com os números, manifesto final.
7. **Arquitetura do fluxo único** — diagrama do README, workflow, persistência do estado entre runners.
8. **Evidências de execução** — links dos runs com gatilho `schedule` (`execucao.json`), prints do MLflow com runs
   agrupados por `cycle_id`, resumo do job, issue de drift.
9. **Critério 10** — problema identificado, justificativa, solução, ponto de integração e resultado demonstrado.
10. **Limitações e conclusões** — ex.: simulação de chegada de dados por lotes pré-definidos, teste com prevalência
    diferente dos primeiros lotes, limiar de classificação fixo em 0,5, ausência de calibração.
