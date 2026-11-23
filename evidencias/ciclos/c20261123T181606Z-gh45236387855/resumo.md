# Ciclo `c20261123T181606Z-gh45236387855`

- **data_version:** `v001-a59ff626` (lotes novos: lote_001.csv)
- **Features eleitas (10):** euribor3m, month, cons.conf.idx, nr.employed, emp.var.rate, faixa_etaria, cons.price.idx, job, education, contact

| Candidato | roc_auc validação | run_id |
|---|---|---|
| modelo_a · logistic_regression | 0.6658 | `4fd2176beabd44e48edc2e69837574f0` |
| modelo_b · hist_gradient_boosting | 0.6479 | `43ce505951ad4514abd20014f450fdb0` |

- **Vencedor do ciclo:** logistic_regression
- **Vigente anterior:** nenhum
- **Decisão:** `promover` - Não há modelo vigente e o candidato supera o piso.
- **Vigente após o ciclo:** logistic_regression (ciclo `c20261123T181606Z-gh45236387855`, run `4fd2176beabd44e48edc2e69837574f0`)
- **roc_auc do vigente no teste reservado:** 0.7164
- **Drift:** sem alerta
