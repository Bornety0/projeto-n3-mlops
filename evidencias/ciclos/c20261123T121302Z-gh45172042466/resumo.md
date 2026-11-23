# Ciclo `c20261123T121302Z-gh45172042466`

- **data_version:** `v001-a59ff626` (lotes novos: lote_001.csv)
- **Features eleitas (10):** euribor3m, month, cons.conf.idx, nr.employed, emp.var.rate, faixa_etaria, cons.price.idx, job, education, contact

| Candidato | roc_auc validação | run_id |
|---|---|---|
| modelo_a · logistic_regression | 0.6658 | `884b726696854a3ab8117109a1e43464` |
| modelo_b · hist_gradient_boosting | 0.6479 | `e1b93e8f31ee456cbb853716547bf13e` |

- **Vencedor do ciclo:** logistic_regression
- **Vigente anterior:** nenhum
- **Decisão:** `promover` - Não há modelo vigente e o candidato supera o piso.
- **Vigente após o ciclo:** logistic_regression (ciclo `c20261123T121302Z-gh45172042466`, run `884b726696854a3ab8117109a1e43464`)
- **roc_auc do vigente no teste reservado:** 0.7164
- **Drift:** sem alerta
