# Ciclo `c20261123T182253Z-gh45237612466`

- **data_version:** `v002-86b526b7` (lotes novos: lote_002.csv)
- **Features eleitas (10):** euribor3m, month, cons.conf.idx, emp.var.rate, education, nr.employed, job, cons.price.idx, poutcome, contact

| Candidato | roc_auc validação | run_id |
|---|---|---|
| modelo_a · logistic_regression | 0.7008 | `7f5e8b5bc5314b9c987ee0952b561de1` |
| modelo_b · hist_gradient_boosting | 0.7073 | `bbb0fc7e3598463680ee838286fd5414` |

- **Vencedor do ciclo:** hist_gradient_boosting
- **Vigente anterior nesta validação:** 0.6947
- **Decisão:** `promover` - Ganho de +0.0126 >= limiar 0.01.
- **Vigente após o ciclo:** hist_gradient_boosting (ciclo `c20261123T182253Z-gh45237612466`, run `bbb0fc7e3598463680ee838286fd5414`)
- **roc_auc do vigente no teste reservado:** 0.644
- **Drift:** ALERTA - emp.var.rate, nr.employed, cons.conf.idx, cons.price.idx, euribor3m, month, pdays_dias, contact, poutcome
