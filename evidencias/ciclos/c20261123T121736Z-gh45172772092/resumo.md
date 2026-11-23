# Ciclo `c20261123T121736Z-gh45172772092`

- **data_version:** `v002-86b526b7` (lotes novos: lote_002.csv)
- **Features eleitas (10):** euribor3m, month, cons.conf.idx, emp.var.rate, education, nr.employed, job, cons.price.idx, poutcome, contact

| Candidato | roc_auc validação | run_id |
|---|---|---|
| modelo_a · logistic_regression | 0.7008 | `cca726e6a7b24869a3360872dfb7cd81` |
| modelo_b · hist_gradient_boosting | 0.7073 | `9ed8843ee01a4e8cb38cc2ad276b03dc` |

- **Vencedor do ciclo:** hist_gradient_boosting
- **Vigente anterior nesta validação:** 0.6947
- **Decisão:** `promover` - Ganho de +0.0126 >= limiar 0.01.
- **Vigente após o ciclo:** hist_gradient_boosting (ciclo `c20261123T121736Z-gh45172772092`, run `9ed8843ee01a4e8cb38cc2ad276b03dc`)
- **roc_auc do vigente no teste reservado:** 0.644
- **Drift:** ALERTA - emp.var.rate, nr.employed, cons.conf.idx, cons.price.idx, euribor3m, month, pdays_dias, contact, poutcome
