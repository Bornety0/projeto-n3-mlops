# Ciclo `c20261123T183614Z-gh45240022843`

- **data_version:** `v004-d9278016` (lotes novos: lote_004.csv)
- **Features eleitas (10):** euribor3m, emp.var.rate, cons.price.idx, contatado_antes, contact, nr.employed, month, poutcome, cons.conf.idx, job

| Candidato | roc_auc validação | run_id |
|---|---|---|
| modelo_a · logistic_regression | 0.7364 | `01f987d99957490aa8562dbe9e1147f7` |
| modelo_b · hist_gradient_boosting | 0.7401 | `b5dc8f68a52a44d4952e8171783d6c09` |

- **Vencedor do ciclo:** hist_gradient_boosting
- **Vigente anterior nesta validação:** 0.6550
- **Decisão:** `promover` - Ganho de +0.0851 >= limiar 0.01.
- **Vigente após o ciclo:** hist_gradient_boosting (ciclo `c20261123T183614Z-gh45240022843`, run `b5dc8f68a52a44d4952e8171783d6c09`)
- **roc_auc do vigente no teste reservado:** 0.811
- **Drift:** ALERTA - cons.price.idx, euribor3m, emp.var.rate, nr.employed, cons.conf.idx, month, pdays_dias, contact, poutcome, default, job, contatado_antes, faixa_etaria
