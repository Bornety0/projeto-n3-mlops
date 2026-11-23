# Ciclo `c20261123T184307Z-gh45241187784`

- **data_version:** `v005-6563af51` (lotes novos: lote_005.csv)
- **Features eleitas (9):** euribor3m, contatado_antes, emp.var.rate, cons.price.idx, poutcome, contact, nr.employed, month, cons.conf.idx

| Candidato | roc_auc validação | run_id |
|---|---|---|
| modelo_a · logistic_regression | 0.7990 | `36e985b920f24206ae62ee4a8fa73a81` |
| modelo_b · hist_gradient_boosting | 0.8064 | `79adaef8a8c94ed088c17eaad45d6023` |

- **Vencedor do ciclo:** hist_gradient_boosting
- **Vigente anterior nesta validação:** 0.7971
- **Decisão:** `manter_vigente` - Ganho de +0.0093 < limiar 0.01.
- **Vigente após o ciclo:** hist_gradient_boosting (ciclo `c20261123T183614Z-gh45240022843`, run `b5dc8f68a52a44d4952e8171783d6c09`)
- **roc_auc do vigente no teste reservado:** 0.811
- **Drift:** ALERTA - nr.employed, euribor3m, emp.var.rate, cons.conf.idx, cons.price.idx, month, poutcome, previous, contatado_antes, pdays_dias, job, faixa_etaria, default, age, contact
