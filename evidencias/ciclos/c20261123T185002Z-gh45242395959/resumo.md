# Ciclo `c20261123T185002Z-gh45242395959`

- **data_version:** `v005-6563af51` (lotes novos: nenhum)
- **Features eleitas (9):** euribor3m, contatado_antes, emp.var.rate, cons.price.idx, poutcome, contact, nr.employed, month, cons.conf.idx

| Candidato | roc_auc validação | run_id |
|---|---|---|
| modelo_a · logistic_regression | 0.7990 | `c21356f12cd84703ba469d882612a1f6` |
| modelo_b · hist_gradient_boosting | 0.8064 | `90aa5d22a2d848d19412708c07a02940` |

- **Vencedor do ciclo:** hist_gradient_boosting
- **Vigente anterior nesta validação:** 0.7971
- **Decisão:** `manter_vigente` - Ganho de +0.0093 < limiar 0.01.
- **Vigente após o ciclo:** hist_gradient_boosting (ciclo `c20261123T183614Z-gh45240022843`, run `b5dc8f68a52a44d4952e8171783d6c09`)
- **roc_auc do vigente no teste reservado:** 0.811
- **Drift:** sem alerta
