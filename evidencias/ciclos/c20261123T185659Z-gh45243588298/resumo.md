# Ciclo `c20261123T185659Z-gh45243588298`

- **data_version:** `v005-6563af51` (lotes novos: nenhum)
- **Features eleitas (9):** euribor3m, contatado_antes, emp.var.rate, cons.price.idx, poutcome, contact, nr.employed, month, cons.conf.idx

| Candidato | roc_auc validação | run_id |
|---|---|---|
| modelo_a · logistic_regression | 0.7990 | `9a991687171146b0a36ff6d5962cf497` |
| modelo_b · hist_gradient_boosting | 0.8064 | `07bbc079709d4c02a58c488cfbc7fdfe` |

- **Vencedor do ciclo:** hist_gradient_boosting
- **Vigente anterior nesta validação:** 0.7971
- **Decisão:** `manter_vigente` - Ganho de +0.0093 < limiar 0.01.
- **Vigente após o ciclo:** hist_gradient_boosting (ciclo `c20261123T183614Z-gh45240022843`, run `b5dc8f68a52a44d4952e8171783d6c09`)
- **roc_auc do vigente no teste reservado:** 0.811
- **Drift:** sem alerta
