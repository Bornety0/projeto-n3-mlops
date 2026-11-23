# Ciclo `c20261123T182934Z-gh45238828713`

- **data_version:** `v003-48d4afe3` (lotes novos: lote_003.csv)
- **Features eleitas (10):** euribor3m, month, nr.employed, cons.conf.idx, emp.var.rate, faixa_etaria, contact, cons.price.idx, job, education

| Candidato | roc_auc validação | run_id |
|---|---|---|
| modelo_a · logistic_regression | 0.6839 | `b54339e0336744bf999def8df5955038` |
| modelo_b · hist_gradient_boosting | 0.6887 | `36d2b11cd28e46039bf724ae89d876ec` |

- **Vencedor do ciclo:** hist_gradient_boosting
- **Vigente anterior nesta validação:** 0.6890
- **Decisão:** `manter_vigente` - Ganho de -0.0003 < limiar 0.01.
- **Vigente após o ciclo:** hist_gradient_boosting (ciclo `c20261123T182253Z-gh45237612466`, run `bbb0fc7e3598463680ee838286fd5414`)
- **roc_auc do vigente no teste reservado:** 0.644
- **Drift:** ALERTA - cons.conf.idx, cons.price.idx, emp.var.rate, nr.employed, euribor3m, month, pdays_dias, contact, poutcome, age, faixa_etaria
