"""Critério 10 - monitoramento de drift entre o lote novo e os dados já incorporados.

Usa o PSI (Population Stability Index): soma de (p_novo - p_ref) * ln(p_novo / p_ref)
sobre faixas (numéricas, cortes pelos quantis da referência) ou categorias.
Faixas usuais: < 0,10 estável; 0,10-0,20 atenção; >= 0,20 mudança relevante.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-4


def _psi(p_ref: np.ndarray, p_novo: np.ndarray) -> float:
    p_ref = np.clip(p_ref, EPS, None)
    p_novo = np.clip(p_novo, EPS, None)
    return float(np.sum((p_novo - p_ref) * np.log(p_novo / p_ref)))


def psi_numerico(referencia: pd.Series, novo: pd.Series, n_bins: int = 10) -> float:
    referencia, novo = referencia.dropna(), novo.dropna()
    if referencia.empty or novo.empty:
        return 0.0
    cortes = np.unique(np.quantile(referencia, np.linspace(0, 1, n_bins + 1)))
    if len(cortes) < 2:
        return _psi(np.array([1.0]), np.array([1.0]))
    cortes[0], cortes[-1] = -np.inf, np.inf
    p_ref = np.histogram(referencia, bins=cortes)[0] / len(referencia)
    p_novo = np.histogram(novo, bins=cortes)[0] / len(novo)
    return _psi(p_ref, p_novo)


def psi_categorico(referencia: pd.Series, novo: pd.Series) -> float:
    categorias = sorted(set(referencia.dropna()) | set(novo.dropna()))
    p_ref = referencia.value_counts(normalize=True).reindex(categorias, fill_value=0).to_numpy()
    p_novo = novo.value_counts(normalize=True).reindex(categorias, fill_value=0).to_numpy()
    return _psi(p_ref, p_novo)


def relatorio_drift(
    referencia: pd.DataFrame, novo: pd.DataFrame, numericas: list[str], categoricas: list[str], regras: dict
) -> tuple[pd.DataFrame, dict]:
    linhas = []
    for coluna in numericas:
        linhas.append({"feature": coluna, "tipo": "numerica", "psi": psi_numerico(referencia[coluna], novo[coluna], regras["n_bins"])})
    for coluna in categoricas:
        linhas.append({"feature": coluna, "tipo": "categorica", "psi": psi_categorico(referencia[coluna], novo[coluna])})

    tabela = pd.DataFrame(linhas).sort_values("psi", ascending=False).reset_index(drop=True)
    tabela["psi"] = tabela["psi"].round(4)
    tabela["nivel"] = np.select(
        [tabela["psi"] >= regras["psi_alerta"], tabela["psi"] >= regras["psi_atencao"]],
        ["alerta", "atencao"],
        default="estavel",
    )
    em_alerta = tabela.loc[tabela["nivel"] == "alerta", "feature"].tolist()
    resumo = {
        "alerta": len(em_alerta) >= regras["min_features_alerta"],
        "features_em_alerta": em_alerta,
        "n_features_em_alerta": len(em_alerta),
        "psi_maximo": float(tabela["psi"].max()) if not tabela.empty else 0.0,
        "limiar_psi": regras["psi_alerta"],
        "min_features_alerta": regras["min_features_alerta"],
    }
    return tabela, resumo


def markdown_drift(cycle_id: str, lotes_novos: list[str], tabela: pd.DataFrame, resumo: dict, taxa_ref: float, taxa_novo: float) -> str:
    estado = "ALERTA DE DRIFT" if resumo["alerta"] else "Sem drift relevante"
    linhas = [
        f"## {estado} - ciclo `{cycle_id}`",
        "",
        f"Lote(s) avaliado(s): {', '.join(lotes_novos)}",
        f"Taxa de conversão (alvo): referência {taxa_ref:.2%} -> lote novo {taxa_novo:.2%}",
        f"Atributos com PSI >= {resumo['limiar_psi']}: {resumo['n_features_em_alerta']} "
        f"(alerta a partir de {resumo['min_features_alerta']})",
        "",
        "| feature | tipo | PSI | nível |",
        "|---|---|---|---|",
    ]
    for _, linha in tabela.head(12).iterrows():
        linhas.append(f"| {linha['feature']} | {linha['tipo']} | {linha['psi']:.4f} | {linha['nivel']} |")
    return "\n".join(linhas) + "\n"
