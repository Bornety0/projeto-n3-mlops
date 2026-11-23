"""Três métodos de importância/seleção de atributos e a votação que elege o conjunto do ciclo.

Os métodos trabalham sobre a matriz pré-processada (one-hot), mas votam em
atributos ORIGINAIS: a importância das colunas one-hot é agregada de volta à
coluna mãe. Assim o conjunto eleito é interpretável e o pipeline do modelo pode
simplesmente selecionar colunas brutas antes de pré-processar.
"""

from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression

from src.preprocessamento import construir_preprocessador, mapear_origem


def _ranquear(pontuacao: pd.Series, top_k: int, maior_melhor: bool = True) -> pd.DataFrame:
    ordenada = pontuacao.sort_values(ascending=not maior_melhor, kind="mergesort")
    tabela = pd.DataFrame({"pontuacao": ordenada})
    tabela["posicao"] = np.arange(1, len(tabela) + 1)
    tabela["aprovada"] = tabela["posicao"] <= top_k
    tabela.index.name = "feature"
    return tabela


def metodo_selectkbest(Xt: np.ndarray, y: np.ndarray, origem: list[str], top_k: int, seed: int) -> pd.DataFrame:
    # Informação mútua captura relações não lineares e serve para colunas numéricas e one-hot.
    seletor = SelectKBest(score_func=partial(mutual_info_classif, random_state=seed), k="all").fit(Xt, y)
    pontuacao = pd.Series(seletor.scores_, index=origem).groupby(level=0).max()
    return _ranquear(pontuacao, top_k)


def metodo_random_forest(
    Xt: np.ndarray, y: np.ndarray, origem: list[str], top_k: int, seed: int, parametros: dict
) -> pd.DataFrame:
    floresta = RandomForestClassifier(class_weight="balanced", n_jobs=-1, random_state=seed, **parametros).fit(Xt, y)
    # Somar as importâncias das colunas one-hot devolve a importância total do atributo.
    pontuacao = pd.Series(floresta.feature_importances_, index=origem).groupby(level=0).sum()
    return _ranquear(pontuacao, top_k)


def metodo_rfe(Xt: np.ndarray, y: np.ndarray, origem: list[str], top_k: int, step: int) -> pd.DataFrame:
    estimador = LogisticRegression(max_iter=2000, class_weight="balanced")
    rfe = RFE(estimador, n_features_to_select=1, step=step).fit(Xt, y)
    # Ranking 1 = última coluna eliminada; o atributo herda a melhor posição de suas colunas.
    melhor_ranking = pd.Series(rfe.ranking_, index=origem).groupby(level=0).min()
    return _ranquear(-melhor_ranking.astype(float), top_k)


def votar(resultados: dict[str, pd.DataFrame], votos_minimos: int, min_features: int) -> tuple[pd.DataFrame, list[str]]:
    """Regra definida antes da comparação (config.yaml):

    1. cada método aprova seu top-k;
    2. é eleito o atributo com >= votos_minimos aprovações;
    3. se forem eleitos menos que min_features, completa-se pelos não eleitos
       com mais votos e, em empate, menor posição média nos três rankings.
    """
    tabela = pd.DataFrame(index=sorted(set().union(*(r.index for r in resultados.values()))))
    for metodo, resultado in resultados.items():
        tabela[f"posicao_{metodo}"] = resultado["posicao"]
        tabela[f"voto_{metodo}"] = resultado["aprovada"].astype(int)
    colunas_voto = [f"voto_{m}" for m in resultados]
    colunas_posicao = [f"posicao_{m}" for m in resultados]
    tabela["total_votos"] = tabela[colunas_voto].sum(axis=1)
    tabela["posicao_media"] = tabela[colunas_posicao].mean(axis=1).round(2)
    tabela = tabela.sort_values(["total_votos", "posicao_media"], ascending=[False, True], kind="mergesort")

    tabela["eleita"] = tabela["total_votos"] >= votos_minimos
    tabela["criterio"] = np.where(tabela["eleita"], "votos", "")
    faltam = min_features - int(tabela["eleita"].sum())
    if faltam > 0:
        complemento = tabela.index[~tabela["eleita"]][:faltam]
        tabela.loc[complemento, "eleita"] = True
        tabela.loc[complemento, "criterio"] = "desempate_ranking_medio"

    tabela.index.name = "feature"
    eleitas = tabela.index[tabela["eleita"]].tolist()
    return tabela, eleitas


def executar_votacao(treino: pd.DataFrame, cfg: dict) -> dict:
    """Ajusta pré-processamento e os 3 métodos SOMENTE no treino do ciclo."""
    numericas = cfg["features"]["numericas"]
    categoricas = cfg["features"]["categoricas"]
    votacao = cfg["votacao"]
    seed = cfg["projeto"]["seed"]
    top_k = votacao["top_k"]

    preprocessador = construir_preprocessador(numericas, categoricas)
    Xt = preprocessador.fit_transform(treino[numericas + categoricas])
    y = treino[cfg["dados"]["alvo"]].to_numpy()
    origem = mapear_origem(preprocessador, numericas, categoricas)

    resultados = {
        "selectkbest": metodo_selectkbest(Xt, y, origem, top_k, seed),
        "random_forest": metodo_random_forest(Xt, y, origem, top_k, seed, votacao["metodos"]["random_forest"]),
        "rfe": metodo_rfe(Xt, y, origem, top_k, votacao["metodos"]["rfe"]["step"]),
    }
    tabela, eleitas = votar(resultados, votacao["votos_minimos"], votacao["min_features"])
    return {
        "resultados": resultados,
        "tabela_votos": tabela,
        "eleitas": eleitas,
        "n_colunas_transformadas": int(Xt.shape[1]),
    }


def separar_tipos(eleitas: list[str], cfg: dict) -> tuple[list[str], list[str]]:
    numericas = [f for f in cfg["features"]["numericas"] if f in eleitas]
    categoricas = [f for f in cfg["features"]["categoricas"] if f in eleitas]
    return numericas, categoricas
