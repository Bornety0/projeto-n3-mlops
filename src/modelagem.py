"""Construção, treino e avaliação dos algoritmos candidatos (Modelo A e Modelo B)."""

from __future__ import annotations

import importlib
import time

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.features import separar_tipos
from src.preprocessamento import construir_preprocessador


def instanciar_estimador(especificacao: dict):
    modulo, _, classe = especificacao["classe"].rpartition(".")
    return getattr(importlib.import_module(modulo), classe)(**especificacao.get("parametros", {}))


def construir_pipeline(especificacao: dict, eleitas: list[str], cfg: dict) -> Pipeline:
    """Pré-processamento + estimador num único objeto: o modelo serializado carrega
    junto as features eleitas e os transformadores ajustados, garantindo previsões consistentes."""
    numericas, categoricas = separar_tipos(eleitas, cfg)
    return Pipeline([
        ("preprocessamento", construir_preprocessador(numericas, categoricas)),
        ("modelo", instanciar_estimador(especificacao)),
    ])


def colunas_entrada(cfg: dict) -> list[str]:
    return cfg["features"]["numericas"] + cfg["features"]["categoricas"]


def avaliar(modelo, X: pd.DataFrame, y: pd.Series) -> dict:
    predito = modelo.predict(X)
    metricas = {
        "f1_macro": f1_score(y, predito, average="macro"),
        "balanced_accuracy": balanced_accuracy_score(y, predito),
        "accuracy": accuracy_score(y, predito),
        "precision_pos": precision_score(y, predito, zero_division=0),
        "recall_pos": recall_score(y, predito, zero_division=0),
        "n": len(y),
        "taxa_positiva_real": float(np.mean(y)),
    }
    if hasattr(modelo, "predict_proba") and len(np.unique(y)) > 1:
        metricas["roc_auc"] = roc_auc_score(y, modelo.predict_proba(X)[:, 1])
    return {k: (round(float(v), 4) if isinstance(v, float) else v) for k, v in metricas.items()}


def treinar_candidato(especificacao: dict, eleitas: list[str], treino: pd.DataFrame, validacao: pd.DataFrame, cfg: dict):
    alvo = cfg["dados"]["alvo"]
    X_cols = colunas_entrada(cfg)
    pipeline = construir_pipeline(especificacao, eleitas, cfg)
    inicio = time.perf_counter()
    pipeline.fit(treino[X_cols], treino[alvo])
    duracao = time.perf_counter() - inicio
    return {
        "nome": especificacao["nome"],
        "classe": especificacao["classe"],
        "parametros": especificacao.get("parametros", {}),
        "pipeline": pipeline,
        "metricas_treino": avaliar(pipeline, treino[X_cols], treino[alvo]),
        "metricas_validacao": avaliar(pipeline, validacao[X_cols], validacao[alvo]),
        "tempo_treino_s": round(duracao, 2),
    }


def avaliar_baseline(treino: pd.DataFrame, validacao: pd.DataFrame, cfg: dict) -> dict:
    """Referência trivial (sempre a classe majoritária) para dar escala às métricas."""
    alvo = cfg["dados"]["alvo"]
    X_cols = colunas_entrada(cfg)
    dummy = DummyClassifier(strategy="most_frequent").fit(treino[X_cols], treino[alvo])
    return avaliar(dummy, validacao[X_cols], validacao[alvo])
