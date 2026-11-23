"""Diagnóstico, limpeza, engenharia de atributos e pré-processamento (sem vazamento)."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAS_ORIGINAIS = [
    "job", "marital", "education", "default", "housing", "loan", "contact", "month", "day_of_week", "poutcome",
]
NUMERICAS_ORIGINAIS = [
    "age", "campaign", "pdays", "previous", "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m",
    "nr.employed",
]
COLUNAS_CONTROLE = ["registro_id", "lote"]
PDAYS_SEM_CONTATO = 999


def diagnosticar(df: pd.DataFrame, alvo: str = "y") -> dict:
    """Retrato do dado: ausentes, 'unknown', duplicatas de conteúdo, outliers (IQR) e balanço do alvo."""
    conteudo = [c for c in df.columns if c not in COLUNAS_CONTROLE]
    numericas = [c for c in NUMERICAS_ORIGINAIS + ["duration"] if c in df.columns]
    categoricas = [c for c in CATEGORICAS_ORIGINAIS if c in df.columns]

    outliers = {}
    for coluna in numericas:
        q1, q3 = df[coluna].quantile([0.25, 0.75])
        iqr = q3 - q1
        fora = (df[coluna] < q1 - 1.5 * iqr) | (df[coluna] > q3 + 1.5 * iqr)
        outliers[coluna] = int(fora.sum())

    alvo_serie = df[alvo]
    taxa_positiva = float((alvo_serie == "yes").mean()) if alvo_serie.dtype == object else float(alvo_serie.mean())
    return {
        "linhas": int(len(df)),
        "colunas": int(df.shape[1]),
        "duplicatas_conteudo": int(df.duplicated(subset=conteudo).sum()),
        "ausentes": {c: int(v) for c, v in df.isna().sum().items() if v},
        "unknown": {c: int((df[c] == "unknown").sum()) for c in categoricas if (df[c] == "unknown").any()},
        "outliers_iqr": outliers,
        "pdays_sem_contato": int((df["pdays"] == PDAYS_SEM_CONTATO).sum()) if "pdays" in df else None,
        "taxa_positiva": round(taxa_positiva, 4),
    }


def limpar(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Tratamentos sem estado (não aprendem nada dos dados), portanto seguros antes do split.

    'unknown' é mantido como categoria: a ausência de informação é ela mesma informativa
    (ex.: 'default' = unknown está associado a perfil de risco diferente).
    """
    alvo = cfg["dados"]["alvo"]
    saida = df.copy()
    for coluna in CATEGORICAS_ORIGINAIS:
        saida[coluna] = saida[coluna].astype(str).str.strip().str.lower()

    # Duplicata = registro idêntico em TODAS as colunas originais. Comparar depois de remover
    # 'duration' descartaria contatos legítimos de clientes com o mesmo perfil no mesmo dia.
    conteudo = [c for c in saida.columns if c not in COLUNAS_CONTROLE]
    antes = len(saida)
    saida = saida.drop_duplicates(subset=conteudo, keep="first")

    removidas = [c for c in cfg["dados"]["colunas_removidas"] if c in saida.columns]
    saida = saida.drop(columns=removidas)
    saida[alvo] = (saida[alvo] == cfg["dados"]["classe_positiva"]).astype(int)
    relatorio = {"colunas_removidas": removidas, "duplicatas_removidas": int(antes - len(saida))}
    return saida.reset_index(drop=True), relatorio


def engenharia_atributos(df: pd.DataFrame) -> pd.DataFrame:
    """Atributos derivados linha a linha (sem estatísticas do conjunto -> sem vazamento)."""
    saida = df.copy()
    contatado = saida["pdays"] != PDAYS_SEM_CONTATO
    # pdays=999 é um código ("nunca contatado"), não 999 dias: vira flag + valor ausente.
    saida["contatado_antes"] = np.where(contatado, "sim", "nao")
    saida["pdays_dias"] = saida["pdays"].where(contatado)
    # campaign tem cauda longa (até dezenas de ligações); log1p reduz o peso dos extremos.
    saida["campaign_log"] = np.log1p(saida["campaign"])
    saida["qtd_desconhecidos"] = (saida[CATEGORICAS_ORIGINAIS] == "unknown").sum(axis=1)
    saida["faixa_etaria"] = pd.cut(
        saida["age"], bins=[0, 25, 35, 45, 55, 65, 200], labels=["ate25", "26-35", "36-45", "46-55", "56-65", "66+"]
    ).astype(str)
    return saida.drop(columns=["pdays", "campaign"])


def pertence_validacao(registro_id, fracao_validacao: float) -> bool:
    digest = hashlib.sha256(str(int(registro_id)).encode("utf-8")).hexdigest()
    return int(digest, 16) % 1000 < round(fracao_validacao * 1000)


def dividir_por_hash(df: pd.DataFrame, fracao_validacao: float, id_col: str = "registro_id"):
    """Partição treino/validação estável entre ciclos.

    Um registro cai sempre do mesmo lado, então o modelo vigente (treinado num ciclo
    anterior) nunca viu a validação atual. Um corte temporal (últimos 20%) foi testado
    e descartado: cada fatia recente é um regime econômico distinto e a validação
    media só a ruptura (AUC ~0,5 para qualquer modelo).
    """
    ordenado = df.sort_values(id_col).reset_index(drop=True)
    na_validacao = ordenado[id_col].map(lambda i: pertence_validacao(i, fracao_validacao))
    return ordenado[~na_validacao].copy(), ordenado[na_validacao].copy()


def construir_preprocessador(numericas: list[str], categoricas: list[str]) -> ColumnTransformer:
    transformadores = []
    if numericas:
        transformadores.append((
            "num",
            Pipeline([
                ("imputar", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("escalar", StandardScaler()),
            ]),
            numericas,
        ))
    if categoricas:
        transformadores.append((
            "cat",
            Pipeline([
                ("imputar", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            categoricas,
        ))
    return ColumnTransformer(transformadores, remainder="drop", verbose_feature_names_out=False)


def mapear_origem(preprocessador: ColumnTransformer, numericas: list[str], categoricas: list[str]) -> list[str]:
    """Para cada coluna transformada, o atributo original de onde ela veio (one-hot -> coluna mãe)."""
    origem = list(numericas)
    if categoricas:
        encoder = preprocessador.named_transformers_["cat"].named_steps["onehot"]
        for coluna, categorias in zip(categoricas, encoder.categories_):
            origem.extend([coluna] * len(categorias))
    return origem
