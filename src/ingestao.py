"""Entrada de lotes, versionamento dos dados (data_version) e validação de qualidade."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from src.config import caminho

COLUNAS_ORIGINAIS = [
    "age", "job", "marital", "education", "default", "housing", "loan", "contact", "month",
    "day_of_week", "duration", "campaign", "pdays", "previous", "poutcome", "emp.var.rate",
    "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed", "y",
]


class ErroQualidadeDados(Exception):
    """Lote reprovado na validação: o ciclo deve parar antes de qualquer treino."""


def listar_lotes(cfg: dict) -> list[str]:
    diretorio = caminho(cfg, cfg["dados"]["diretorio_lotes"])
    return sorted(p.name for p in diretorio.glob("lote_*.csv"))


def carregar_estado(cfg: dict) -> dict:
    arquivo = caminho(cfg, cfg["dados"]["estado_ingestao"])
    if not arquivo.exists():
        return {"lotes_incorporados": [], "versoes": []}
    return json.loads(arquivo.read_text(encoding="utf-8"))


def salvar_estado(cfg: dict, estado: dict) -> None:
    arquivo = caminho(cfg, cfg["dados"]["estado_ingestao"])
    arquivo.parent.mkdir(parents=True, exist_ok=True)
    arquivo.write_text(json.dumps(estado, indent=2, ensure_ascii=False), encoding="utf-8")


def selecionar_lotes_novos(cfg: dict, estado: dict) -> list[str]:
    ja_incorporados = set(estado.get("lotes_incorporados", []))
    pendentes = [lote for lote in listar_lotes(cfg) if lote not in ja_incorporados]
    return pendentes[: cfg["particionamento"]["lotes_por_ciclo"]]


def hash_arquivo(arquivo: Path) -> str:
    return hashlib.sha256(arquivo.read_bytes()).hexdigest()


def calcular_data_version(cfg: dict, lotes: list[str]) -> str:
    """Versão determinística: muda se e somente se o conjunto/conteúdo dos lotes mudar."""
    diretorio = caminho(cfg, cfg["dados"]["diretorio_lotes"])
    assinatura = "|".join(f"{lote}:{hash_arquivo(diretorio / lote)}" for lote in sorted(lotes))
    resumo = hashlib.sha256(assinatura.encode("utf-8")).hexdigest()[:8]
    return f"v{len(lotes):03d}-{resumo}"


def ler_lote(cfg: dict, nome: str) -> pd.DataFrame:
    df = pd.read_csv(caminho(cfg, cfg["dados"]["diretorio_lotes"]) / nome)
    df["lote"] = nome.removesuffix(".csv")
    return df


def montar_pool(cfg: dict, lotes: list[str]) -> tuple[pd.DataFrame, int]:
    """Concatena os lotes sem duplicar registros já incorporados (chave: registro_id)."""
    if not lotes:
        raise ErroQualidadeDados("Nenhum lote disponível para montar o conjunto de dados.")
    id_col = cfg["dados"]["id"]
    pool = pd.concat([ler_lote(cfg, lote) for lote in sorted(lotes)], ignore_index=True)
    antes = len(pool)
    pool = pool.drop_duplicates(subset=id_col, keep="first").sort_values(id_col).reset_index(drop=True)
    return pool, antes - len(pool)


def ler_teste(cfg: dict) -> pd.DataFrame:
    return pd.read_csv(caminho(cfg, cfg["dados"]["arquivo_teste"]))


def validar_qualidade(df: pd.DataFrame, cfg: dict, nome: str = "dados") -> list[dict]:
    """Checagens de contrato do dado bruto; cada item registra se passou e o detalhe."""
    regras = cfg["qualidade_dados"]
    alvo = cfg["dados"]["alvo"]
    id_col = cfg["dados"]["id"]
    checagens: list[dict] = []

    def checar(nome_regra: str, ok: bool, detalhe: str) -> None:
        checagens.append({"conjunto": nome, "regra": nome_regra, "ok": bool(ok), "detalhe": detalhe})

    faltantes = sorted(set(COLUNAS_ORIGINAIS + [id_col]) - set(df.columns))
    checar("colunas_esperadas", not faltantes, f"faltantes={faltantes}")
    checar("nao_vazio", len(df) > 0, f"linhas={len(df)}")
    if faltantes or df.empty:
        return checagens

    valores_alvo = sorted(df[alvo].dropna().unique().tolist())
    checar("alvo_binario", set(valores_alvo) <= {"yes", "no"}, f"valores={valores_alvo}")
    checar("id_unico", df[id_col].is_unique, f"duplicados={int(df[id_col].duplicated().sum())}")
    idade_ok = df["age"].between(regras["idade_min"], regras["idade_max"]).all()
    checar("idade_no_intervalo", idade_ok, f"min={df['age'].min()} max={df['age'].max()}")
    campanha_ok = df["campaign"].between(1, regras["campaign_max"]).all()
    checar("campaign_no_intervalo", campanha_ok, f"min={df['campaign'].min()} max={df['campaign'].max()}")
    fracao_ausentes = df[COLUNAS_ORIGINAIS].isna().mean()
    pior = fracao_ausentes.idxmax()
    checar(
        "ausentes_toleraveis",
        fracao_ausentes.max() <= regras["max_fracao_ausentes"],
        f"maior fração={fracao_ausentes.max():.3f} ({pior})",
    )
    return checagens


def garantir_qualidade(checagens: list[dict]) -> None:
    reprovadas = [c for c in checagens if not c["ok"]]
    if reprovadas:
        detalhes = "; ".join(f"{c['conjunto']}.{c['regra']}: {c['detalhe']}" for c in reprovadas)
        raise ErroQualidadeDados("Validação de dados reprovada -> " + detalhes)
