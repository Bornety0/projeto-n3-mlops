"""Configuração do MLflow com armazenamento versionado no próprio repositório.

O runner do GitHub Actions é descartado ao fim de cada execução; por isso o banco
SQLite e os artefatos ficam em state/ e são devolvidos ao repositório pelo workflow.
O MLflow grava caminhos absolutos dos artefatos; ao abrir o histórico em outra
máquina (ou outro runner), os caminhos são reescritos para a raiz atual.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

import mlflow

from src.config import caminho

COLUNAS_COM_CAMINHO = {
    "experiments": ["artifact_location"],
    "runs": ["artifact_uri"],
    "model_versions": ["source", "storage_location"],
}


def _rebasear_caminhos(banco: Path, artefatos: Path, artefatos_relativo: str) -> None:
    if not banco.exists():
        return
    marcador = "/" + artefatos_relativo.strip("/")
    novo_prefixo = artefatos.as_uri()
    with sqlite3.connect(banco) as conexao:
        tabelas = {linha[0] for linha in conexao.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "experiments" not in tabelas:
            return
        prefixos_antigos = set()
        for (local,) in conexao.execute("SELECT artifact_location FROM experiments"):
            if local and marcador in local:
                prefixos_antigos.add(local[: local.index(marcador) + len(marcador)])
        for antigo in prefixos_antigos - {novo_prefixo}:
            for tabela, colunas in COLUNAS_COM_CAMINHO.items():
                if tabela not in tabelas:
                    continue
                existentes = {linha[1] for linha in conexao.execute(f"PRAGMA table_info({tabela})")}
                for coluna in colunas:
                    if coluna in existentes:
                        conexao.execute(
                            f"UPDATE {tabela} SET {coluna} = REPLACE({coluna}, ?, ?) WHERE {coluna} LIKE ?",
                            (antigo, novo_prefixo, antigo + "%"),
                        )


def configurar_mlflow(cfg: dict) -> str:
    banco = caminho(cfg, cfg["mlflow"]["tracking_db"])
    artefatos = caminho(cfg, cfg["mlflow"]["artifacts_dir"])
    banco.parent.mkdir(parents=True, exist_ok=True)
    artefatos.mkdir(parents=True, exist_ok=True)
    _rebasear_caminhos(banco, artefatos, cfg["mlflow"]["artifacts_dir"])

    mlflow.set_tracking_uri(f"sqlite:///{banco}")
    nome = cfg["mlflow"]["experimento"]
    experimento = mlflow.get_experiment_by_name(nome)
    if experimento is None:
        mlflow.create_experiment(nome, artifact_location=artefatos.as_uri())
    mlflow.set_experiment(nome)
    return mlflow.get_experiment_by_name(nome).experiment_id


def tags_ciclo(contexto: dict, etapa: str) -> dict:
    return {"cycle_id": contexto["cycle_id"], "data_version": contexto["data_version"], "etapa": etapa}


def iniciar_run_ciclo(cfg: dict, contexto: dict) -> str:
    """Run 'pai' do ciclo: agrupa no MLflow as etapas executadas por notebooks diferentes."""
    configurar_mlflow(cfg)
    with mlflow.start_run(run_name=f"ciclo-{contexto['cycle_id']}", tags=tags_ciclo(contexto, "ciclo")) as run:
        mlflow.log_params({"lotes_novos": ",".join(contexto["lotes_novos"]) or "nenhum", "n_lotes": len(contexto["lotes_pool"])})
        return run.info.run_id


@contextmanager
def run_etapa(cfg: dict, contexto: dict, etapa: str, nome: str | None = None, tags: dict | None = None):
    configurar_mlflow(cfg)
    with mlflow.start_run(run_id=contexto["mlflow_parent_run_id"]):
        with mlflow.start_run(run_name=nome or etapa, nested=True, tags={**tags_ciclo(contexto, etapa), **(tags or {})}) as run:
            yield run


def marcar_ciclo(cfg: dict, contexto: dict, tags: dict) -> None:
    configurar_mlflow(cfg)
    with mlflow.start_run(run_id=contexto["mlflow_parent_run_id"]):
        mlflow.set_tags(tags)
