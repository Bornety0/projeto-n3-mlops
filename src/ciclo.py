"""Identificação do ciclo (cycle_id) e contexto compartilhado entre os notebooks."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from src.config import caminho

PONTEIRO_CICLO = "state/.ciclo_corrente"


def novo_cycle_id() -> str:
    instante = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_github = os.environ.get("GITHUB_RUN_ID")
    origem = f"gh{run_github}" if run_github else "local"
    return f"c{instante}-{origem}"


def obter_cycle_id(cfg: dict, criar: bool = False) -> str:
    """Retorna o cycle_id do ciclo em andamento.

    No workflow o id chega pela variável CYCLE_ID; em execução interativa o
    notebook 01 cria um id novo e os demais o leem do ponteiro em state/.
    """
    ponteiro = caminho(cfg, PONTEIRO_CICLO)
    cycle_id = os.environ.get("CYCLE_ID")
    if not cycle_id and criar:
        cycle_id = novo_cycle_id()
    if cycle_id:
        if criar:
            ponteiro.parent.mkdir(parents=True, exist_ok=True)
            ponteiro.write_text(cycle_id, encoding="utf-8")
        return cycle_id
    if ponteiro.exists():
        return ponteiro.read_text(encoding="utf-8").strip()
    raise RuntimeError("Nenhum ciclo em andamento: execute o notebook 01 primeiro.")


def diretorio_ciclo(cfg: dict, cycle_id: str) -> Path:
    destino = caminho(cfg, cfg["saida"]["diretorio_evidencias"]) / cycle_id
    destino.mkdir(parents=True, exist_ok=True)
    return destino


def carregar_contexto(cfg: dict, cycle_id: str) -> dict:
    arquivo = diretorio_ciclo(cfg, cycle_id) / "contexto.json"
    if not arquivo.exists():
        raise RuntimeError(f"Contexto do ciclo {cycle_id} não encontrado: execute o notebook 01 primeiro.")
    return json.loads(arquivo.read_text(encoding="utf-8"))


def atualizar_contexto(cfg: dict, cycle_id: str, /, **campos) -> dict:
    arquivo = diretorio_ciclo(cfg, cycle_id) / "contexto.json"
    contexto = json.loads(arquivo.read_text(encoding="utf-8")) if arquivo.exists() else {}
    contexto.update(campos)
    salvar_json(arquivo, contexto)
    return contexto


def salvar_json(arquivo: Path, conteudo) -> None:
    arquivo.parent.mkdir(parents=True, exist_ok=True)
    arquivo.write_text(json.dumps(conteudo, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
