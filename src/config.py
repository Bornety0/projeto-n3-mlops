"""Carregamento da configuração e resolução de caminhos a partir da raiz do projeto."""

from __future__ import annotations

from pathlib import Path

import yaml


def raiz_projeto(inicio: str | Path | None = None) -> Path:
    """Sobe a árvore de diretórios até achar config/config.yaml.

    Os notebooks rodam com o diretório de trabalho em notebooks/ (nbconvert) ou
    na raiz (execução interativa); procurar a raiz evita caminhos relativos frágeis.
    """
    atual = Path(inicio or Path.cwd()).resolve()
    for candidato in [atual, *atual.parents]:
        if (candidato / "config" / "config.yaml").exists():
            return candidato
    raise FileNotFoundError("config/config.yaml não encontrado a partir de " + str(atual))


def carregar_config(raiz: str | Path | None = None) -> dict:
    raiz = Path(raiz).resolve() if raiz else raiz_projeto()
    with open(raiz / "config" / "config.yaml", encoding="utf-8") as arquivo:
        cfg = yaml.safe_load(arquivo)
    cfg["_raiz"] = str(raiz)
    return cfg


def caminho(cfg: dict, relativo: str | Path) -> Path:
    return Path(cfg["_raiz"]) / relativo
