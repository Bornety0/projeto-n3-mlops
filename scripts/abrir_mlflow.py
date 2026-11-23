"""Abre a interface do MLflow com o histórico versionado em state/.

Uso:
    python scripts/abrir_mlflow.py            # http://127.0.0.1:5000
    python scripts/abrir_mlflow.py --porta 5001

Antes de subir a interface, os caminhos de artefatos gravados por outra máquina
(ex.: o runner do GitHub Actions) são reescritos para esta cópia do repositório.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src.config import caminho, carregar_config  # noqa: E402
from src.rastreamento import configurar_mlflow  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--porta", default="5000")
    args = parser.parse_args()

    cfg = carregar_config(RAIZ)
    configurar_mlflow(cfg)
    banco = caminho(cfg, cfg["mlflow"]["tracking_db"])
    print(f"MLflow em http://127.0.0.1:{args.porta} (experimento '{cfg['mlflow']['experimento']}')")
    subprocess.run(
        [sys.executable, "-m", "mlflow", "ui", "--backend-store-uri", f"sqlite:///{banco}", "--port", args.porta],
        check=True,
    )


if __name__ == "__main__":
    main()
