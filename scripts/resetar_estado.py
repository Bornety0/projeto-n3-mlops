"""Apaga o estado dos ciclos (MLflow, modelos, evidências) para recomeçar do lote_001.

Uso:
    python scripts/resetar_estado.py --confirmar

Útil depois de testes locais: o primeiro ciclo agendado no GitHub deve partir do zero.
Os lotes em data/raw/lotes e o teste em data/holdout NÃO são apagados.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]

ARQUIVOS = [
    "state/mlflow.db",
    "state/estado_ingestao.json",
    "state/historico_ciclos.csv",
    "state/.ciclo_corrente",
    "models/modelo_vigente.json",
    "data/processed/treino.csv",
    "data/processed/validacao.csv",
    "data/processed/teste.csv",
]
DIRETORIOS_COM_CONTEUDO = ["state/mlartifacts", "models/historico", "evidencias/ciclos"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirmar", action="store_true", help="necessário para apagar de fato")
    args = parser.parse_args()

    alvos = [RAIZ / a for a in ARQUIVOS if (RAIZ / a).exists()]
    for diretorio in DIRETORIOS_COM_CONTEUDO:
        if (RAIZ / diretorio).exists():
            alvos += [p for p in (RAIZ / diretorio).iterdir() if p.name != ".gitkeep"]
    if not alvos:
        print("Nada a apagar: o estado já está limpo.")
        return
    for alvo in alvos:
        print(("apagando " if args.confirmar else "seria apagado: ") + str(alvo.relative_to(RAIZ)))
        if args.confirmar:
            shutil.rmtree(alvo) if alvo.is_dir() else alvo.unlink()
    if not args.confirmar:
        print("\nNada foi apagado. Rode novamente com --confirmar.")


if __name__ == "__main__":
    main()
