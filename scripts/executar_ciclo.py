"""Executa um ciclo completo localmente, na mesma ordem do workflow agendado.

Uso:
    python scripts/executar_ciclo.py            # notebooks 01 -> 05
    python scripts/executar_ciclo.py --ate 04   # para no notebook 04

Os notebooks executados (com saídas) ficam em evidencias/ciclos/<cycle_id>/notebooks/.
Qualquer erro interrompe a sequência, como no GitHub Actions.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src.ciclo import novo_cycle_id  # noqa: E402

NOTEBOOKS = ["01_etl", "02_votacao_features", "03_treino_competicao", "04_selecao_promocao", "05_previsoes"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ate", default="05", help="último notebook a executar (01-05)")
    args = parser.parse_args()

    cycle_id = os.environ.get("CYCLE_ID") or novo_cycle_id()
    ambiente = {**os.environ, "CYCLE_ID": cycle_id}
    saida = RAIZ / "evidencias" / "ciclos" / cycle_id / "notebooks"
    print(f"cycle_id = {cycle_id}")

    for notebook in NOTEBOOKS:
        if notebook[:2] > args.ate:
            break
        print(f"-> executando {notebook}.ipynb")
        subprocess.run(
            [
                sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
                f"notebooks/{notebook}.ipynb", "--output-dir", str(saida), "--output", f"{notebook}_executado",
                "--ExecutePreprocessor.timeout=1800",
            ],
            cwd=RAIZ, env=ambiente, check=True,
        )
    print(f"Ciclo {cycle_id} concluído. Evidências em {saida.parent}")


if __name__ == "__main__":
    main()
