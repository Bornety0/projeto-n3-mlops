"""Prepara os dados uma única vez: baixa o Bank Marketing (UCI), confere o SHA-256,
numera os registros na ordem original (cronológica) e gera:

- data/holdout/teste.csv      -> 10% estratificado, reservado para a avaliação final;
- data/raw/lotes/lote_NNN.csv -> fila de lotes que o pipeline incorpora, um por ciclo.

Uso:
    python scripts/preparar_dados.py                      # baixa da UCI
    python scripts/preparar_dados.py --origem caminho.csv # usa um arquivo local
"""

from __future__ import annotations

import argparse
import hashlib
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src.config import caminho, carregar_config  # noqa: E402

ARQUIVO_NO_ZIP = "bank-additional/bank-additional-full.csv"


def baixar_csv(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as resposta:
        externo = zipfile.ZipFile(io.BytesIO(resposta.read()))
    interno = zipfile.ZipFile(io.BytesIO(externo.read("bank-additional.zip")))
    return interno.read(ARQUIVO_NO_ZIP)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--origem", help="CSV bank-additional-full.csv já baixado")
    args = parser.parse_args()

    cfg = carregar_config(RAIZ)
    conteudo = Path(args.origem).read_bytes() if args.origem else baixar_csv(cfg["dados"]["url"])
    sha = hashlib.sha256(conteudo).hexdigest()
    if sha != cfg["dados"]["sha256_original"]:
        raise SystemExit(f"SHA-256 inesperado ({sha}); o arquivo de origem não é o esperado.")

    df = pd.read_csv(io.BytesIO(conteudo), sep=cfg["dados"]["separador"])
    df.insert(0, cfg["dados"]["id"], np.arange(1, len(df) + 1))

    part = cfg["particionamento"]
    # O teste é amostrado ao longo de todo o período: o fim da série (2010) tem outro regime
    # (taxa de 'yes' ~46%) e um teste só com ele mediria apenas essa ruptura.
    disponivel, teste = train_test_split(
        df, test_size=part["fracao_teste"], stratify=df[cfg["dados"]["alvo"]], random_state=cfg["projeto"]["seed"]
    )
    disponivel, teste = disponivel.sort_values(cfg["dados"]["id"]), teste.sort_values(cfg["dados"]["id"])
    n_inicial = int(round(len(disponivel) * part["fracao_lote_inicial"]))
    restante = disponivel.iloc[n_inicial:]
    cortes = np.linspace(0, len(restante), part["n_lotes_adicionais"] + 1).round().astype(int)
    lotes = [disponivel.iloc[:n_inicial]] + [restante.iloc[a:b] for a, b in zip(cortes[:-1], cortes[1:])]

    destino_lotes = caminho(cfg, cfg["dados"]["diretorio_lotes"])
    destino_lotes.mkdir(parents=True, exist_ok=True)
    for antigo in destino_lotes.glob("lote_*.csv"):
        antigo.unlink()
    arquivo_teste = caminho(cfg, cfg["dados"]["arquivo_teste"])
    arquivo_teste.parent.mkdir(parents=True, exist_ok=True)
    teste.to_csv(arquivo_teste, index=False)

    print(f"Origem: {len(df)} registros, SHA-256 ok")
    print(f"teste.csv: {len(teste)} registros, taxa 'yes' = {(teste['y'] == 'yes').mean():.2%}")
    for i, lote in enumerate(lotes, start=1):
        nome = f"lote_{i:03d}.csv"
        lote.to_csv(destino_lotes / nome, index=False)
        print(f"{nome}: {len(lote)} registros, ids {lote.iloc[0, 0]}-{lote.iloc[-1, 0]}, taxa 'yes' = {(lote['y'] == 'yes').mean():.2%}")


if __name__ == "__main__":
    main()
