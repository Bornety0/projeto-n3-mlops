"""Testes das regras de decisão do pipeline (votação, promoção, drift e partição).

Rodam no workflow antes dos notebooks: se uma regra quebrar, o ciclo não executa.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.drift import psi_categorico, psi_numerico  # noqa: E402
from src.features import votar  # noqa: E402
from src.preprocessamento import dividir_por_hash, engenharia_atributos  # noqa: E402
from src.promocao import MANTER, PROMOVER, SEM_MODELO, decidir_promocao, eleger_vencedor  # noqa: E402

REGRA = {"metrica_minima": 0.60, "melhoria_minima": 0.01}


def ranking(ordem, top_k):
    return pd.DataFrame(
        {"posicao": range(1, len(ordem) + 1), "aprovada": [i < top_k for i in range(len(ordem))]},
        index=pd.Index(ordem, name="feature"),
    )


def test_votacao_elege_quem_tem_dois_votos():
    resultados = {
        "m1": ranking(["a", "b", "c", "d"], 2),
        "m2": ranking(["a", "c", "b", "d"], 2),
        "m3": ranking(["b", "d", "a", "c"], 2),
    }
    tabela, eleitas = votar(resultados, votos_minimos=2, min_features=1)
    assert sorted(eleitas) == ["a", "b"]
    assert tabela.loc["a", "total_votos"] == 2 and tabela.loc["c", "total_votos"] == 1


def test_votacao_completa_ate_o_minimo_pelo_ranking_medio():
    resultados = {
        "m1": ranking(["a", "b", "c", "d"], 1),
        "m2": ranking(["b", "c", "a", "d"], 1),
        "m3": ranking(["c", "b", "a", "d"], 1),
    }
    tabela, eleitas = votar(resultados, votos_minimos=2, min_features=2)
    # ninguém tem 2 votos; b (posições 2,1,2) vence o desempate contra a e c
    assert eleitas[0] == "b" and len(eleitas) == 2
    assert set(tabela.loc[eleitas, "criterio"]) == {"desempate_ranking_medio"}


def test_primeiro_modelo_promovido_se_acima_do_piso():
    assert decidir_promocao(0.65, None, REGRA)["decisao"] == PROMOVER
    assert decidir_promocao(0.55, None, REGRA)["decisao"] == SEM_MODELO


@pytest.mark.parametrize(
    "candidato,vigente,esperado",
    [(0.80, 0.78, PROMOVER), (0.785, 0.78, MANTER), (0.79, 0.78, PROMOVER), (0.70, 0.78, MANTER)],
)
def test_promocao_exige_melhoria_minima(candidato, vigente, esperado):
    assert decidir_promocao(candidato, vigente, REGRA)["decisao"] == esperado


def test_falha_de_qualidade_impede_promocao():
    assert decidir_promocao(0.95, 0.70, REGRA, qualidade_ok=False)["decisao"] == MANTER


def test_empate_entre_candidatos_fica_com_o_modelo_a():
    assert eleger_vencedor({"modelo_a": 0.7, "modelo_b": 0.7}, "maximizar") == "modelo_a"
    assert eleger_vencedor({"modelo_a": 0.7, "modelo_b": 0.71}, "maximizar") == "modelo_b"


def test_psi_zero_para_mesma_distribuicao_e_alto_para_deslocada():
    rng = np.random.default_rng(0)
    base = pd.Series(rng.normal(0, 1, 5000))
    assert psi_numerico(base, base) == pytest.approx(0, abs=1e-9)
    assert psi_numerico(base, base + 2) > 0.2
    assert psi_categorico(pd.Series(list("aab") * 100), pd.Series(list("abb") * 100)) > 0.1


def test_particao_por_hash_e_estavel_entre_ciclos():
    pool_ciclo1 = pd.DataFrame({"registro_id": range(1, 1001)})
    pool_ciclo2 = pd.DataFrame({"registro_id": range(1, 1501)})
    _, val1 = dividir_por_hash(pool_ciclo1, 0.2)
    treino2, val2 = dividir_por_hash(pool_ciclo2, 0.2)
    assert set(val1.registro_id) <= set(val2.registro_id)
    assert not set(val1.registro_id) & set(treino2.registro_id)
    assert 0.15 < len(val2) / len(pool_ciclo2) < 0.25


def test_engenharia_trata_pdays_999_como_nao_contatado():
    df = pd.DataFrame({
        "pdays": [999, 3], "campaign": [1, 10], "age": [30, 70],
        **{c: ["unknown", "x"] for c in ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "day_of_week", "poutcome"]},
    })
    saida = engenharia_atributos(df)
    assert saida["contatado_antes"].tolist() == ["nao", "sim"]
    assert np.isnan(saida.loc[0, "pdays_dias"]) and saida.loc[1, "pdays_dias"] == 3
    assert saida["qtd_desconhecidos"].tolist() == [10, 0]
    assert "pdays" not in saida and "campaign" not in saida
