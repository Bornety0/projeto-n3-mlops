"""Eleição do vencedor do ciclo, regra de promoção e manifesto do modelo vigente."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib

from src.config import caminho

PROMOVER = "promover"
MANTER = "manter_vigente"
SEM_MODELO = "sem_promocao"


def _maior_e_melhor(direcao: str) -> bool:
    return direcao == "maximizar"


def eleger_vencedor(metricas_por_candidato: dict[str, float], direcao: str) -> str:
    """Melhor candidato pela métrica principal; empate fica com o primeiro (Modelo A, o mais simples)."""
    ordem = list(metricas_por_candidato)
    sinal = 1 if _maior_e_melhor(direcao) else -1
    return max(ordem, key=lambda nome: (sinal * metricas_por_candidato[nome], -ordem.index(nome)))


def decidir_promocao(
    metrica_candidato: float,
    metrica_vigente: float | None,
    regra: dict,
    direcao: str = "maximizar",
    qualidade_ok: bool = True,
) -> dict:
    """Aplica a regra pré-definida em config.yaml (seção 'promocao')."""
    sinal = 1 if _maior_e_melhor(direcao) else -1
    piso = regra["metrica_minima"]
    melhoria = regra["melhoria_minima"]
    criterios = {
        "qualidade_ok": qualidade_ok,
        "acima_do_piso": sinal * (metrica_candidato - piso) >= 0,
        "metrica_candidato": round(metrica_candidato, 4),
        "metrica_vigente": None if metrica_vigente is None else round(metrica_vigente, 4),
        "piso": piso,
        "melhoria_minima": melhoria,
    }

    if metrica_vigente is None:
        if qualidade_ok and criterios["acima_do_piso"]:
            return {"decisao": PROMOVER, "motivo": "Não há modelo vigente e o candidato supera o piso.", "criterios": criterios}
        return {"decisao": SEM_MODELO, "motivo": "Não há modelo vigente e o candidato não atende ao piso/qualidade.", "criterios": criterios}

    ganho = sinal * (metrica_candidato - metrica_vigente)
    criterios["ganho"] = round(ganho, 4)
    criterios["supera_limiar"] = ganho >= melhoria
    if not qualidade_ok:
        return {"decisao": MANTER, "motivo": "Verificações de qualidade falharam.", "criterios": criterios}
    if not criterios["acima_do_piso"]:
        return {"decisao": MANTER, "motivo": "Candidato abaixo do piso absoluto.", "criterios": criterios}
    if criterios["supera_limiar"]:
        return {"decisao": PROMOVER, "motivo": f"Ganho de {ganho:+.4f} >= limiar {melhoria}.", "criterios": criterios}
    return {"decisao": MANTER, "motivo": f"Ganho de {ganho:+.4f} < limiar {melhoria}.", "criterios": criterios}


def carregar_manifesto(cfg: dict) -> dict | None:
    arquivo = caminho(cfg, cfg["promocao"]["manifesto"])
    if not arquivo.exists():
        return None
    return json.loads(arquivo.read_text(encoding="utf-8"))


def carregar_modelo_vigente(cfg: dict):
    """Retorna (None, manifesto) quando já houve ciclos, mas nenhum candidato foi promovido."""
    manifesto = carregar_manifesto(cfg)
    if not manifesto or not manifesto.get("caminho_modelo"):
        return None, manifesto
    return joblib.load(caminho(cfg, manifesto["caminho_modelo"])), manifesto


def salvar_modelo_promovido(pipeline, cfg: dict, cycle_id: str, nome: str) -> str:
    """Cada promoção gera um arquivo novo: modelos anteriores nunca são sobrescritos."""
    relativo = Path(cfg["promocao"]["diretorio_historico"]) / f"{cycle_id}__{nome}.joblib"
    destino = caminho(cfg, relativo)
    destino.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, destino, compress=3)
    return relativo.as_posix()


def atualizar_manifesto(cfg: dict, manifesto_atual: dict | None, ciclo: dict, decisao: dict, novo_vigente: dict | None) -> dict:
    """Mantém no manifesto o vigente e o resumo da última decisão (promoção ou manutenção)."""
    agora = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifesto = dict(manifesto_atual or {})
    if novo_vigente:
        anteriores = manifesto.get("promocoes_anteriores", [])
        if manifesto_atual:
            anteriores.append({k: manifesto_atual.get(k) for k in ("cycle_id", "algoritmo", "run_id", "caminho_modelo", "data_version", "promovido_em")})
        manifesto = {**novo_vigente, "promovido_em": agora, "promocoes_anteriores": anteriores}
    manifesto["ultimo_ciclo"] = {
        "cycle_id": ciclo["cycle_id"],
        "data_version": ciclo["data_version"],
        "decisao": decisao["decisao"],
        "motivo": decisao["motivo"],
        "criterios": decisao["criterios"],
        "avaliado_em": agora,
    }
    arquivo = caminho(cfg, cfg["promocao"]["manifesto"])
    arquivo.parent.mkdir(parents=True, exist_ok=True)
    arquivo.write_text(json.dumps(manifesto, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return manifesto


def registrar_historico(cfg: dict, linha: dict) -> None:
    arquivo = caminho(cfg, cfg["saida"]["historico_ciclos"])
    arquivo.parent.mkdir(parents=True, exist_ok=True)
    novo = not arquivo.exists()
    with open(arquivo, "a", newline="", encoding="utf-8") as saida:
        escritor = csv.DictWriter(saida, fieldnames=list(linha))
        if novo:
            escritor.writeheader()
        escritor.writerow(linha)
