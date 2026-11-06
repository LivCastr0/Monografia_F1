"""
Etapa 2 - Coleta volta a volta, para o piloto do caminho A1.

A coleta original (coleta_corridas.py) resumiu as voltas em stints e não guardou
a posição de cada volta. Este script guarda a tabela de voltas inteira, para não
ser preciso recoletar se a medida do A1 mudar: posição, tempo de volta, entrada
e saída dos boxes, pneu e bandeira.

  python src/coleta_voltas.py           só as 10 corridas do piloto
  python src/coleta_voltas.py --todas   todas as corridas da base

Salva em data/interim/voltas/{ano}_{round}_voltas.csv, uma linha por piloto por
volta. Retomável como a coleta original: corrida já salva é pulada, e pode
interromper e rodar de novo.

O carregamento é idêntico ao de coleta_corridas.py e usa a mesma pasta de cache.
Se data/cache ainda existir e a versão da FastF1 for a mesma, as corridas devem
vir do cache, sem gastar o limite de 500 requisições por hora. O tempo de cada
corrida aparece na saída, e ao fim o script projeta o tempo para a base inteira.

As 10 corridas do piloto foram sorteadas uma única vez, antes de qualquer dado
de volta existir, e não devem ser trocadas depois de ver resultado. Sorteio:
numpy default_rng(2026), dentro de cada perfil e metade em cada era (2018-21 e
2022+), com 3 corridas de alto downforce, 3 de stop-and-go, 2 de potência e 2 de
urbano-técnico.
"""
import argparse
import time
from pathlib import Path

import fastf1
import pandas as pd
from fastf1.exceptions import RateLimitExceededError

ENTRADA = Path("data/raw/sessoes_incluidas.csv")
DESTINO = Path("data/interim/voltas")

CORRIDAS_PILOTO = [
    # alto downforce
    (2021, 13),  # Zandvoort
    (2022, 15),  # Zandvoort
    (2022, 2),   # Jeddah
    # stop-and-go
    (2019, 2),   # Sakhir
    (2022, 8),   # Baku
    (2023, 22),  # Yas Island
    # potência
    (2020, 7),   # Spa-Francorchamps
    (2023, 12),  # Spa-Francorchamps
    # urbano-técnico
    (2021, 5),   # Monaco
    (2025, 8),   # Monaco
]

# coluna da FastF1 -> coluna salva
COLUNAS = {
    "Driver": "piloto",
    "Team": "equipe",
    "LapNumber": "volta",
    "Position": "posicao",
    "LapTime": "tempo_volta_s",
    "Time": "fim_volta_s",
    "PitInTime": "entrada_box_s",
    "PitOutTime": "saida_box_s",
    "Stint": "stint",
    "Compound": "composto",
    "TyreLife": "vida_pneu",
    "FreshTyre": "pneu_novo",
    "TrackStatus": "bandeira",
    "IsAccurate": "volta_precisa",
    "FastF1Generated": "gerada_pela_fastf1",
}
COLUNAS_TEMPO = ["LapTime", "Time", "PitInTime", "PitOutTime"]


class LimiteAtingido(Exception):
    """Bateu o teto de requisições da F1. Para tudo e volta depois."""


def caminho(ano, rnd):
    return DESTINO / f"{ano}_{rnd:02d}_voltas.csv"


def carregar(ano, rnd, tentativas=2):
    """Os mesmos argumentos de coleta_corridas.py, para aproveitar o cache."""
    for t in range(tentativas):
        try:
            ses = fastf1.get_session(int(ano), int(rnd), "R")
            ses.load(laps=True, telemetry=False, weather=True, messages=False)
            if ses.laps.empty:
                raise ValueError("tabela de voltas vazia")
            return ses
        except RateLimitExceededError:
            raise LimiteAtingido()
        except Exception as e:
            print(f"    [{t+1}/{tentativas}] {type(e).__name__}: {e}")
            time.sleep(6 * (t + 1))
    return None


def tabela_voltas(ses, ano, rnd):
    """Uma linha por piloto por volta, com tempos em segundos de sessão."""
    voltas = ses.laps
    faltando = [c for c in COLUNAS if c not in voltas.columns]
    if faltando:
        raise KeyError(f"a FastF1 não trouxe as colunas {faltando}")

    df = pd.DataFrame({novo: voltas[orig].values for orig, novo in COLUNAS.items()})
    for orig in COLUNAS_TEMPO:
        df[COLUNAS[orig]] = pd.to_timedelta(voltas[orig].values).total_seconds()
    df.insert(0, "round", rnd)
    df.insert(0, "ano", ano)
    return df.sort_values(["volta", "posicao"]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--todas", action="store_true",
                        help="coleta todas as corridas da base, não só o piloto")
    args = parser.parse_args()

    fastf1.Cache.enable_cache("data/cache")
    DESTINO.mkdir(parents=True, exist_ok=True)

    sessoes = pd.read_csv(ENTRADA).sort_values(["ano", "round"])
    if args.todas:
        lista = [(int(a), int(r)) for a, r in zip(sessoes["ano"], sessoes["round"])]
    else:
        lista = list(CORRIDAS_PILOTO)

    nomes = {(int(a), int(r)): c
             for a, r, c in zip(sessoes["ano"], sessoes["round"], sessoes["circuito"])}
    pendentes = [c for c in lista if not caminho(*c).exists()]
    print(f"Corridas na lista: {len(lista)}   já salvas: {len(lista) - len(pendentes)}"
          f"   a baixar: {len(pendentes)}\n")

    tempos, falhas = [], []
    for i, (ano, rnd) in enumerate(pendentes, 1):
        print(f"[{i}/{len(pendentes)}] {ano} r{rnd} {nomes.get((ano, rnd), '?')}")
        inicio = time.monotonic()
        try:
            ses = carregar(ano, rnd)
        except LimiteAtingido:
            print("\nLimite de requisições da F1 atingido. Parando aqui.")
            print("Rode este mesmo script daqui a uma hora: ele continua sozinho.")
            break

        if ses is None:
            print("    falhou nas duas tentativas, seguindo")
            falhas.append((ano, rnd))
            continue

        tabela = tabela_voltas(ses, ano, rnd)
        tabela.to_csv(caminho(ano, rnd), index=False)
        segundos = time.monotonic() - inicio
        tempos.append(segundos)
        print(f"    salvo: {tabela['piloto'].nunique()} pilotos, "
              f"{len(tabela)} voltas, {segundos:.0f} s")

    prontas = sum(caminho(*c).exists() for c in lista)
    print(f"\nTotal pronto: {prontas}/{len(lista)}")
    if falhas:
        print("Falharam: " + ", ".join(f"{a} r{r}" for a, r in falhas))
    if tempos and not args.todas:
        minutos = sum(tempos) / len(tempos) * len(sessoes) / 60
        print(f"Tempo médio por corrida: {sum(tempos) / len(tempos):.0f} s. "
              f"Projeção para {len(sessoes)} corridas: {minutos:.0f} min, sem contar "
              f"as pausas do limite de requisições.")
    if prontas == len(lista):
        print("Próximo passo: python src/avalia_piloto_a1.py")


if __name__ == "__main__":
    main()
