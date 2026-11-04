"""
Etapa 2 - Coleta das corridas (2018-2025, 164 corridas, 26 circuitos).

Baixa e salva cru. Não filtra, não agrega, não junta com nada. A limpeza e a
montagem da base de piloto-corrida são o passo seguinte, em outro script.

Motivo: recoletar custa ~3 horas presa no limite de 500 requisições por hora.
Refazer uma conta em cima de arquivo já salvo custa segundos.

Por corrida, salva quatro arquivos em data/interim/corridas/:

  {ano}_{round}_pilotos.csv    uma linha por piloto   (largada, chegada, status)
  {ano}_{round}_stints.csv     uma linha por pneu     (composto, volta início/fim)
  {ano}_{round}_bandeiras.csv  uma linha por bandeira (safety car, VSC, vermelha)
  {ano}_{round}_clima.csv      uma linha por medição  (chuva, temperatura)

Retomável: se os quatro arquivos de uma corrida já existem, ela é pulada.
Pode interromper com Ctrl+C e rodar de novo depois - continua de onde parou.

Se bater o limite da F1, o script para sozinho e avisa. É só rodar de novo
na hora seguinte.

telemetry=False: os dados de sensor do carro não são usados na Etapa 2.
2022 incluído: o teste confirmou que as corridas de 2022 carregam normalmente
(o problema da Etapa 1 era só da classificação).
"""
import time
from pathlib import Path

import fastf1
import pandas as pd
from fastf1.exceptions import RateLimitExceededError

fastf1.Cache.enable_cache("data/cache")

ENTRADA = Path("data/raw/sessoes_incluidas.csv")
DESTINO = Path("data/interim/corridas")
DESTINO.mkdir(parents=True, exist_ok=True)

# Status que significam "chegou ao fim da corrida".
# Cuidado: "Finished" sozinho exclui quem tomou volta, que também terminou.
STATUS_TERMINOU = {"Finished", "Lapped"}


class LimiteAtingido(Exception):
    """Bateu o teto de requisições da F1. Para tudo e volta depois."""


def caminhos(ano, rnd):
    base = f"{ano}_{rnd:02d}"
    return {
        "stints": DESTINO / f"{base}_stints.csv",
        "bandeiras": DESTINO / f"{base}_bandeiras.csv",
        "clima": DESTINO / f"{base}_clima.csv",
        # pilotos é salvo por último e serve de marca de "corrida pronta"
        "pilotos": DESTINO / f"{base}_pilotos.csv",
    }


def ja_baixada(ano, rnd):
    return all(p.exists() for p in caminhos(ano, rnd).values())


def carregar(ano, rnd, tentativas=2):
    """Carrega a corrida sem telemetria de carro. Aborta tudo no limite."""
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


def tabela_pilotos(ses, ano, rnd):
    """Uma linha por piloto: largada, chegada, status."""
    voltas_por_piloto = ses.laps.groupby("Driver")["LapNumber"].max()

    res = ses.results.copy()
    df = pd.DataFrame({
        "ano": ano,
        "round": rnd,
        "piloto": res["Abbreviation"],
        "numero_piloto": res["DriverNumber"],
        "equipe": res["TeamName"],
        "grid": res["GridPosition"],
        "posicao_final": res["Position"],
        "posicao_classificada": res["ClassifiedPosition"],
        "status": res["Status"],
        "pontos": res["Points"],
    })

    # GridPosition = 0 é largada do box, não posição zero.
    # Guardo o valor cru E a marca; quem for calcular posições ganhas
    # decide o que fazer com essas linhas.
    df["largou_do_pit"] = df["grid"] == 0
    df["terminou"] = df["status"].isin(STATUS_TERMINOU)
    df["voltas_completadas"] = df["piloto"].map(voltas_por_piloto)
    df["total_voltas"] = ses.total_laps
    return df.reset_index(drop=True)


def tabela_stints(ses, ano, rnd):
    """Uma linha por pneu usado, com as voltas de início e fim."""
    stints = (
        ses.laps
        .groupby(["Driver", "Stint"], dropna=True)
        .agg(
            composto=("Compound", "first"),
            volta_inicio=("LapNumber", "min"),
            volta_fim=("LapNumber", "max"),
            pneu_novo=("FreshTyre", "first"),
            vida_pneu_inicial=("TyreLife", "min"),
        )
        .reset_index()
        .rename(columns={"Driver": "piloto", "Stint": "stint"})
    )
    stints.insert(0, "round", rnd)
    stints.insert(0, "ano", ano)
    return stints


def tabela_bandeiras(ses, ano, rnd):
    """Uma linha por mudança de bandeira, com o instante em que ocorreu.

    Códigos: 1 livre, 2 amarela, 4 safety car, 5 vermelha,
             6 VSC ativado, 7 VSC encerrando.
    Quantas voltas cada período durou é conta do passo seguinte.
    """
    ts = ses.track_status.copy()
    if ts.empty:
        return pd.DataFrame(columns=["ano", "round", "tempo_s", "codigo", "mensagem"])
    return pd.DataFrame({
        "ano": ano,
        "round": rnd,
        "tempo_s": ts["Time"].dt.total_seconds(),
        "codigo": ts["Status"],
        "mensagem": ts.get("Message", pd.Series(index=ts.index, dtype=object)),
    }).reset_index(drop=True)


def tabela_clima(ses, ano, rnd):
    """Uma linha por medição. Guardo cru porque chuva que começa no meio da
    corrida é evento estratégico, e uma média esconderia isso."""
    cl = ses.weather_data.copy()
    if cl.empty:
        return pd.DataFrame(columns=["ano", "round", "tempo_s", "chuva",
                                     "temp_pista", "temp_ar", "umidade"])
    return pd.DataFrame({
        "ano": ano,
        "round": rnd,
        "tempo_s": cl["Time"].dt.total_seconds(),
        "chuva": cl["Rainfall"],
        "temp_pista": cl["TrackTemp"],
        "temp_ar": cl["AirTemp"],
        "umidade": cl.get("Humidity"),
    }).reset_index(drop=True)


# ---------------------------------------------------------------------------

sessoes = pd.read_csv(ENTRADA).sort_values(["ano", "round"])
pendentes = [s for _, s in sessoes.iterrows()
             if not ja_baixada(int(s["ano"]), int(s["round"]))]

print(f"Corridas na lista: {len(sessoes)}")
print(f"Já baixadas: {len(sessoes) - len(pendentes)}")
print(f"A baixar agora: {len(pendentes)}\n")

baixadas, falhas = 0, []

for i, s in enumerate(pendentes, 1):
    ano, rnd, circ = int(s["ano"]), int(s["round"]), s["circuito"]
    print(f"[{i}/{len(pendentes)}] {ano} r{rnd} {circ}")

    try:
        ses = carregar(ano, rnd)
    except LimiteAtingido:
        print("\n" + "=" * 60)
        print("Limite de requisições da F1 atingido. Parando aqui.")
        print("Rode este mesmo script daqui a uma hora - ele continua sozinho.")
        print("=" * 60)
        break

    if ses is None:
        print("    falhou nas duas tentativas, seguindo")
        falhas.append((ano, rnd, circ))
        continue

    p = caminhos(ano, rnd)
    tabela_stints(ses, ano, rnd).to_csv(p["stints"], index=False)
    tabela_bandeiras(ses, ano, rnd).to_csv(p["bandeiras"], index=False)
    tabela_clima(ses, ano, rnd).to_csv(p["clima"], index=False)
    # pilotos por último: se o script morrer no meio, a corrida não conta
    # como pronta e será refeita na próxima execução
    tabela_pilotos(ses, ano, rnd).to_csv(p["pilotos"], index=False)

    baixadas += 1
    print(f"    salvo: {len(ses.results)} pilotos, {len(ses.laps)} voltas")

# ---------------------------------------------------------------------------

restantes = [s for _, s in sessoes.iterrows()
             if not ja_baixada(int(s["ano"]), int(s["round"]))]

print(f"\nBaixadas nesta execução: {baixadas}")
print(f"Total pronto: {len(sessoes) - len(restantes)}/{len(sessoes)}")

if falhas:
    print(f"\nFalharam ({len(falhas)}):")
    for ano, rnd, circ in falhas:
        print(f"  {ano} r{rnd} {circ}")
    pd.DataFrame(falhas, columns=["ano", "round", "circuito"]).to_csv(
        "data/interim/falhas_coleta_corridas.csv", index=False
    )

if restantes:
    print(f"\nFaltam {len(restantes)}. Rode de novo.")
else:
    print("\nColeta completa. Próximo passo: montar a base de piloto-corrida.")