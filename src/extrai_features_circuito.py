"""
Etapa 1 - Extração das features dinâmicas por circuito (grão circuito).
Fingerprint (#2 #3 #4) E temperatura (#8) saem da MESMA sessão de CLASSIFICAÇÃO.
2022 é EXCLUÍDO: a API da F1 não serve o session_info dessa temporada,
então não há o que baixar — tentá-la só queima o limite de 500 req/h à toa.
Com 2022 fora, todas as sessões restantes vêm do cache e o run completa sem
tocar no limite. Decisão documentada na metodologia.

Features de extremos (vel_minima e os picos de aceleração) usam percentil 1/99
em vez do mín/máx absoluto, para robustez ao ruído de telemetria.
"""
import time
import fastf1
import numpy as np
import pandas as pd
from fastf1.exceptions import RateLimitExceededError

fastf1.Cache.enable_cache("data/cache")

CIRCUITOS_TESTE = None        # None = todos os 26
ANOS_EXCLUIR = {2022}         # indisponível na API (session_info) — decisão de projeto

sessoes = pd.read_csv("data/raw/sessoes_incluidas.csv")
if CIRCUITOS_TESTE:
    sessoes = sessoes[sessoes["circuito"].isin(CIRCUITOS_TESTE)]
sessoes = sessoes[~sessoes["ano"].isin(ANOS_EXCLUIR)]
print(f"Sessões a processar: {len(sessoes)} (anos excluídos: {sorted(ANOS_EXCLUIR)})")


class RateLimitHit(Exception):
    pass


def carregar_q(ano, rnd, tentativas=2):
    """Classificação com telemetria + clima. Aborta tudo se bater o rate limit."""
    for t in range(tentativas):
        try:
            ses = fastf1.get_session(int(ano), int(rnd), "Q")
            ses.load(laps=True, telemetry=True, weather=True, messages=False)
            if ses.laps.empty:
                raise ValueError("tabela de voltas vazia")
            return ses
        except RateLimitExceededError:
            raise RateLimitHit()
        except Exception as e:
            print(f"    [{t+1}/{tentativas}] {ano} r{rnd} Q: {type(e).__name__}: {e}")
            time.sleep(6 * (t + 1))
    return None


def features_de_volta(car):
    speed = car["Speed"].to_numpy(dtype=float)
    throttle = car["Throttle"].to_numpy(dtype=float)
    brake = np.asarray(car["Brake"].to_numpy(), dtype=bool)
    t = car["Time"].dt.total_seconds().to_numpy()
    throttle = np.where(throttle > 100, np.nan, throttle)
    dt = np.diff(t)
    dv = np.diff(speed) * (1000 / 3600)
    with np.errstate(divide="ignore", invalid="ignore"):
        acc = dv / dt
    acc = acc[np.isfinite(acc)]
    if len(acc) >= 5:
        acc = pd.Series(acc).rolling(5, center=True, min_periods=1).median().to_numpy()
    zonas_freio = int(np.sum((~brake[:-1]) & (brake[1:])))
    return {
        "vel_media": np.nanmean(speed),
        # percentil 1 em vez de nanmin: ignora pontos de telemetria espúrios
        "vel_minima": np.nanpercentile(speed, 1),
        "pct_baixa_vel": np.nanmean(speed < 120) * 100,
        "pct_full_throttle": np.nanmean(throttle >= 99) * 100,
        "pct_freando": np.mean(brake) * 100,
        "zonas_freio": zonas_freio,
        # percentil 1/99 em vez de min/max: robustez ao spike da derivada dv/dt
        "pico_desac": np.nanpercentile(acc, 1) if len(acc) else np.nan,
        "pico_acel": np.nanpercentile(acc, 99) if len(acc) else np.nan,
    }


voltas, temps, curvas, falhas = [], [], {}, []
rate_limited = False

try:
    for _, s in sessoes.iterrows():
        ano, rnd, circ = int(s["ano"]), int(s["round"]), s["circuito"]
        print(f"{circ} {ano} (r{rnd})")
        q = carregar_q(ano, rnd)
        if q is None:
            falhas.append((ano, rnd, circ))
            continue
        if circ not in curvas:
            try:
                curvas[circ] = len(q.get_circuit_info().corners)
            except Exception:
                pass
        try:
            temps.append({"circuito": circ, "ano": ano,
                          "temp_pista": q.weather_data["TrackTemp"].median()})
        except Exception:
            pass
        for drv in q.laps["Driver"].unique():
            try:
                volta = q.laps[q.laps["Driver"] == drv].pick_fastest()
                car = volta.get_car_data()
                if car.empty:
                    continue
                f = features_de_volta(car)
            except Exception:
                continue
            f.update({"circuito": circ, "ano": ano, "piloto": drv})
            voltas.append(f)
except RateLimitHit:
    rate_limited = True

df_voltas = pd.DataFrame(voltas)
df_temps = pd.DataFrame(temps)
df_voltas.to_csv("data/raw/voltas_fingerprint.csv", index=False)
df_temps.to_csv("data/raw/temp_por_corrida.csv", index=False)
pd.DataFrame(falhas, columns=["ano", "round", "circuito"]).to_csv(
    "data/raw/falhas_extracao.csv", index=False)

cols_fp = ["vel_media", "vel_minima", "pct_baixa_vel", "pct_full_throttle",
           "pct_freando", "zonas_freio", "pico_desac", "pico_acel"]
if not df_voltas.empty:
    circuitos = (
        df_voltas.groupby("circuito")[cols_fp].median()
        .join(df_temps.groupby("circuito")["temp_pista"].median())
        .join(pd.Series(curvas, name="n_curvas"))
    )
    circuitos.to_csv("data/processed/features_fastf1_por_circuito.csv")
    print(f"\nCircuitos com features: {circuitos.shape[0]}/26")

if rate_limited:
    print("\n>>> Limite de 500 req/h atingido. O progresso ja esta no cache.")
    print(">>> Espere ~1 hora e rode o MESMO script de novo para continuar.")
else:
    print(f"\nConcluido. Falhas: {len(falhas)} -> {falhas if falhas else 'nenhuma'}")