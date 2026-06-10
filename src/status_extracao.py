import fastf1
import pandas as pd

fastf1.Cache.enable_cache("data/cache")
fastf1.Cache.offline_mode(True)

ANOS_EXCLUIR = {2022}

sessoes = pd.read_csv("data/raw/sessoes_incluidas.csv")
sessoes_auditadas = sessoes[~sessoes["ano"].isin(ANOS_EXCLUIR)].copy()

prontas, faltam = [], []

for _, s in sessoes_auditadas.iterrows():
    ano, rnd, circ = int(s["ano"]), int(s["round"]), s["circuito"]
    try:
        q = fastf1.get_session(ano, rnd, "Q")
        q.load(laps=True, telemetry=True, weather=True, messages=False)
        ok = not q.laps.empty
    except Exception:
        ok = False

    (prontas if ok else faltam).append((ano, rnd, circ))

df_p = pd.DataFrame(prontas, columns=["ano", "round", "circuito"])
df_f = pd.DataFrame(faltam, columns=["ano", "round", "circuito"])

print(f"Sessões auditadas: {len(sessoes_auditadas)}")
print(f"Anos excluídos da auditoria: {sorted(ANOS_EXCLUIR)}")
print(f"Sessões prontas no cache: {len(df_p)}/{len(sessoes_auditadas)}")
print(f"Faltam: {len(df_f)}\n")

prontas_set = set(prontas)

por_circ = (
    sessoes_auditadas.assign(
        pronta=sessoes_auditadas.apply(
            lambda r: (
                int(r["ano"]),
                int(r["round"]),
                r["circuito"]
            ) in prontas_set,
            axis=1
        )
    )
    .groupby("circuito")
    .agg(
        prontas=("pronta", "sum"),
        total=("pronta", "size")
    )
    .sort_values(["prontas", "total"])
)

print(por_circ.to_string())
print("\nCorridas que ainda faltam:")
print(df_f.sort_values(["circuito", "ano"]).to_string(index=False))