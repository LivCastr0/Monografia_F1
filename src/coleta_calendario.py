"""
Etapa 1 - Coleta: mapeamento do calendário e dos circuitos (2018-2025).
Gera a base que as etapas seguintes vão percorrer.
"""
import fastf1
import pandas as pd

fastf1.Cache.enable_cache("data/cache")

ANOS = range(2018, 2026)  # 2018 a 2025 inclusive

linhas = []
for ano in ANOS:
    schedule = fastf1.get_event_schedule(ano, include_testing=False)
    for _, ev in schedule.iterrows():
        linhas.append({
            "ano": ano,
            "round": ev["RoundNumber"],
            "pais": ev["Country"],
            "local": ev["Location"],
            "evento": ev["EventName"],
            "formato": ev["EventFormat"],
            "data": ev["EventDate"],
        })

calendario = pd.DataFrame(linhas)
calendario.to_csv("data/raw/calendario_2018_2025.csv", index=False)

# Resumo por circuito, usando "local" como chave canônica candidata
resumo = (
    calendario
    .groupby("local")
    .agg(
        n_corridas=("ano", "count"),
        anos=("ano", lambda s: sorted(s.unique().tolist())),
        nomes_evento=("evento", lambda s: sorted(s.unique().tolist())),
    )
    .sort_values("n_corridas", ascending=False)
)
resumo.to_csv("data/raw/resumo_circuitos.csv")

print(f"Total de GPs 2018-2025: {len(calendario)}")
print(f"Locais distintos: {resumo.shape[0]}\n")
print(resumo.to_string())