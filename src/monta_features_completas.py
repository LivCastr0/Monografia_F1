"""
Etapa 1 - Derivação: monta o vetor de features final para a clusterização.

Junta as três fontes numa tabela única circuito x feature (26 x 8), input
direto do clustering:
  - Telemetria (5): subconjunto PODADO de features_fastf1_por_circuito.csv.
  - Pirelli (1):    abrasion, da agregação por mediana (agrega_pirelli.py).
  - Manuais (2):    pit_lane e altitude, coleta manual não-Pirelli.

As features de validação (braking, lateral) e opcionais (grip, track_evolution)
ficam fora deste vetor de propósito - vivem no agregado Pirelli para uso a jusante.

Robusto a nomes de coluna: localiza abrasão/pit_lane/altitude por substring,
para tolerar pequenas variações de nomenclatura entre planilhas.
"""
import pandas as pd

TELEMETRIA = "data/processed/features_fastf1_por_circuito.csv"
PIRELLI = "data/processed/pirelli_agregado_por_circuito.csv"
MANUAIS = "data/external/features_manuais.csv"
SAIDA = "data/processed/features_completas_por_circuito.csv"

COLS_TELEMETRIA = ["vel_media", "vel_minima", "zonas_freio", "pico_desac", "n_curvas"]


def acha_coluna(df, *chaves):
    """Localiza a 1a coluna que contém todas as 'chaves' (case-insensitive)."""
    for col in df.columns:
        baixo = col.lower()
        if all(k.lower() in baixo for k in chaves):
            return col
    raise KeyError(
        f"Nenhuma coluna com {chaves}. Colunas disponíveis: {list(df.columns)}"
    )


# Telemetria podada
tele = pd.read_csv(TELEMETRIA)[["circuito"] + COLS_TELEMETRIA]

# Abrasão: fonte única = agregado Pirelli (localizada por substring)
pir = pd.read_csv(PIRELLI)
col_abr = acha_coluna(pir, "abras")
abrasao = pir[["circuito", col_abr]].rename(columns={col_abr: "abrasion"})

# Manuais não-Pirelli (detecta separador e decimal automaticamente)
try:
    man = pd.read_csv(MANUAIS, sep=None, engine="python", decimal=",", encoding="utf-8")
except (UnicodeDecodeError, UnicodeError):
    man = pd.read_csv(MANUAIS, sep=None, engine="python", decimal=",", encoding="latin-1")
col_pit = acha_coluna(man, "pit")
col_alt = acha_coluna(man, "alt")
manuais = man[["circuito", col_pit, col_alt]].rename(
    columns={col_pit: "pit_lane_time_loss_s", col_alt: "altitude_m"}
)

final = tele.merge(abrasao, on="circuito").merge(manuais, on="circuito")

assert final.isnull().sum().sum() == 0, "Há nulos no vetor final!"
assert len(final) == 26, f"Esperado 26 circuitos, obtido {len(final)}"

final.to_csv(SAIDA, index=False)
print(f"OK: {SAIDA} {final.shape} - zero nulos")
print("Features:", [c for c in final.columns if c != "circuito"])