"""
Etapa 1 - Derivação: monta o vetor de features final para a clusterização.

Junta as fontes numa tabela única circuito x feature (26 x 8), input direto
do clustering:
  - Telemetria (5): vel_media, vel_minima, zonas_freio, pico_desac e
        densidade_curvas (= n_curvas / comprimento_km). A densidade SUBSTITUI
        n_curvas cru (normaliza a contagem de curvas pelo tamanho da pista).
  - Pirelli (1):    abrasion, da agregação por mediana (agrega_pirelli.py).
  - Manuais (2):    pit_lane e altitude, coleta manual não-Pirelli.

As features de validação (braking, lateral) e opcionais (grip, track_evolution)
ficam fora deste vetor de propósito - vivem no agregado Pirelli para uso a jusante.

Robusto a nomes/codificação: localiza colunas por substring; lê utf-8 com
fallback latin-1.
"""
import pandas as pd

TELEMETRIA = "data/processed/features_fastf1_por_circuito.csv"
PIRELLI = "data/processed/pirelli_agregado_por_circuito.csv"
MANUAIS = "data/external/features_manuais.csv"
COMPRIMENTOS = "data/external/circuit_length.csv"
SAIDA = "data/processed/features_completas_por_circuito.csv"

# n_curvas é lido para derivar a densidade, depois descartado do vetor final
COLS_TELEMETRIA = ["vel_media", "vel_minima", "zonas_freio", "pico_desac", "n_curvas"]


def le_csv(caminho, **kw):
    """Lê CSV tolerando separador desconhecido e utf-8/latin-1."""
    try:
        return pd.read_csv(caminho, sep=None, engine="python", encoding="utf-8", **kw)
    except (UnicodeDecodeError, UnicodeError):
        return pd.read_csv(caminho, sep=None, engine="python", encoding="latin-1", **kw)


def acha_coluna(df, *chaves):
    """1a coluna que contém todas as 'chaves' (case-insensitive)."""
    for col in df.columns:
        if all(k.lower() in col.lower() for k in chaves):
            return col
    raise KeyError(f"Nenhuma coluna com {chaves}. Disponíveis: {list(df.columns)}")


# --- Telemetria + comprimento -> densidade_curvas ---
tele = pd.read_csv(TELEMETRIA)[["circuito"] + COLS_TELEMETRIA]

comp = le_csv(COMPRIMENTOS)
# lê por posição (cabeçalho pode vir com typo, ex.: 'cirucito')
comp = comp.rename(columns={comp.columns[0]: "circuito", comp.columns[1]: "comprimento_m"})
comp["comprimento_m"] = pd.to_numeric(
    comp["comprimento_m"].astype(str).str.replace(",", ".", regex=False), errors="coerce"
)

tele = tele.merge(comp[["circuito", "comprimento_m"]], on="circuito")
tele["densidade_curvas"] = tele["n_curvas"] / (tele["comprimento_m"] / 1000)
tele = tele.drop(columns=["n_curvas", "comprimento_m"])  # densidade substitui n_curvas

# --- Abrasão (fonte única = agregado Pirelli) ---
pir = pd.read_csv(PIRELLI)
col_abr = acha_coluna(pir, "abras")
abrasao = pir[["circuito", col_abr]].rename(columns={col_abr: "abrasion"})

# --- Manuais não-Pirelli ---
man = le_csv(MANUAIS, decimal=",")
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