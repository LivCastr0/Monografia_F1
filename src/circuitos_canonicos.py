"""
Etapa 1 - Coleta: lista canônica de circuitos (recorte do projeto).
Funde as grafias divergentes do mesmo circuito, separa o traçado outer
de Sakhir (2020) e filtra circuitos com pelo menos 3 corridas.
Gera a base mestre que a extração de features vai percorrer.
"""
import pandas as pd

MIN_CORRIDAS = 3

# Mesma pista, grafias diferentes no FastF1 -> uma chave única
FUSOES = {
    "Monte Carlo": "Monaco",
    "Marina Bay": "Singapore",
    "Yas Marina": "Yas Island",
    "Miami Gardens": "Miami",
}

cal = pd.read_csv("data/raw/calendario_2018_2025.csv")

def circuito_canonico(row):
    # O GP de Sakhir 2020 correu no traçado outer: pista fisicamente distinta
    if row["evento"] == "Sakhir Grand Prix":
        return "Bahrain Outer"
    return FUSOES.get(row["local"], row["local"])

cal["circuito"] = cal.apply(circuito_canonico, axis=1)

# Conta LINHAS (não anos únicos): capta as dobradinhas, ex. Áustria em 2020/2021
n_por_circuito = cal.groupby("circuito").size()
incluidos = n_por_circuito[n_por_circuito >= MIN_CORRIDAS].index.tolist()
cal["incluir"] = cal["circuito"].isin(incluidos)

# Tabela mestre de sessões a processar (só circuitos incluídos)
sessoes = cal[cal["incluir"]].sort_values(["circuito", "ano", "round"])
sessoes.to_csv("data/raw/sessoes_incluidas.csv", index=False)

# Resumo por circuito incluído
resumo_inc = (
    sessoes.groupby("circuito")
    .agg(n_corridas=("ano", "size"),
         anos=("ano", lambda s: sorted(s.unique().tolist())))
    .sort_values("n_corridas", ascending=False)
)
resumo_inc.to_csv("data/raw/circuitos_canonicos.csv")

excluidos = n_por_circuito[n_por_circuito < MIN_CORRIDAS].sort_values(ascending=False)

print(f"Circuitos incluídos: {len(incluidos)}")
print(f"Sessões a processar: {len(sessoes)}\n")
print(resumo_inc.to_string())
print(f"\nExcluídos (< {MIN_CORRIDAS} corridas):")
print(excluidos.to_string())