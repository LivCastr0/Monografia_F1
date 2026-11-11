"""
Etapa 2 - Números das fichas dos clusters (descrição do comportamento de corrida
por tipo de pista). Só corridas secas (as 122 da base de duelos).

Sai: data/processed/fichas/visao_geral.csv, por_circuito.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd

P = Path("data/processed")
SAI = P / "fichas"


def main():
    SAI.mkdir(parents=True, exist_ok=True)
    du = pd.read_csv(P / "base_duelos.csv")
    secas = du[["ano", "round"]].drop_duplicates()
    mapa = du.drop_duplicates("circuito").set_index("circuito")["d_perfil"]
    co = pd.read_csv(P / "base_corrida.csv").merge(secas)
    pa = pd.read_csv(P / "base_decisao_parada.csv", usecols=["ano", "round", "circuito", "parou", "d_perfil"])
    st = pd.read_csv(P / "base_stint.csv").merge(secas)
    pc = pd.read_csv(P / "base_piloto_corrida.csv").merge(secas)
    pc = pc[pc["terminou"] == True]
    for d in (co, st, pc):
        d["d_perfil"] = d["circuito"].map(mapa)

    def medidas(g_du, g_co, g_pa, g_st, g_pc):
        n_corr = g_du.groupby(["ano", "round"]).ngroups
        return {
            "corridas_secas": n_corr,
            "circuitos": g_du["circuito"].nunique(),
            "voltas_duelo_por_corrida": len(g_du) / n_corr,
            "ultrapassagens_por_corrida": g_du["passou"].sum() / n_corr,
            "taxa_ultrapassagem_pct": 100 * g_du["passou"].mean(),
            "duelos_em_trem_pct": 100 * (g_du["d_trem"] == "defensor em trem de DRS").mean(),
            "rho_largada_mediana": g_co["rho_largada"].median(),
            "corridas_com_sc_pct": 100 * (g_co["d_safety_car"] == "com safety car").mean(),
            "paradas_por_piloto": g_pc["n_paradas"].mean(),
            "taxa_parada_por_volta_pct": 100 * g_pa["parou"].mean(),
            "maior_stint_voltas_mediana": g_pc["stint_mais_longo"].median(),
            "degradacao_mediana_pct_volta": g_st["deg_pct"].median(),
        }

    linhas = [{"perfil": "todos", **medidas(du, co, pa, st, pc)}]
    for p in ["alto downforce", "stop-and-go", "potência", "urbano-técnico"]:
        f = lambda d: d[d["d_perfil"] == p]
        linhas.append({"perfil": p, **medidas(f(du), f(co), f(pa), f(st), f(pc))})
    vg = pd.DataFrame(linhas)
    vg.to_csv(SAI / "visao_geral.csv", index=False)

    pcirc = pd.DataFrame({
        "perfil": mapa,
        "corridas": du.groupby("circuito").apply(lambda g: g.groupby(["ano", "round"]).ngroups),
        "taxa_ultrapassagem_pct": 100 * du.groupby("circuito")["passou"].mean(),
        "duelos_em_trem_pct": 100 * du.groupby("circuito")["d_trem"].apply(lambda s: (s == "defensor em trem de DRS").mean()),
        "rho_largada_mediana": co.groupby("circuito")["rho_largada"].median(),
        "paradas_por_piloto": pc.groupby("circuito")["n_paradas"].mean(),
        "degradacao_mediana_pct_volta": st.groupby("circuito")["deg_pct"].median(),
    })
    pcirc.index.name = "circuito"
    pcirc.reset_index().sort_values(["perfil", "circuito"]).to_csv(SAI / "por_circuito.csv", index=False)
    pd.set_option("display.width", 250)
    print(vg.round(3).T.to_string())
    print(pcirc.round(3).sort_values("perfil").to_string())


if __name__ == "__main__":
    main()
