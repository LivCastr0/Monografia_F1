"""
Etapa 2 - Monta a base de afinidade carro × perfil: uma linha por carro em cada
corrida seca.

Entra: data/interim/voltas/*_voltas.csv
       data/processed/base_piloto_corrida.csv
Sai:   data/processed/base_ritmo.csv          (carro por corrida)
       data/processed/base_ritmo_pilotos.csv  (piloto por corrida, para verificação)

Não usa internet. Precisa de avalia_piloto_ritmo.py na mesma pasta.

Ritmo: o mesmo do piloto (voltas limpas, ajuste de pneu e combustível por corrida,
mediana dos restos em % do tempo mediano). Entra o carro com os dois pilotos medidos.
Em cada corrida, o ritmo é centrado na média dos carros.

Carro = equipe e ano (o carro muda a cada temporada).

Alvo `afinidade`: quanto o carro andou mais rápido que o próprio padrão na temporada,
em % do tempo de volta. Positivo = mais rápido que o normal dele. O padrão é a média
do carro nas outras corridas secas da temporada (a própria corrida fica fora), e o
carro precisa de pelo menos MIN_OUTRAS outras corridas. O mesmo é feito por piloto.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from avalia_piloto_ritmo import BASE, VOLTAS, ritmo_por_piloto

SAIDA = Path("data/processed/base_ritmo.csv")
SAIDA_PILOTOS = Path("data/processed/base_ritmo_pilotos.csv")
MIN_OUTRAS = 4
NOMES_PERFIL = {0: "alto downforce", 1: "stop-and-go", 2: "potência", 3: "urbano-técnico"}


def padrao_sem_a_propria(df, chave, valor):
    """Média de cada grupo sem a própria linha, e quantas outras linhas há."""
    g = df.groupby(chave)[valor]
    soma, n = g.transform("sum"), g.transform("size")
    return (soma - df[valor]) / (n - 1), n - 1


def main():
    base = pd.read_csv(BASE).drop_duplicates(["ano", "round"])
    secas = base[(base["faixa_chuva"] == "seco") & ~base["corrida_sem_disputa"]]
    carros, pilotos = [], []
    for _, l in secas.iterrows():
        ano, rnd = int(l["ano"]), int(l["round"])
        arquivo = VOLTAS / f"{ano}_{rnd:02d}_voltas.csv"
        if not arquivo.exists():
            continue
        r = ritmo_por_piloto(pd.read_csv(arquivo, dtype={"bandeira": str}))
        if r.empty:
            continue
        r = r[r.groupby("equipe")["piloto"].transform("size") == 2].copy()
        if r.empty:
            continue
        info = {"ano": ano, "round": rnd, "circuito": l["circuito"],
                "d_perfil": NOMES_PERFIL[l["cluster"]]}
        c = r.groupby("equipe")["ritmo"].mean().reset_index()
        c["ritmo"] -= c["ritmo"].mean()
        r["ritmo"] -= r["ritmo"].mean()
        carros.append(c.assign(**info))
        pilotos.append(r.assign(**info))

    df = pd.concat(carros, ignore_index=True)
    df["d_carro"] = df["equipe"] + " " + df["ano"].astype(str)
    padrao, outras = padrao_sem_a_propria(df, "d_carro", "ritmo")
    df["afinidade"] = -(df["ritmo"] - padrao)
    df["outras_corridas"] = outras
    df = df[df["outras_corridas"] >= MIN_OUTRAS]

    p = pd.concat(pilotos, ignore_index=True)
    p["d_carro"] = p["equipe"] + " " + p["ano"].astype(str)
    p["chave"] = p["d_carro"] + " " + p["piloto"]
    padrao_p, outras_p = padrao_sem_a_propria(p, "chave", "ritmo")
    p["afinidade"] = -(p["ritmo"] - padrao_p)
    p = p[outras_p >= MIN_OUTRAS]

    df.to_csv(SAIDA, index=False)
    p.to_csv(SAIDA_PILOTOS, index=False)
    print(f"Carros-corrida: {len(df)}, carros: {df['d_carro'].nunique()}, "
          f"corridas: {df.groupby(['ano', 'round']).ngroups}, circuitos: {df['circuito'].nunique()}")
    print(f"Desvio da afinidade: {df['afinidade'].std():.3f}% do tempo de volta")
    print("\nCorridas por carro em cada perfil (mediana):")
    print(df.groupby(["d_carro", "d_perfil"]).size().groupby("d_perfil").median().to_string())
    print(f"\nSalvo em {SAIDA} e {SAIDA_PILOTOS}")
    print("A afinidade dentro de cada carro e perfil não foi examinada aqui.")


if __name__ == "__main__":
    main()
