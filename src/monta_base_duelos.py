"""
Etapa 2 - Monta a base de duelos: uma linha por volta de duelo, em corrida seca.

Pergunta: em que condições o carro de trás consegue passar?

Entra: data/interim/voltas/*_voltas.csv
       data/interim/corridas/*_stints.csv
       data/processed/base_piloto_corrida.csv
Sai:   data/processed/base_duelos.csv

Não usa internet. Precisa de avalia_piloto_a1.py na mesma pasta.

Volta de duelo (mesma definição do piloto, com uma mudança): ao fim da volta
anterior, o atacante está logo atrás do defensor, a até 1,0 s; as duas voltas são
de bandeira verde; nenhum dos dois entra ou sai do box nelas; os dois completam a
volta. Mudança: só a partir da volta 3, porque o DRS fica desligado nas duas
primeiras voltas.
Alvo: `passou`, o atacante está à frente ao fim da volta.

Ficam de fora: corridas com chuva, corrida sem disputa efetiva, corridas com dado de
parada incompleto (regra de 80% do A1, porque uma parada não registrada viraria uma
ultrapassagem falsa) e voltas em que algum dos dois está com pneu de chuva.

Descritores, fixados antes de ver os resultados:
  d_perfil        perfil do circuito (Etapa 1)
  d_era           2018-21 ou 2022+
  d_dif_idade     idade do pneu do defensor menos a do atacante:
                  atacante 10+ voltas mais novo / 5 a 9 / diferença de até 4 /
                  defensor 5+ voltas mais novo
  d_dif_composto  atacante mais macio / igual / mais duro (dureza relativa)
  d_distancia     até 0,3 s / 0,3 a 0,6 s / 0,6 a 1,0 s
  d_trem          o defensor também está a até 1,0 s do carro da frente
  d_largada       quem largou à frente (largada do box conta como a última)
  d_fase          terço da corrida
  d_relargada     a volta é uma das 3 seguintes a uma neutralização
  d_posicao       posição disputada: top 5 / 6.º a 10.º / 11.º em diante
  d_mesma_equipe  os dois são da mesma equipe
"""
from pathlib import Path

import numpy as np
import pandas as pd

import avalia_piloto_a1 as A

BASE = Path("data/processed/base_piloto_corrida.csv")
SAIDA = Path("data/processed/base_duelos.csv")

DISTANCIA_S = 1.0
VOLTA_MINIMA = 3
VOLTAS_RELARGADA = 3
MIN_CONFERIDOS = 0.80
SECOS = ["HYPERSOFT", "ULTRASOFT", "SUPERSOFT", "SOFT", "MEDIUM", "HARD"]
NOMES_PERFIL = {0: "alto downforce", 1: "stop-and-go", 2: "potência", 3: "urbano-técnico"}


def paradas_conferidas(arquivo, ano, rnd):
    par = A.acha_paradas(A.le_voltas(arquivo))
    s = pd.read_csv(A.CORRIDAS / f"{ano}_{rnd:02d}_stints.csv")
    esperadas = s.groupby("piloto").size() - 1
    medidas = par[par["trocou_pneu"]].groupby("piloto").size()
    return (medidas.reindex(esperadas.index, fill_value=0) == esperadas).mean()


def dureza(compostos):
    """Posição de cada composto seco na escolha do fim de semana (0 = mais macio)."""
    presentes = [c for c in SECOS if c in compostos]
    return {c: i for i, c in enumerate(presentes)}


def faixa_idade(d):
    if d >= 10:
        return "atacante 10+ voltas mais novo"
    if d >= 5:
        return "atacante 5 a 9 voltas mais novo"
    if d > -5:
        return "diferença de até 4 voltas"
    return "defensor 5+ voltas mais novo"


def faixa_distancia(g):
    return "até 0,3 s" if g <= 0.3 else "0,3 a 0,6 s" if g <= 0.6 else "0,6 a 1,0 s"


def faixa_posicao(p):
    return "top 5" if p <= 5 else "6.º a 10.º" if p <= 10 else "11.º em diante"


def duelos(v, info):
    v = v.dropna(subset=["volta", "posicao"]).copy()
    v["volta"] = v["volta"].astype(int)
    v["box"] = v["entrada_box_s"].notna() | v["saida_box_s"].notna()
    verde = v.groupby("volta")["bandeira"].agg(lambda s: (s.astype(str) == "1").mean() >= 0.8)
    neutra = sorted(verde.index[~verde.to_numpy()])
    ordem = dureza(set(v["composto"].dropna()))
    por_volta = {k: g.set_index("piloto") for k, g in v.groupby("volta")}
    grid, equipe, total = info["grid"], info["equipe"], info["voltas"]

    linhas = []
    for volta in sorted(por_volta):
        antes = volta - 1
        if volta < VOLTA_MINIMA or antes not in por_volta:
            continue
        if not (verde.get(volta, False) and verde.get(antes, False)):
            continue
        ant = por_volta[antes].sort_values("posicao")
        atu = por_volta[volta]
        nomes = ant.index.to_list()
        relargada = any(antes - VOLTAS_RELARGADA <= n < antes for n in neutra)
        for i in range(1, len(nomes)):
            defe, ata = nomes[i - 1], nomes[i]
            if ant.at[ata, "posicao"] - ant.at[defe, "posicao"] != 1:
                continue
            if defe not in atu.index or ata not in atu.index:
                continue
            gap = ant.at[ata, "fim_volta_s"] - ant.at[defe, "fim_volta_s"]
            if not (0 < gap <= DISTANCIA_S):
                continue
            if ant.at[defe, "box"] or ant.at[ata, "box"] or atu.at[defe, "box"] or atu.at[ata, "box"]:
                continue
            cd, ca = atu.at[defe, "composto"], atu.at[ata, "composto"]
            if cd not in ordem or ca not in ordem:
                continue
            idade_d, idade_a = atu.at[defe, "vida_pneu"], atu.at[ata, "vida_pneu"]
            if pd.isna(idade_d) or pd.isna(idade_a):
                continue
            trem = False
            if i >= 2:
                frente = nomes[i - 2]
                trem = 0 < ant.at[defe, "fim_volta_s"] - ant.at[frente, "fim_volta_s"] <= DISTANCIA_S
            gd, ga = grid.get(defe, np.nan), grid.get(ata, np.nan)
            if pd.isna(gd) or pd.isna(ga):
                continue
            linhas.append({
                "volta": volta, "defensor": defe, "atacante": ata, "distancia_s": gap,
                "passou": int(atu.at[ata, "posicao"] < atu.at[defe, "posicao"]),
                "dif_idade": idade_d - idade_a,
                "d_dif_idade": faixa_idade(idade_d - idade_a),
                "d_dif_composto": ("atacante mais macio" if ordem[ca] < ordem[cd] else
                                   "atacante mais duro" if ordem[ca] > ordem[cd] else "mesmo composto"),
                "d_distancia": faixa_distancia(gap),
                "d_trem": "defensor em trem de DRS" if trem else "defensor livre",
                "d_largada": "atacante largou à frente" if ga < gd else "defensor largou à frente",
                "d_fase": ("primeiro terço" if volta <= total / 3 else
                           "segundo terço" if volta <= 2 * total / 3 else "último terço"),
                "d_relargada": "após relargada" if relargada else "corrida normal",
                "d_posicao": faixa_posicao(ant.at[defe, "posicao"]),
                "d_mesma_equipe": ("mesma equipe" if equipe.get(defe) == equipe.get(ata)
                                   else "equipes diferentes"),
            })
    return linhas


def main():
    base = pd.read_csv(BASE)
    todas, fora = [], {"chuva": 0, "sem_disputa": 0, "paradas": 0, "sem_arquivo": 0}
    for (ano, rnd), g in base.groupby(["ano", "round"]):
        linha = g.iloc[0]
        if linha["corrida_sem_disputa"]:
            fora["sem_disputa"] += 1
            continue
        if linha["faixa_chuva"] != "seco":
            fora["chuva"] += 1
            continue
        arquivo = A.VOLTAS / f"{ano}_{rnd:02d}_voltas.csv"
        if not arquivo.exists():
            fora["sem_arquivo"] += 1
            continue
        if paradas_conferidas(arquivo, ano, rnd) < MIN_CONFERIDOS:
            fora["paradas"] += 1
            continue
        largou = g[~g["nao_largou"]]
        grid = largou.set_index("piloto")["grid"].replace(0, len(g) + 1)
        info = {"grid": grid.to_dict(), "equipe": g.set_index("piloto")["equipe"].to_dict(),
                "voltas": linha["voltas_corrida"]}
        v = pd.read_csv(arquivo, dtype={"bandeira": str})
        for d in duelos(v, info):
            d.update({"ano": ano, "round": rnd, "circuito": linha["circuito"],
                      "d_perfil": NOMES_PERFIL[linha["cluster"]],
                      "d_era": "2022+" if ano >= 2022 else "2018-21"})
            todas.append(d)

    df = pd.DataFrame(todas)
    frente = ["ano", "round", "circuito", "volta", "defensor", "atacante", "passou"]
    df = df[frente + [c for c in df.columns if c not in frente]]
    df.to_csv(SAIDA, index=False)
    print(f"Corridas fora: {fora}")
    print(f"Voltas de duelo: {len(df)}, corridas: {df.groupby(['ano', 'round']).ngroups}, "
          f"circuitos: {df['circuito'].nunique()}")
    print("\nOs descritores:")
    for c in [c for c in df.columns if c.startswith("d_")]:
        print(f"\n  {c}")
        print(df[c].value_counts().to_string())
    print(f"\nSalvo em {SAIDA}")
    print("A taxa de ultrapassagem dentro de cada grupo não foi examinada aqui.")


if __name__ == "__main__":
    main()
