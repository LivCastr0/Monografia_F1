"""
Etapa 2 - Monta a base de degradação: uma linha por stint em corrida seca.

Entra: data/interim/voltas/*_voltas.csv
       data/interim/corridas/*_stints.csv
       data/processed/base_piloto_corrida.csv
       data/processed/contexto_corridas.csv
       data/processed/pirelli_agregado_por_circuito.csv
       data/external/circuit_length.csv
Sai:   data/processed/base_stint.csv

Não usa internet. Precisa de avalia_piloto_a1.py na mesma pasta.

A medida: quanto o tempo de volta piora a cada volta de vida do pneu, dentro de um
stint. É a inclinação (Theil-Sen, resistente a voltas estranhas) do tempo contra a
vida do pneu, em % do tempo de volta típico daquele stint.

Voltas limpas: precisas segundo a FastF1, bandeira verde em todo o trecho, fora de
entrada e saída do box, sem as 2 primeiras voltas do stint (aquecimento do pneu) e a
pelo menos TRAFEGO_S segundos do carro da frente. Vale o stint com pelo menos
MIN_VOLTAS voltas limpas, cobrindo pelo menos MIN_VIDA voltas de vida do pneu.

Combustível: o carro fica mais leve a cada volta, o que esconde parte da degradação.
Dentro do stint isso não pode ser separado da idade do pneu, porque as duas andam
juntas, então a correção é uma SUPOSIÇÃO externa, não uma medida:
    ganho por volta = (combustível ÷ voltas da corrida) × s/kg × (comprimento ÷ 5 km)
com 105 kg em 2018, 110 kg depois, e 0,03 s/kg. deg_pct_baixo e deg_pct_alto usam
0,02 e 0,04 s/kg, para verificar se o resultado depende dessa suposição.

Limitações: a pista fica mais rápida ao longo da corrida (evolução), o que também
esconde degradação; o piloto pode poupar pneu de propósito; só entram stints sob
bandeira verde e fora de tráfego.

Corridas com chuva ou com dado de parada incompleto ficam de fora.

Descritores (prefixo d_), fixados antes de ver a medida:
  d_composto      dureza relativa à escolha do fim de semana
  d_pneu_novo     pneu novo ou usado
  d_ordem_stint   primeiro stint ou seguintes
  d_temp_pista    temperatura média da pista, em terços
  d_abrasao, d_estresse_pneu, d_lateral   notas Pirelli do circuito, em terços
  d_perfil        perfil da Etapa 1
  d_era           2018-21 ou 2022+
"""
from pathlib import Path

import numpy as np
import pandas as pd

import avalia_piloto_a1 as A

BASE = Path("data/processed/base_piloto_corrida.csv")
CONTEXTO = Path("data/processed/contexto_corridas.csv")
PIRELLI = Path("data/processed/pirelli_agregado_por_circuito.csv")
COMPRIMENTO = Path("data/external/circuit_length.csv")
SAIDA = Path("data/processed/base_stint.csv")

TRAFEGO_S = 1.5
MIN_VOLTAS = 8
MIN_VIDA = 6
AQUECIMENTO = 2
MIN_CONFERIDOS = 0.80
S_POR_KG = {"deg_pct_baixo": 0.02, "deg_pct": 0.03, "deg_pct_alto": 0.04}
SECOS = ["HYPERSOFT", "ULTRASOFT", "SUPERSOFT", "SOFT", "MEDIUM", "HARD"]
NOTAS = {"d_abrasao": "#7_asphalt_abrasion_1_5",
         "d_estresse_pneu": "pirelli_tyre_stress_1_5",
         "d_lateral": "pirelli_lateral_1_5"}
NOMES_PERFIL = {0: "alto downforce", 1: "stop-and-go", 2: "potência", 3: "urbano-técnico"}


def e_verdadeiro(serie):
    return serie.astype(str).str.lower() == "true"


def inclinacao(x, y):
    """Theil-Sen: a mediana das inclinações entre todos os pares de voltas."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    i, j = np.triu_indices(len(x), k=1)
    dx = x[j] - x[i]
    ok = dx != 0
    return float(np.median((y[j] - y[i])[ok] / dx[ok]))


def dureza_relativa(compostos):
    presentes = [c for c in SECOS if c in compostos]
    n = len(presentes)
    if n >= 3:
        rotulos = ["macio"] + ["médio"] * (n - 2) + ["duro"]
    elif n == 2:
        rotulos = ["macio", "duro"]
    else:
        rotulos = ["médio"] * n
    return dict(zip(presentes, rotulos))


def voltas_limpas(v):
    v = v.sort_values(["volta", "posicao"]).copy()
    # Distância para o carro da frente, entre carros na mesma volta
    v["distancia"] = v.groupby("volta")["fim_volta_s"].diff()
    v.loc[v["posicao"] == v.groupby("volta")["posicao"].transform("min"), "distancia"] = np.inf
    # A bandeira da volta inteira: qualquer carro em amarela ou neutralização a tira
    verde = v.groupby("volta")["bandeira"].agg(lambda s: (s.astype(str) == "1").mean() >= 0.8)
    v = v[v["volta"].map(verde)
          & e_verdadeiro(v["volta_precisa"])
          & ~e_verdadeiro(v["gerada_pela_fastf1"])
          & (v["bandeira"].astype(str) == "1")
          & v["entrada_box_s"].isna() & v["saida_box_s"].isna()
          & v["tempo_volta_s"].notna() & v["vida_pneu"].notna()
          & (v["volta"] > 1) & (v["distancia"] >= TRAFEGO_S)]
    return v


def stints_da_corrida(arquivo, info):
    ano, rnd = (int(t) for t in arquivo.name.split("_")[:2])
    v = pd.read_csv(arquivo, dtype={"bandeira": str}).dropna(subset=["volta", "posicao"])
    v["volta"] = v["volta"].astype(int)

    par = A.acha_paradas(A.le_voltas(arquivo))
    s = pd.read_csv(A.CORRIDAS / f"{ano}_{rnd:02d}_stints.csv")
    esperadas = s.groupby("piloto").size() - 1
    medidas = par[par["trocou_pneu"]].groupby("piloto").size()
    if (medidas.reindex(esperadas.index, fill_value=0) == esperadas).mean() < MIN_CONFERIDOS:
        return [], "paradas"

    rotulo = dureza_relativa(set(v["composto"].dropna()))
    primeira = v.groupby(["piloto", "stint"])["volta"].transform("min")
    v = v[v["volta"] >= primeira + AQUECIMENTO]
    limpas = voltas_limpas(v)

    comb = 105 if ano == 2018 else 110
    linhas = []
    for (piloto, stint), g in limpas.groupby(["piloto", "stint"]):
        if len(g) < MIN_VOLTAS or g["vida_pneu"].max() - g["vida_pneu"].min() < MIN_VIDA:
            continue
        if g["composto"].iloc[0] not in rotulo:
            continue
        # Tira voltas muito lentas: erro, disputa ou bandeira azul
        tipico = g["tempo_volta_s"].median()
        g = g[g["tempo_volta_s"] <= 1.05 * tipico]
        if len(g) < MIN_VOLTAS:
            continue
        cru = inclinacao(g["vida_pneu"], g["tempo_volta_s"])
        linha = {"ano": ano, "round": rnd, "piloto": piloto, "stint": stint,
                 "equipe": g["equipe"].iloc[0], "circuito": info["circuito"],
                 "voltas_limpas": len(g), "vida_inicio": g["vida_pneu"].min(),
                 "vida_fim": g["vida_pneu"].max(), "tempo_tipico_s": tipico,
                 "inclinacao_crua_s": cru, "composto": g["composto"].iloc[0],
                 "d_composto": rotulo[g["composto"].iloc[0]],
                 "d_pneu_novo": "pneu novo" if str(g["pneu_novo"].iloc[0]) == "True" else "pneu usado",
                 "d_ordem_stint": "primeiro stint" if stint == v[v["piloto"] == piloto]["stint"].min()
                 else "stints seguintes"}
        for nome, s_kg in S_POR_KG.items():
            ganho = comb / info["voltas"] * s_kg * info["km"] / 5
            linha[nome] = (cru + ganho) / tipico * 100
        linhas.append(linha)
    return linhas, None


def tercos(serie):
    q1, q2 = serie.quantile([1 / 3, 2 / 3])
    return serie.map(lambda x: np.nan if pd.isna(x) else
                     "baixa" if x <= q1 else "média" if x <= q2 else "alta")


def main():
    base = pd.read_csv(BASE).drop_duplicates(["ano", "round"]).set_index(["ano", "round"])
    ctx = pd.read_csv(CONTEXTO).set_index(["ano", "round"])
    comp = pd.read_csv(COMPRIMENTO, sep=";", decimal=",", encoding="utf-8-sig")
    km = dict(zip(comp.iloc[:, 0], comp.iloc[:, 1].astype(float) / 1000))
    pirelli = pd.read_csv(PIRELLI).set_index("circuito")

    todas, fora = [], {"chuva": 0, "sem_disputa": 0, "paradas": 0}
    arquivos = sorted(A.VOLTAS.glob("*_voltas.csv"))
    for a in arquivos:
        ano, rnd = (int(t) for t in a.name.split("_")[:2])
        if (ano, rnd) not in base.index:
            continue
        b = base.loc[(ano, rnd)]
        if b["corrida_sem_disputa"]:
            fora["sem_disputa"] += 1
            continue
        if b["faixa_chuva"] != "seco":
            fora["chuva"] += 1
            continue
        info = {"circuito": b["circuito"], "voltas": b["voltas_corrida"], "km": km[b["circuito"]]}
        linhas, motivo = stints_da_corrida(a, info)
        if motivo:
            fora[motivo] += 1
        todas += linhas

    df = pd.DataFrame(todas)
    chaves = list(zip(df["ano"], df["round"]))
    df["cluster"] = [base.at[k, "cluster"] for k in chaves]
    df["d_perfil"] = df["cluster"].map(NOMES_PERFIL)
    df["d_era"] = np.where(df["ano"] >= 2022, "2022+", "2018-21")
    df["temp_pista"] = [ctx.at[k, "temp_pista_media"] for k in chaves]
    df["d_temp_pista"] = tercos(df["temp_pista"])
    for d, col in NOTAS.items():
        df[d] = df["circuito"].map(tercos(pirelli[col]))

    df.to_csv(SAIDA, index=False)
    print(f"Arquivos de voltas: {len(arquivos)}")
    print(f"Corridas fora: {fora}")
    print(f"Stints: {len(df)}, corridas: {df.groupby(['ano', 'round']).ngroups}, "
          f"circuitos: {df['circuito'].nunique()}")
    print(f"\nSalvo em {SAIDA}")
    print("Nenhum valor da medida dentro de nenhum grupo foi examinado aqui.")


if __name__ == "__main__":
    main()
