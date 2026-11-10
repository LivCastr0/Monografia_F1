"""
Etapa 2 - Opção 5: monta a base da decisão de parar.

Pergunta: em que situações as equipes decidem parar?

Entra: data/interim/voltas/*_voltas.csv
       data/interim/corridas/*_stints.csv
       data/processed/base_piloto_corrida.csv
       data/processed/features_completas_por_circuito.csv   (tempo perdido no box)
Sai:   data/processed/base_decisao_parada.csv

Não usa internet. Precisa de avalia_piloto_a1.py na mesma pasta.

Grão: piloto em cada volta L, em corrida seca. Alvo `parou`: o piloto entra no box na
volta L e troca de pneu (mesma detecção de parada do A1).

Os descritores descrevem a situação no fim da volta L-1, antes da decisão. A exceção
é a neutralização, medida no ponto de decisão da volta L: safety car e VSC aparecem
durante a volta e a equipe decide na hora.

Neutralização (corrigida em 2026-11-10, ver verifica_horario_sc_parada.py): SC ou VSC
ativo no ponto de decisão de cada piloto, pelo horário dos avisos de bandeira
(data/interim/corridas/*_bandeiras.csv, códigos 4, 6 e 7). Voltas com bandeira
vermelha no ponto de decisão saem da base: a troca de pneu sob vermelha é livre e
geral, não é decisão (antes contavam como neutralização; eram 97 paradas). Ponto de decisão = entrada
do box para quem parou, fim da volta L para quem não parou (em média 2,6 s depois).
Antes, a neutralização valia para a volta L inteira e juntava a bandeira de todos os
pilotos; com isso, paradas feitas antes do SC ou depois de ele sair contavam como "sob
SC", e voltas de quem nunca viu o SC também.

Ficam de fora: voltas com bandeira vermelha no ponto de decisão, volta 1, a última volta, a volta de saída do box, voltas com pneu de
chuva, corridas com chuva, corrida sem disputa efetiva e corridas com dado de
parada incompleto (regra de 80% do A1).

Descritores, fixados antes de ver os resultados:
  d_idade_pneu      0–9, 10–19, 20–29, 30+ voltas
  d_composto        macio, médio, duro (relativo ao fim de semana)
  d_neutralizacao   safety car ou VSC na volta, ou bandeira verde
  d_rival_frente    o carro logo à frente parou na volta anterior
  d_rival_tras      o carro logo atrás parou na volta anterior
  d_janela          a distância para o carro de trás é maior que o tempo perdido no
                    box daquele circuito (cabe a parada sem perder a posição)
  d_dist_frente     até 1 s, 1 a 3 s, mais de 3 s (líder conta como mais de 3 s)
  d_paradas_feitas  0, 1, 2 ou mais
  d_fase            terço da corrida
  d_posicao         top 5, 6.º a 10.º, 11.º em diante
  d_perfil          perfil do circuito (Etapa 1)
  d_era             2018-21 ou 2022+
"""
from pathlib import Path

import numpy as np
import pandas as pd

import avalia_piloto_a1 as A

BASE = Path("data/processed/base_piloto_corrida.csv")
FEATURES = Path("data/processed/features_completas_por_circuito.csv")
SAIDA = Path("data/processed/base_decisao_parada.csv")

MIN_CONFERIDOS = 0.80
NEUTRALIZACAO = {"4", "6", "7"}   # SC, VSC, VSC terminando
VERMELHA = "5"
SECOS = ["HYPERSOFT", "ULTRASOFT", "SUPERSOFT", "SOFT", "MEDIUM", "HARD"]
NOMES_PERFIL = {0: "alto downforce", 1: "stop-and-go", 2: "potência", 3: "urbano-técnico"}


def dureza(compostos):
    p = [c for c in SECOS if c in compostos]
    if len(p) >= 3:
        r = ["macio"] + ["médio"] * (len(p) - 2) + ["duro"]
    elif len(p) == 2:
        r = ["macio", "duro"]
    else:
        r = ["médio"] * len(p)
    return dict(zip(p, r))


def faixa(x, cortes, rotulos):
    for c, r in zip(cortes, rotulos):
        if x <= c:
            return r
    return rotulos[-1]


def corrida(arquivo, info):
    v = A.le_voltas(arquivo)
    v = v.dropna(subset=["posicao"]).copy()
    par = A.acha_paradas(v)
    paradas = par[par["trocou_pneu"]][["piloto", "volta_parada"]]
    parou = set(zip(paradas["piloto"], paradas["volta_parada"]))

    s = pd.read_csv(A.CORRIDAS / f"{info['ano']}_{info['round']:02d}_stints.csv")
    esperadas = s.groupby("piloto").size() - 1
    medidas = paradas.groupby("piloto").size().reindex(esperadas.index, fill_value=0)
    if (medidas == esperadas).mean() < MIN_CONFERIDOS:
        return None

    rotulo = dureza(set(v["composto"].dropna()))
    b = pd.read_csv(A.CORRIDAS / f"{info['ano']}_{info['round']:02d}_bandeiras.csv") \
        .sort_values("tempo_s")
    avisos_t, avisos_cod = b["tempo_s"].to_numpy(), b["codigo"].astype(str).to_numpy()

    def bandeira_em(t):
        if pd.isna(t):
            return "1"
        i = np.searchsorted(avisos_t, t, side="right") - 1
        return avisos_cod[max(i, 0)]
    ultima = int(v["volta"].max())

    # Ordem e distâncias no fim de cada volta
    v = v.sort_values(["volta", "posicao"])
    g = v.groupby("volta")
    v["frente"] = g["piloto"].shift(1)
    v["tras"] = g["piloto"].shift(-1)
    v["dist_frente"] = v["fim_volta_s"] - g["fim_volta_s"].shift(1)
    v["dist_tras"] = g["fim_volta_s"].shift(-1) - v["fim_volta_s"]
    v["parou_aqui"] = [(p, l) in parou for p, l in zip(v["piloto"], v["volta"])]
    v["saida"] = v["saida_box_s"].notna()
    v = v.sort_values(["piloto", "volta"])
    v["paradas_feitas"] = v.groupby("piloto")["parou_aqui"].cumsum() - v["parou_aqui"]

    estado = v.set_index(["piloto", "volta"])
    # O estado no fim da volta L-1 vira a linha da volta L
    ant = v.rename(columns=lambda c: c + "_ant").rename(columns={"piloto_ant": "piloto"})
    ant["volta"] = ant["volta_ant"] + 1
    v["t_decisao"] = np.where(v["parou_aqui"] & v["entrada_box_s"].notna(),
                              v["entrada_box_s"], v["fim_volta_s"])
    atual = v[["piloto", "volta", "parou_aqui", "saida", "composto", "t_decisao"]]
    d = atual.merge(ant, on=["piloto", "volta"])
    d = d[(d["volta"] >= 2) & (d["volta"] < ultima) & ~d["saida"] & ~d["saida_ant"]]

    def parou_em(piloto, volta):
        return isinstance(piloto, str) and (piloto, volta) in parou

    d["rival_frente_parou"] = [parou_em(p, l - 1) for p, l in zip(d["frente_ant"], d["volta"])]
    d["rival_tras_parou"] = [parou_em(p, l - 1) for p, l in zip(d["tras_ant"], d["volta"])]
    d = d[d["composto_ant"].isin(rotulo.keys())]
    # Bandeira vermelha no ponto de decisão: troca de pneu livre, não é decisão; sai.
    d = d.assign(bandeira_decisao=[bandeira_em(t) for t in d["t_decisao"]])
    d = d[d["bandeira_decisao"] != VERMELHA]

    total = info["voltas"]
    perda = info["perda_box"]
    linhas = pd.DataFrame({
        "ano": info["ano"], "round": info["round"], "circuito": info["circuito"],
        "piloto": d["piloto"].to_numpy(), "volta": d["volta"].to_numpy(),
        "parou": d["parou_aqui"].astype(int).to_numpy(),
        "idade_pneu": d["vida_pneu_ant"].to_numpy(),
        "d_idade_pneu": [faixa(x, [9, 19, 29], ["0–9 voltas", "10–19 voltas", "20–29 voltas",
                                                "30+ voltas"]) for x in d["vida_pneu_ant"]],
        "d_composto": d["composto_ant"].map(rotulo).to_numpy(),
        "d_neutralizacao": np.where(d["bandeira_decisao"].isin(NEUTRALIZACAO),
                                    "safety car ou VSC", "bandeira verde"),
        "d_rival_frente": np.where(d["rival_frente_parou"], "rival da frente parou",
                                   "rival da frente não parou"),
        "d_rival_tras": np.where(d["rival_tras_parou"], "rival de trás parou",
                                 "rival de trás não parou"),
        "d_janela": np.where(d["dist_tras_ant"].fillna(np.inf) > perda, "cabe a parada",
                             "não cabe a parada"),
        "d_dist_frente": [faixa(x, [1, 3], ["até 1 s", "1 a 3 s", "mais de 3 s"])
                          if pd.notna(x) else "mais de 3 s" for x in d["dist_frente_ant"]],
        "d_paradas_feitas": [faixa(x, [0, 1], ["0 paradas", "1 parada", "2+ paradas"])
                             for x in d["paradas_feitas_ant"] + d["parou_aqui_ant"]],
        "d_fase": ["primeiro terço" if l <= total / 3 else "segundo terço" if l <= 2 * total / 3
                   else "último terço" for l in d["volta"]],
        "d_posicao": [faixa(x, [5, 10], ["top 5", "6.º a 10.º", "11.º em diante"])
                      for x in d["posicao_ant"]],
        "d_perfil": info["perfil"],
        "d_era": "2022+" if info["ano"] >= 2022 else "2018-21",
    })
    return linhas.dropna(subset=["d_idade_pneu", "d_composto"])


def main():
    base = pd.read_csv(BASE)
    perda = pd.read_csv(FEATURES).set_index("circuito")["pit_lane_time_loss_s"]
    partes, fora = [], {"chuva": 0, "sem_disputa": 0, "paradas": 0, "sem_arquivo": 0}
    for (ano, rnd), g in base.groupby(["ano", "round"]):
        l = g.iloc[0]
        if l["corrida_sem_disputa"]:
            fora["sem_disputa"] += 1
            continue
        if l["faixa_chuva"] != "seco":
            fora["chuva"] += 1
            continue
        arquivo = A.VOLTAS / f"{ano}_{rnd:02d}_voltas.csv"
        if not arquivo.exists():
            fora["sem_arquivo"] += 1
            continue
        info = {"ano": int(ano), "round": int(rnd), "circuito": l["circuito"],
                "voltas": l["voltas_corrida"], "perda_box": perda[l["circuito"]],
                "perfil": NOMES_PERFIL[l["cluster"]]}
        d = corrida(arquivo, info)
        if d is None:
            fora["paradas"] += 1
            continue
        partes.append(d)
    df = pd.concat(partes, ignore_index=True)
    df.to_csv(SAIDA, index=False)
    print(f"Corridas fora: {fora}")
    print(f"Linhas (piloto-volta): {len(df)}, paradas: {int(df['parou'].sum())}, "
          f"corridas: {df.groupby(['ano', 'round']).ngroups}, circuitos: {df['circuito'].nunique()}")
    print("\nOs descritores:")
    for c in [c for c in df.columns if c.startswith("d_")]:
        print(f"\n  {c}")
        print(df[c].value_counts().to_string())
    print(f"\nSalvo em {SAIDA}")
    print("A taxa de parada dentro de cada grupo não foi examinada aqui.")


if __name__ == "__main__":
    main()
