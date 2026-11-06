"""
Etapa 2 - Monta a base do caminho B: uma linha por corrida.

Pergunta: em que condições a posição de largada deixa de determinar a de chegada?

Entra: data/processed/base_piloto_corrida.csv
       data/interim/voltas/*_voltas.csv
       data/interim/corridas/*_stints.csv
Sai:   data/processed/base_corrida.csv

Não usa internet. Precisa de avalia_piloto_a1.py na mesma pasta.

Alvos:
  rho_largada       principal. Correlação de Spearman entre largada e chegada,
                    entre quem terminou. Perto de 1 = a chegada repetiu a largada.
                    Quem largou do box fica fora da conta.
  trocas_por_volta  secundário. Trocas de posição por volta sob bandeira verde,
                    entre carros que não passaram pelo box naquela volta ou na
                    anterior. Mede disputa na pista, sem contar as trocas do ciclo
                    de paradas. Fica vazio se a corrida tem dado de parada
                    incompleto (mesma regra de 80% do A1) ou menos de MIN_VOLTAS
                    voltas válidas.

Descritores, fixados antes de ver os resultados destes alvos:
  d_perfil       perfil do circuito (Etapa 1)
  d_chuva        chuva em três faixas
  d_safety_car   se a corrida teve safety car
  d_era          2018-21 ou 2022+
  d_paradas      número típico de paradas da corrida (mediana de quem terminou)

Abandono não é alvo: a relação com safety car é óbvia, porque batida causa safety car.
"""
from pathlib import Path

import numpy as np
import pandas as pd

import avalia_piloto_a1 as A

BASE = Path("data/processed/base_piloto_corrida.csv")
VOLTAS = Path("data/interim/voltas")
CORRIDAS = Path("data/interim/corridas")
SAIDA = Path("data/processed/base_corrida.csv")

MIN_VOLTAS = 10
MIN_CONFERIDOS = 0.80
MIN_FINALISTAS = 8
NEUTRALIZACAO = set("4567")
NOMES_PERFIL = {0: "alto downforce", 1: "stop-and-go", 2: "potência", 3: "urbano-técnico"}


def neutralizada(codigo):
    return bool(set(str(codigo)) & NEUTRALIZACAO)


def inversoes(antes, depois):
    """Quantos pares de carros trocaram de ordem entre as duas voltas."""
    a = np.asarray(antes)[np.argsort(antes)]
    d = np.asarray(depois)[np.argsort(antes)]
    return sum(int((d[i + 1:] < d[i]).sum()) for i in range(len(a)))


def trocas_por_volta(arquivo):
    """Trocas por volta verde e número de voltas usadas."""
    v = pd.read_csv(arquivo, dtype={"bandeira": str}).dropna(subset=["volta", "posicao"])
    v["volta"] = v["volta"].astype(int)
    v["box"] = v["entrada_box_s"].notna() | v["saida_box_s"].notna()
    verde = ~v.groupby("volta")["bandeira"].agg(lambda s: any(map(neutralizada, s)))
    trocas, usadas = 0, 0
    for volta in sorted(v["volta"].unique()):
        anterior = volta - 1
        # A volta 1 fica fora: a largada não é disputa volta a volta.
        if anterior < 2 or anterior not in verde.index or not (verde[volta] and verde[anterior]):
            continue
        a = v[v["volta"] == anterior].set_index("piloto")
        d = v[v["volta"] == volta].set_index("piloto")
        comum = a.index.intersection(d.index)
        comum = comum[~a.loc[comum, "box"].to_numpy() & ~d.loc[comum, "box"].to_numpy()]
        if len(comum) < 2:
            continue
        trocas += inversoes(a.loc[comum, "posicao"].to_numpy(), d.loc[comum, "posicao"].to_numpy())
        usadas += 1
    return (trocas / usadas if usadas >= MIN_VOLTAS else np.nan), usadas


def paradas_conferidas(arquivo, ano, rnd):
    """Fração de pilotos com o número de paradas igual ao dos stints.
    Usa a mesma detecção de parada do A1: entrada no box, saída e troca de pneu."""
    par = A.acha_paradas(A.le_voltas(arquivo))
    s = pd.read_csv(CORRIDAS / f"{ano}_{rnd:02d}_stints.csv")
    esperadas = s.groupby("piloto").size() - 1
    medidas = par[par["trocou_pneu"]].groupby("piloto").size()
    return (medidas.reindex(esperadas.index, fill_value=0) == esperadas).mean()


def faixa_paradas(m):
    return "até 1 parada" if m <= 1 else "2 paradas" if m <= 2 else "3+ paradas"


def main():
    base = pd.read_csv(BASE)
    base = base[~base["corrida_sem_disputa"]]
    linhas, sem_voltas, fora = [], 0, []
    for (ano, rnd), g in base.groupby(["ano", "round"]):
        fin = g[g["terminou"] & (g["grid"] > 0)]
        if len(fin) < MIN_FINALISTAS:
            continue
        arquivo = VOLTAS / f"{ano}_{rnd:02d}_voltas.csv"
        trocas, usadas = np.nan, 0
        if arquivo.exists():
            trocas, usadas = trocas_por_volta(arquivo)
            if paradas_conferidas(arquivo, ano, rnd) < MIN_CONFERIDOS:
                trocas = np.nan
                fora.append(f"{ano} r{rnd}")
        else:
            sem_voltas += 1
        linha = g.iloc[0]
        linhas.append({
            "ano": ano, "round": rnd, "circuito": linha["circuito"],
            "rho_largada": fin["grid"].corr(fin["posicao_final"], method="spearman"),
            "trocas_por_volta": trocas, "voltas_verdes": usadas,
            "finalistas": len(fin),
            "corrida_encurtada": linha["corrida_encurtada"],
            "d_perfil": NOMES_PERFIL[linha["cluster"]],
            "d_chuva": linha["faixa_chuva"],
            "d_safety_car": "com safety car" if linha["n_periodos_sc"] > 0 else "sem safety car",
            "d_era": "2022+" if ano >= 2022 else "2018-21",
            "d_paradas": faixa_paradas(fin["n_paradas"].median()),
        })
    df = pd.DataFrame(linhas)
    df.to_csv(SAIDA, index=False)

    print(f"Corridas: {len(df)}   circuitos: {df['circuito'].nunique()}")
    print(f"Sem arquivo de voltas: {sem_voltas}")
    print(f"Com trocas_por_volta: {df['trocas_por_volta'].notna().sum()}")
    if fora:
        print("Sem trocas_por_volta por dado de parada incompleto: " + ", ".join(fora))
    print("\nOs descritores:")
    for c in [c for c in df.columns if c.startswith("d_")]:
        print(f"\n  {c}")
        print(df[c].value_counts().to_string())
    print(f"\nSalvo em {SAIDA}")
    print("Nenhum valor dos alvos dentro de nenhum grupo foi examinado aqui.")


if __name__ == "__main__":
    main()
