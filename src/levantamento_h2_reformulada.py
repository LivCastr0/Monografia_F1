"""
Levantamento da H2 reformulada: os subgrupos de ESTRATÉGIA têm linhas?

A H2 original caía porque um dos descritores era o nome do piloto, e aí o
grupo urbano-técnico travava em 13 corridas. A reformulação mantém o mesmo
alvo (desempenho comparado ao próprio normal do piloto) e troca os descritores
de "quem" para "o que foi feito": paradas, composto de largada, momento da
parada, posição de largada, chuva, safety car.

Este script NÃO roda a busca de subgrupos e NÃO olha resultado nenhum. Ele só
conta linhas: para cada descrição possível, quantas corridas existem em cada
grupo de pista. A pergunta é "dá pra procurar aqui?", não "o que tem aqui" -
olhar a resposta antes de fechar a metodologia é como escolher a hipótese
depois de ver o dado.

Lê a base montada. Não usa internet.
"""
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("data/processed/base_piloto_corrida.csv")
SAIDA = Path("data/processed")
JANELA_MOVEL = 11
LIMIARES = [30, 50, 80, 120]

base = pd.read_csv(BASE)

# Fora: quem não terminou (abandono não tem posição) e as corridas sem disputa
# efetiva (Spa 2021 - a chegada foi a ordem do grid, as linhas são ruído e
# ainda distorcem a referência das corridas vizinhas).
fin = base[base["terminou"] & ~base["corrida_sem_disputa"]].copy()
fin = fin.sort_values(["piloto", "ano", "round"])


# --- o alvo, recalculado aqui para o script ser autossuficiente -------------
def referencia_movel(g):
    y = g["posicao_final"].astype(float)
    soma = y.rolling(JANELA_MOVEL, center=True, min_periods=4).sum()
    cont = y.rolling(JANELA_MOVEL, center=True, min_periods=4).count()
    return (soma - y) / (cont - 1)


fin["ref_movel"] = (
    fin.groupby(["piloto", "ano"], group_keys=False).apply(
        referencia_movel, include_groups=False
    )
)
fin["alvo"] = fin["ref_movel"] - fin["posicao_final"]
desvio = fin["alvo"].std()

print(f"Linhas (só quem terminou): {len(fin)}")
print(f"Desvio do alvo: {desvio:.2f} posições\n")


# ---------------------------------------------------------------------------
# Os descritores candidatos, já em faixas
# ---------------------------------------------------------------------------
def faixa_paradas(n):
    if pd.isna(n):
        return np.nan
    if n <= 1:
        return "0-1 paradas"
    if n == 2:
        return "2 paradas"
    return "3+ paradas"


def faixa_momento(m):
    if pd.isna(m):
        return "sem parada"
    if m < 0.33:
        return "parou cedo"
    if m < 0.66:
        return "parou no meio"
    return "parou tarde"


def faixa_grid(g):
    if pd.isna(g) or g == 0:
        return np.nan
    if g <= 5:
        return "largou top 5"
    if g <= 10:
        return "largou 6-10"
    return "largou 11+"


fin["d_paradas"] = fin["n_paradas"].map(faixa_paradas)
fin["d_momento"] = fin["momento_primeira_parada"].map(faixa_momento)
fin["d_grid"] = fin["grid"].map(faixa_grid)

# Composto pela dureza relativa à escolha do fim de semana, não pelo nome -
# o nome não é comparável entre corridas nem entre anos.
fin["d_composto"] = fin["composto_largada_rel"].map(
    lambda c: f"largou {c}" if pd.notna(c) else np.nan
)

# Chuva em três faixas: a marca booleana disparava com uma só medição.
fin["d_chuva"] = fin["faixa_chuva"]
fin["d_sc"] = np.where(fin["n_periodos_sc"] > 0, "com safety car", "sem safety car")

DESCRITORES = ["d_paradas", "d_momento", "d_grid", "d_composto", "d_chuva", "d_sc"]
NOMES = {
    "d_paradas": "paradas", "d_momento": "momento", "d_grid": "largada",
    "d_composto": "composto", "d_chuva": "clima", "d_sc": "safety car",
}


# ---------------------------------------------------------------------------
# A. Como cada descritor se distribui dentro de cada grupo
# ---------------------------------------------------------------------------
print("=" * 70)
print("A. OS DESCRITORES DENTRO DE CADA GRUPO")
print("=" * 70)
print("(linhas por valor; grupo 3 é Mônaco + Singapura, o que preocupava)\n")

for d in DESCRITORES:
    tab = fin.pivot_table(index=d, columns="cluster", values="ano",
                          aggfunc="size", fill_value=0)
    print(f"--- {NOMES[d]} ---")
    print(tab.to_string())
    print()


# ---------------------------------------------------------------------------
# B. Enumera as descrições possíveis e conta o suporte de cada uma
# ---------------------------------------------------------------------------
print("=" * 70)
print("B. QUANTAS DESCRIÇÕES TÊM LINHAS SUFICIENTES")
print("=" * 70)

linhas_resultado = []

for cluster, g in fin.groupby("cluster"):
    for profundidade in [1, 2, 3]:
        for cols in combinations(DESCRITORES, profundidade):
            valores = [sorted(fin[c].dropna().unique()) for c in cols]
            for combo in product(*valores):
                mask = np.ones(len(g), dtype=bool)
                for c, v in zip(cols, combo):
                    mask &= (g[c] == v).values
                n = int(mask.sum())
                if n == 0:
                    continue
                linhas_resultado.append({
                    "cluster": int(cluster),
                    "profundidade": profundidade,
                    "descricao": " + ".join(combo),
                    "linhas": n,
                    "efeito_detectavel": 2 * desvio / np.sqrt(n),
                })

res = pd.DataFrame(linhas_resultado)

print("Descrições que alcançam cada limiar de linhas, por grupo e profundidade:")
print(f"\n{'prof':>5} {'limiar':>7} {'g0':>6} {'g1':>6} {'g2':>6} {'g3':>6}")
for prof in [1, 2, 3]:
    for lim in LIMIARES:
        sub = res[(res["profundidade"] == prof) & (res["linhas"] >= lim)]
        cont = sub.groupby("cluster").size()
        print(f"{prof:>5} {lim:>7} "
              + " ".join(f"{int(cont.get(c, 0)):>6}" for c in [0, 1, 2, 3]))
    print()


# ---------------------------------------------------------------------------
# C. O grupo que preocupava, de perto
# ---------------------------------------------------------------------------
print("=" * 70)
print("C. GRUPO 3 (MÔNACO + SINGAPURA) DE PERTO")
print("=" * 70)

g3 = res[res["cluster"] == 3].sort_values("linhas", ascending=False)
print(f"Total de linhas do grupo 3 (quem terminou): "
      f"{int(fin[fin['cluster'] == 3].shape[0])}")
print(f"Antes, com nome de piloto como descritor, o teto era 13.\n")

print("As 12 descrições mais populosas do grupo 3:")
print(g3.head(12)[["profundidade", "descricao", "linhas", "efeito_detectavel"]]
      .to_string(index=False, float_format=lambda x: f"{x:.2f}"))

print(f"\nDescrições do grupo 3 com 50+ linhas: {int((g3['linhas'] >= 50).sum())}")
print(f"  dessas, com 2 ou mais condições: "
      f"{int(((g3['linhas'] >= 50) & (g3['profundidade'] >= 2)).sum())}")


# ---------------------------------------------------------------------------
# D. Tradução: quantas linhas são precisas para enxergar um efeito
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("D. LINHAS x EFEITO QUE DÁ PRA ENXERGAR")
print("=" * 70)
print("Um subgrupo precisa se afastar do zero mais que esse valor para não")
print("ser ruído. Em posições, no alvo (desempenho vs. o próprio normal).\n")
for n in [13, 30, 50, 80, 120, 200, 400]:
    print(f"  {n:>4} linhas -> {2 * desvio / np.sqrt(n):.2f} posições")

res.to_csv(SAIDA / "levantamento_subgrupos_suporte.csv", index=False)
print(f"\nEnumeração completa salva em "
      f"{SAIDA}/levantamento_subgrupos_suporte.csv")
print("(só contagem de linhas - nenhum resultado do alvo foi olhado)")