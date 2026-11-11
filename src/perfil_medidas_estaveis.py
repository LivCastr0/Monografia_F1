"""
Etapa 2 - Quanto o perfil físico (Etapa 1) explica das medidas de corrida estáveis?

Definido DEPOIS do resultado da tipologia comportamental (tipologia_comportamental.py,
que parou no passo 4). Declarar assim. Desenho fixado antes de rodar:

  - Medidas: as 8 aprovadas no passo 2 da tipologia comportamental (estáveis entre
    metades das corridas): N1, N5, N6, N7, D2, D3, D8, P4.
  - Valores: assinatura com todos os dados (assinatura_todos.csv), 25 circuitos.
  - Estatística: eta² do perfil (variância entre perfis / variância total).
  - Referência: 10.000 partições aleatórias dos circuitos com os mesmos tamanhos.
    p = proporção de partições com eta² >= o real; correção de Holm para as 8.
  - Principal: sem Mônaco e Singapura (3 perfis, 10/8/5), porque os dois são extremos
    conhecidos e a pergunta é se a tipologia organiza os outros circuitos.
    Com eles (4 perfis): relatado, não decide.
  - Figura: valor padronizado de cada circuito por medida, marcado pelo perfil.

Entra: data/processed/tipologia_comportamental/assinatura_todos.csv,
       data/processed/clusters_circuitos.csv
Sai:   data/processed/perfil_medidas_estaveis/eta2.csv, valores.csv, figura.png/.pdf
Uso:   python src/perfil_medidas_estaveis.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

P = Path("data/processed")
SAI = P / "perfil_medidas_estaveis"
SEMENTE, PERM = 2026, 10_000
URBANOS = {"Monaco", "Singapore"}
NOMES_FIS = {0: "alto downforce", 1: "stop-and-go", 2: "potência", 3: "urbano-técnico"}
ORDEM = ["alto downforce", "stop-and-go", "potência", "urbano-técnico"]
CORES = dict(zip(ORDEM, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))
MARCAS = dict(zip(ORDEM, ["o", "s", "^", "D"]))
MEDIDAS = {
    "N1": "Taxa de ultrapassagem",
    "N5": "Taxa de parada por volta",
    "N6": "Idade do pneu na parada",
    "N7": "Degradação do pneu",
    "D2": "Efeito: pneu 10+ mais novo, defensor livre",
    "D3": "Efeito: 0,3 a 0,6 s, defensor livre",
    "D8": "Efeito: defensor em trem de DRS",
    "P4": "Efeito: 2.º terço e pneu 20–29 voltas",
}


def eta2(y, g):
    tot = ((y - y.mean()) ** 2).sum()
    entre = sum(((y[g == k].mean() - y.mean()) ** 2) * (g == k).sum() for k in np.unique(g))
    return entre / tot


def holm(p):
    p = np.asarray(p)
    o = np.argsort(p)
    adj = np.empty_like(p)
    acc = 0.0
    for i, j in enumerate(o):
        acc = max(acc, (len(p) - i) * p[j])
        adj[j] = min(1.0, acc)
    return adj


def testa(S, fis, rng):
    g = fis.reindex(S.index).to_numpy()
    linhas = []
    for m in MEDIDAS:
        y = S[m].to_numpy(dtype=float)
        obs = eta2(y, g)
        perm = np.array([eta2(y, rng.permutation(g)) for _ in range(PERM)])
        linhas.append({"medida": m, "descricao": MEDIDAS[m], "eta2": obs,
                       "eta2_mediana_acaso": np.median(perm),
                       "eta2_95_acaso": np.quantile(perm, 0.95),
                       "p": (np.sum(perm >= obs) + 1) / (PERM + 1)})
    t = pd.DataFrame(linhas)
    t["p_holm"] = holm(t["p"])
    return t


def figura(S, fis, saida):
    Z = (S - S.mean()) / S.std(ddof=0)
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=200)
    jit = np.random.default_rng(SEMENTE)
    for i, m in enumerate(MEDIDAS):
        for perfil in ORDEM:
            cs = [c for c in S.index if fis[c] == perfil]
            y = np.full(len(cs), len(MEDIDAS) - 1 - i) + jit.uniform(-0.18, 0.18, len(cs))
            ax.scatter(Z.loc[cs, m], y, s=34, marker=MARCAS[perfil], color=CORES[perfil],
                       edgecolor="white", linewidth=0.8, zorder=3,
                       label=perfil if i == 0 else None)
    ax.set_yticks(range(len(MEDIDAS)))
    ax.set_yticklabels(list(MEDIDAS.values())[::-1], fontsize=8.5, color="#3b3b38")
    ax.axvline(0, color="#c9c8c1", linewidth=1, zorder=1)
    for y in range(len(MEDIDAS)):
        ax.axhline(y, color="#ecebe6", linewidth=0.6, zorder=0)
    ax.set_xlabel("Valor do circuito, padronizado (desvios em relação à média dos 25)",
                  fontsize=8.5, color="#3b3b38")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#c9c8c1")
    ax.tick_params(axis="both", length=0, labelsize=8)
    ax.legend(loc="upper center", bbox_to_anchor=(0.42, 1.09), ncol=4, frameon=False,
              fontsize=8.5, handletextpad=0.3, columnspacing=1.2)
    fig.tight_layout()
    fig.savefig(saida / "figura.png")
    fig.savefig(saida / "figura.pdf")
    plt.close(fig)


def main():
    SAI.mkdir(parents=True, exist_ok=True)
    S = pd.read_csv(P / "tipologia_comportamental" / "assinatura_todos.csv", index_col=0)
    S = S[list(MEDIDAS)]
    c = pd.read_csv(P / "clusters_circuitos.csv").set_index("circuito")
    fis = c["cluster_kmeans_so_dinamicas_k4"].map(NOMES_FIS).reindex(S.index)
    rng = np.random.default_rng(SEMENTE)
    sem = S.loc[[x for x in S.index if x not in URBANOS]]
    t1 = testa(sem, fis, rng).assign(versao="sem urbanos (principal)")
    t2 = testa(S, fis, rng).assign(versao="com urbanos")
    t = pd.concat([t1, t2])
    t.to_csv(SAI / "eta2.csv", index=False)
    S.assign(perfil=fis).to_csv(SAI / "valores.csv")
    with pd.option_context("display.width", 200):
        print(t.round(3).to_string(index=False))
    print("\nMediana por perfil (todos os dados):")
    print(S.groupby(fis).median().reindex(ORDEM).round(3).T.to_string())
    figura(S, fis, SAI)
    print(f"\nFigura em {SAI / 'figura.png'}")


if __name__ == "__main__":
    main()
