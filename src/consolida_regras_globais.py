"""
Etapa 2 - Tarefa 2: conjunto pequeno de regras globais, com efeito, cobertura e robustez.

Entra: data/processed/base_duelos.csv                 + subgrupos_duelos_finais.csv (principal)
       data/processed/base_decisao_parada.csv         + subgrupos_decisao_parada_finais.csv
Sai:   data/processed/regras_globais_duelos.csv
       data/processed/regras_globais_parada.csv

Não usa internet.

Por que: os subgrupos finais (104 nos duelos, 63 na parada) ainda se sobrepõem muito.
A redundância anterior compara cada regra só com as versões mais simples dela, não com
as outras regras.

Seleção, fixada antes de rodar (cobertura ponderada, como no CN2-SD, Lavrač et al. 2004):
  - candidatas: os subgrupos finais SEM condição de perfil e SEM condição de era. O
    perfil não acrescentou em nenhum caso, e a era não é uma condição de corrida; as
    duas entram depois, na abrangência (tarefa 3) e na robustez.
  - cada volta começa com peso 1. A cada passo entra a candidata com maior |z|
    ponderado no sentido em questão; as voltas que ela cobre passam a pesar 1/(1+k),
    k = quantas regras escolhidas já as cobrem.
  - z ponderado = (taxa ponderada − taxa geral) × √(soma dos pesos) / desvio geral,
    a mesma medida da busca, com pesos.
  - para quando o melhor z ponderado fica abaixo do limiar da busca, ou em 8 regras
    acima da média e 3 abaixo.
  - a primeira regra escolhida é a de maior |z| da busca (o z ponderado coincide com o
    z original enquanto todos os pesos são 1).

Para cada regra escolhida:
  efeito     taxa, diferença para a taxa geral (p.p.) e razão, com intervalo de 95%
             por reamostragem de circuitos (1.000 sorteios dos 26 circuitos)
  cobertura  voltas, % da base, % dos casos positivos (ultrapassagens ou paradas) e
             % acumulado do conjunto
  robustez   taxa em cada era, menor e maior diferença retirando um circuito de cada
             vez, e em quantos circuitos (com 20+ voltas na regra) a taxa da regra
             passa da taxa do próprio circuito

Associação, não causa.
"""
from pathlib import Path

import numpy as np
import pandas as pd

P = Path("data/processed")
ANALISES = {
    "duelos": dict(base="base_duelos.csv", finais="subgrupos_duelos_finais.csv",
                   alvo="passou", n_col="voltas_de_duelo", limiar=8.61,
                   saida="regras_globais_duelos.csv"),
    "parada": dict(base="base_decisao_parada.csv", finais="subgrupos_decisao_parada_finais.csv",
                   alvo="parou", n_col="linhas", limiar=7.48,
                   saida="regras_globais_parada.csv"),
}
MAX_ACIMA, MAX_ABAIXO = 8, 3
SORTEIOS, SEMENTE = 1000, 2026
MIN_VOLTAS_CIRCUITO = 20


def mascara(df, descricao):
    m = np.ones(len(df), dtype=bool)
    for cond in descricao.split(" E "):
        c, v = cond.split("=", 1)
        m &= df["d_" + c].to_numpy() == v
    return m


def seleciona(y, mascaras, sentido, limiar, maximo, base, dp):
    """Cobertura ponderada num sentido (+1 acima da média, −1 abaixo)."""
    cobertas = np.zeros(len(y))
    escolhidas = []
    restantes = list(mascaras)
    while restantes and len(escolhidas) < maximo:
        w = 1.0 / (1.0 + cobertas)
        melhor, melhor_z = None, -np.inf
        for d in restantes:
            m = mascaras[d]
            sw = w[m].sum()
            z = sentido * ((w[m] * y[m]).sum() / sw - base) * np.sqrt(sw) / dp
            if z > melhor_z:
                melhor, melhor_z = d, z
        if melhor_z < limiar:
            break
        escolhidas.append((melhor, sentido * melhor_z))
        cobertas += mascaras[melhor]
        restantes.remove(melhor)
    return escolhidas


def efeito_por_circuito(y, m, circ, circuitos):
    """Somas por circuito, para reamostrar circuitos sem refazer as máscaras."""
    tot = pd.DataFrame({"c": circ, "y": y, "m": m, "ym": y * m})
    s = tot.groupby("c")[["y", "m", "ym"]].sum().reindex(circuitos)
    s["n"] = tot.groupby("c").size().reindex(circuitos)
    return s


def analisa(nome, cfg, rng):
    df = pd.read_csv(P / cfg["base"])
    fin = pd.read_csv(P / cfg["finais"])
    y = df[cfg["alvo"]].to_numpy(dtype=float)
    base, dp = y.mean(), y.std()
    circ = df["circuito"].to_numpy()
    circuitos = np.array(sorted(df["circuito"].unique()))
    era = df["d_era"].to_numpy()

    cand = fin[~fin["descricao"].str.contains("perfil=|era=", regex=True)]
    mascaras = {d: mascara(df, d) for d in cand["descricao"]}
    z_orig = dict(zip(fin["descricao"], fin["z"]))
    print(f"\n{'=' * 78}\n{nome.upper()}: {len(df)} voltas, taxa geral {100 * base:.2f}%; "
          f"finais {len(fin)}, candidatas sem perfil/era {len(cand)}\n{'=' * 78}")

    escolhidas = (seleciona(y, mascaras, +1, cfg["limiar"], MAX_ACIMA, base, dp)
                  + seleciona(y, mascaras, -1, cfg["limiar"], MAX_ABAIXO, base, dp))

    # reamostragem de circuitos: os mesmos sorteios para todas as regras
    idx = rng.integers(0, len(circuitos), size=(SORTEIOS, len(circuitos)))
    linhas, acumulado = [], np.zeros(len(y), dtype=bool)
    for ordem, (d, zp) in enumerate(escolhidas, 1):
        m = mascaras[d]
        n, taxa = m.sum(), y[m].mean()
        s = efeito_por_circuito(y, m, circ, circuitos)
        a = s.to_numpy(dtype=float)  # colunas y, m, ym, n
        A = a[idx]                   # sorteios × circuitos × 4
        t_regra = A[..., 2].sum(1) / A[..., 1].sum(1)
        t_geral = A[..., 0].sum(1) / A[..., 3].sum(1)
        dif_b, raz_b = t_regra - t_geral, t_regra / t_geral

        loo = []
        for k in range(len(circuitos)):
            r = np.delete(a, k, axis=0)
            loo.append(r[:, 2].sum() / r[:, 1].sum() - r[:, 0].sum() / r[:, 3].sum())
        sentido = np.sign(taxa - base)
        com = s[s["m"] >= MIN_VOLTAS_CIRCUITO]
        no_sentido = (np.sign(com["ym"] / com["m"] - com["y"] / com["n"]) == sentido).sum()

        if sentido > 0:
            acumulado |= m
        linhas.append({
            "ordem": ordem, "sentido": "acima" if sentido > 0 else "abaixo", "regra": d,
            "voltas": int(n), "pct_base": 100 * n / len(y),
            "pct_positivos": 100 * y[m].sum() / y.sum(),
            "pct_positivos_acumulado": 100 * y[acumulado].sum() / y.sum() if sentido > 0 else np.nan,
            "circuitos": int(pd.Series(circ[m]).nunique()),
            "taxa_pct": 100 * taxa,
            "dif_pp": 100 * (taxa - base),
            "dif_pp_ic_inf": 100 * np.nanpercentile(dif_b, 2.5),
            "dif_pp_ic_sup": 100 * np.nanpercentile(dif_b, 97.5),
            "razao": taxa / base,
            "razao_ic_inf": np.nanpercentile(raz_b, 2.5),
            "razao_ic_sup": np.nanpercentile(raz_b, 97.5),
            "taxa_2018_21_pct": 100 * y[m & (era == "2018-21")].mean(),
            "taxa_2022_mais_pct": 100 * y[m & (era == "2022+")].mean(),
            "dif_pp_sem_1_circuito_min": 100 * min(loo),
            "dif_pp_sem_1_circuito_max": 100 * max(loo),
            "circuitos_no_sentido": f"{no_sentido} de {len(com)}",
            "z_busca": z_orig[d], "z_ponderado": zp,
        })
    t = pd.DataFrame(linhas)
    t.to_csv(P / cfg["saida"], index=False)

    pd.set_option("display.width", 250)
    v = t.assign(
        efeito=[f"{r.taxa_pct:.1f}% ({r.dif_pp:+.1f} p.p. [{r.dif_pp_ic_inf:+.1f}; {r.dif_pp_ic_sup:+.1f}]; "
                f"×{r.razao:.2f})" for r in t.itertuples()],
        cobertura=[f"{r.voltas} ({r.pct_base:.1f}% base; {r.pct_positivos:.1f}% pos.)" for r in t.itertuples()],
        eras=[f"{r.taxa_2018_21_pct:.1f} / {r.taxa_2022_mais_pct:.1f}" for r in t.itertuples()],
        sem_1=[f"{r.dif_pp_sem_1_circuito_min:+.1f} a {r.dif_pp_sem_1_circuito_max:+.1f}" for r in t.itertuples()],
    )
    print(v[["ordem", "regra", "efeito", "cobertura", "eras", "sem_1", "circuitos_no_sentido",
             "z_busca", "z_ponderado"]].to_string(index=False, max_colwidth=70,
                                                  float_format=lambda x: f"{x:.1f}"))
    acima = t[t["sentido"] == "acima"]
    if len(acima):
        print(f"\nAs {len(acima)} regras acima da média cobrem "
              f"{acima['pct_positivos_acumulado'].iloc[-1]:.1f}% dos casos positivos.")
    print(f"Salvo em {P / cfg['saida']}")
    return t


def main():
    rng = np.random.default_rng(SEMENTE)
    for nome, cfg in ANALISES.items():
        analisa(nome, cfg, rng)


if __name__ == "__main__":
    main()
