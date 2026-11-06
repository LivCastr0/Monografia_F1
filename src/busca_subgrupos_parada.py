"""
Etapa 2 - Busca de subgrupos no grão de parada (caminho A1).

Pergunta: em que tipo de pista e em que situação parar primeiro rende posição?

Entra: data/processed/base_parada.csv
Sai:   data/processed/subgrupos_parada.csv

Configuração igual à da rodada 2: StandardQFNumeric (a=0.5), profundidade 3,
suporte mínimo 50, os dois sentidos.

São dois testes, porque há dois tipos de descritor:

  1. O perfil sozinho. Ele vale para a corrida inteira, então o que se repete são
     os circuitos, não as paradas. Os perfis são embaralhados entre os circuitos.

  2. Os subgrupos com alguma condição da própria parada. O saldo é embaralhado
     entre as paradas de cada corrida e o maior valor de cada repetição define o
     limiar, descontando o número de subgrupos testados. Subgrupos só com condições
     da corrida (perfil, chuva, safety car) não são cobertos por esse teste e ficam
     marcados na tabela, sem veredito.

Verificações de cada subgrupo: mesmo sinal com a medida alternativa, mesmo sinal
nas duas eras (2018-21 e 2022+) e peso do maior circuito.

Descoberta de subgrupos descreve associação, não causa: quem para primeiro escolhe
parar, e essa escolha depende de como a corrida está.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pysubgroup as ps

ENTRADA = Path("data/processed/base_parada.csv")
SAIDA = Path("data/processed/subgrupos_parada.csv")

ALVO = "saldo"
SUPORTE_MINIMO = 50
PROFUNDIDADE = 3
PESO_TAMANHO = 0.5
PERMUTACOES = 1000
PERMUTACOES_PERFIL = 5000
SEMENTE = 4
TODOS = 100_000
QUANTOS_MOSTRAR = 15
DA_CORRIDA = {"d_perfil", "d_chuva", "d_safety_car"}


def teste_perfil(df, rng):
    """Os perfis têm saldo médio diferente? Embaralha perfis entre circuitos."""
    print("=" * 78)
    print("1. O SALDO POR PERFIL")
    print("=" * 78)
    t = df.groupby("d_perfil").agg(paradas=(ALVO, "size"), corridas=("corrida", "nunique"),
                                   circuitos=("circuito", "nunique"), saldo=(ALVO, "mean"),
                                   nao_zero=(ALVO, lambda s: (s != 0).mean()))
    print(t.to_string(float_format=lambda x: f"{x:+.3f}"))
    print(f"\nSaldo médio geral: {df[ALVO].mean():+.3f}")

    media = df[ALVO].mean()

    def separacao(rotulos):
        g = df[ALVO].groupby(rotulos).agg(["size", "mean"])
        return (g["size"] * (g["mean"] - media) ** 2).sum()

    mapa = df.drop_duplicates("circuito").set_index("circuito")["d_perfil"]
    circuitos, perfis = mapa.index.to_numpy(), mapa.to_numpy()
    observado = separacao(df["d_perfil"].to_numpy())
    maiores = 0
    for _ in range(PERMUTACOES_PERFIL):
        novo = dict(zip(circuitos, rng.permutation(perfis)))
        maiores += separacao(df["circuito"].map(novo).to_numpy()) >= observado
    p = (maiores + 1) / (PERMUTACOES_PERFIL + 1)
    print(f"O perfil muda o saldo? p = {p:.3f} (permutação entre {len(circuitos)} circuitos)\n")


def main():
    df = pd.read_csv(ENTRADA)
    df["corrida"] = df["ano"].astype(str) + "_" + df["round"].astype(str)
    df["era"] = np.where(df["ano"] >= 2022, "2022+", "2018-21")
    descritores = [c for c in df.columns if c.startswith("d_")]
    rng = np.random.default_rng(SEMENTE)

    print(f"Base: {len(df)} paradas, {df['corrida'].nunique()} corridas")
    print(f"Descritores ({len(descritores)}): {', '.join(descritores)}\n")
    teste_perfil(df, rng)

    print("=" * 78)
    print("2. A BUSCA")
    print("=" * 78)
    busca_df = df[descritores + [ALVO]].copy()
    tarefa = ps.SubgroupDiscoveryTask(
        busca_df, ps.NumericTarget(ALVO), ps.create_selectors(busca_df, ignore=[ALVO]),
        result_set_size=TODOS, depth=PROFUNDIDADE, qf=ps.StandardQFNumeric(a=PESO_TAMANHO),
        constraints=[ps.MinSupportConstraint(SUPORTE_MINIMO)],
    )
    achados = ps.Apriori().execute(tarefa).to_descriptions()
    cobre = np.array([np.asarray(sg.covers(busca_df)) for _, sg in achados])
    n = cobre.sum(axis=1)
    y = df[ALVO].to_numpy(dtype=float)
    so_corrida = np.array([all(s.attribute_name in DA_CORRIDA for s in sg.selectors)
                           for _, sg in achados])

    def medida(v):
        return cobre @ (v - v.mean()) / np.sqrt(n) / v.std(ddof=1)

    z = medida(y)
    testados = ~so_corrida
    linhas_por_corrida = list(df.groupby("corrida").indices.values())
    maximos = np.empty(PERMUTACOES)
    for i in range(PERMUTACOES):
        v = y.copy()
        for linhas in linhas_por_corrida:
            v[linhas] = y[rng.permutation(linhas)]
        maximos[i] = np.abs(medida(v)[testados]).max()
    limiar = np.quantile(maximos, 0.95)

    linhas = []
    for i, (qualidade, sg) in enumerate(achados):
        sub = df[cobre[i]]
        eras = sub.groupby("era")[ALVO].mean()
        linhas.append({
            "descricao": str(sg),
            "paradas": int(n[i]),
            "corridas": sub["corrida"].nunique(),
            "saldo": sub[ALVO].mean(),
            "z": z[i],
            "so_condicoes_de_corrida": bool(so_corrida[i]),
            "acima_do_limiar": bool(testados[i] and abs(z[i]) > limiar),
            "saldo_alt": sub["saldo_alt"].mean(),
            "saldo_2018_21": eras.get("2018-21", np.nan),
            "saldo_2022_mais": eras.get("2022+", np.nan),
            "maior_circuito_pct": sub["circuito"].value_counts(normalize=True).iloc[0],
            "qualidade": qualidade,
        })
    res = pd.DataFrame(linhas)
    res = res.reindex(res["z"].abs().sort_values(ascending=False).index)

    print(f"Subgrupos com {SUPORTE_MINIMO}+ paradas: {len(res)}")
    print(f"Testados: {int(testados.sum())}; só com condições de corrida: {int(so_corrida.sum())}")
    print(f"Limiar (|z|, 5%, {PERMUTACOES} permutações): {limiar:.2f}")
    passou = res[res["acima_do_limiar"]]
    print(f"Acima do limiar: {len(passou)}\n")

    print(res[testados[res.index]].head(QUANTOS_MOSTRAR)[
        ["descricao", "paradas", "saldo", "z", "acima_do_limiar"]
    ].to_string(index=False, float_format=lambda x: f"{x:+.2f}", max_colwidth=78))

    if len(passou):
        s = np.sign(passou["saldo"])
        firmes = passou[(np.sign(passou["saldo_alt"]) == s)
                        & (np.sign(passou["saldo_2018_21"]) == s)
                        & (np.sign(passou["saldo_2022_mais"]) == s)
                        & (passou["maior_circuito_pct"] <= 0.5)]
        print(f"\nDos {len(passou)} acima do limiar, passam nas verificações: {len(firmes)}")
        print(f"  com condição de perfil: "
              f"{int(firmes['descricao'].str.contains('d_perfil', regex=False).sum())}")
        print("\nOs que passam em tudo:")
        print(firmes[["descricao", "paradas", "saldo", "saldo_alt", "saldo_2018_21",
                      "saldo_2022_mais"]].head(QUANTOS_MOSTRAR).to_string(
            index=False, float_format=lambda x: f"{x:+.2f}", max_colwidth=70))

    res.to_csv(SAIDA, index=False)
    print(f"\nSalvo em {SAIDA}")


if __name__ == "__main__":
    main()
