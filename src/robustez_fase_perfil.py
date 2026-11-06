"""
Etapa 2 - Robustez do único efeito de perfil encontrado: fase da parada × perfil.

Entra: data/processed/base_parada.csv
Não salva nada e não usa internet. Precisa de verifica_perfil_parada.py na mesma pasta.

verifica_perfil_parada.py deu p = 0,0028 para a fase da parada mudar de efeito
entre os perfis. Esse teste foi definido depois de ver a busca, e o sinal pode
depender de poucos circuitos, em especial do urbano-técnico, que só tem Mônaco e
Singapura. Aqui ele é repetido de seis formas. O que se olha em cada uma é
"cedo - tarde": quanto parar cedo rende a mais que parar tarde, por duelo.

Regra fixada antes de rodar:
  FIRME se todas valerem:
    1. p < 0,05 sem o urbano-técnico;
    2. p < 0,05 ao retirar cada um dos 26 circuitos, um de cada vez;
    3. p < 0,05 com a medida alternativa (vizinhança 2, janela 5);
    4. p < 0,05 com a fase relativa à própria corrida;
    5. a ordem dos perfis em "cedo - tarde" é parecida nas duas eras
       (correlação de Spearman positiva).
  SÓ URBANO-TÉCNICO se falhar apenas o item 1: vale como pista sobre dois
  circuitos, não como diferença entre perfis.
  FRÁGIL em qualquer outro caso: registrar como não confirmado e não seguir testando.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from verifica_perfil_parada import MEDIDA, ORDEM_PERFIL, interacao

BASE = Path("data/processed/base_parada.csv")
PERMUTACOES = 2000
PERMUTACOES_CIRCUITO = 500
SEMENTE = 6
LIMITE = 0.05


def teste(df, y, faixa, rng, repeticoes=PERMUTACOES):
    """p da interação faixa × perfil, embaralhando perfis entre circuitos."""
    y = df[y].to_numpy()
    f = df[faixa].to_numpy()
    mapa = df.drop_duplicates("circuito").set_index("circuito")["d_perfil"]
    circuitos, perfis = mapa.index.to_numpy(), mapa.to_numpy()
    observado = interacao(y, f, df["d_perfil"].to_numpy())
    maiores = 0
    for _ in range(repeticoes):
        novo = dict(zip(circuitos, rng.permutation(perfis)))
        maiores += interacao(y, f, df["circuito"].map(novo).to_numpy()) >= observado
    return (maiores + 1) / (repeticoes + 1)


def cedo_menos_tarde(df, y="por_duelo", faixa="d_fase"):
    m = df.pivot_table(index="d_perfil", columns=faixa, values=y, aggfunc="mean")
    return (m["parada cedo"] - m["parada tarde"]).reindex(ORDEM_PERFIL)


def linha(rotulo, p, dif):
    valores = "  ".join(f"{v:+.3f}" if pd.notna(v) else "   -  " for v in dif)
    print(f"{rotulo:<26} p = {p:.4f}   {valores}")


def main():
    df = pd.read_csv(BASE)
    df[MEDIDA] = df["saldo"] / df["duelos"]
    df["corrida"] = df["ano"].astype(str) + "_" + df["round"].astype(str)
    # Fase dentro da própria corrida, em vez dos cortes da base inteira
    ordem = df.groupby("corrida")["volta_parada"].rank(pct=True)
    df["fase_relativa"] = pd.cut(ordem, [0, 1 / 3, 2 / 3, 1.0001],
                                 labels=["parada cedo", "parada no meio", "parada tarde"]
                                 ).astype(str)
    rng = np.random.default_rng(SEMENTE)

    print("Cedo - tarde por perfil, nesta ordem: " + ", ".join(ORDEM_PERFIL) + "\n")

    p_todos = teste(df, MEDIDA, "d_fase", rng)
    linha("Todos", p_todos, cedo_menos_tarde(df))

    sem = df[df["d_perfil"] != "urbano-técnico"]
    p_sem = teste(sem, MEDIDA, "d_fase", rng)
    linha("Sem urbano-técnico", p_sem, cedo_menos_tarde(sem))

    p_alt = teste(df, "saldo_alt", "d_fase", rng)
    linha("Medida alternativa", p_alt, cedo_menos_tarde(df, "saldo_alt"))

    p_rel = teste(df, MEDIDA, "fase_relativa", rng)
    linha("Fase relativa à corrida", p_rel, cedo_menos_tarde(df, faixa="fase_relativa"))

    antigos = df[df["ano"] < 2022]
    novos = df[df["ano"] >= 2022]
    dif_a, dif_n = cedo_menos_tarde(antigos), cedo_menos_tarde(novos)
    print()
    linha("Só 2018-21", teste(antigos, MEDIDA, "d_fase", rng), dif_a)
    linha("Só 2022+", teste(novos, MEDIDA, "d_fase", rng), dif_n)
    corr_eras = dif_a.corr(dif_n, method="spearman")
    print(f"Correlação entre as eras: {corr_eras:+.2f}")

    print("\nRetirando um circuito de cada vez:")
    ps = {}
    for c in sorted(df["circuito"].unique()):
        ps[c] = teste(df[df["circuito"] != c], MEDIDA, "d_fase", rng, PERMUTACOES_CIRCUITO)
    ps = pd.Series(ps).sort_values(ascending=False)
    print(ps.head(5).to_string(float_format=lambda x: f"{x:.4f}"))
    p_circ = ps.iloc[0]
    print(f"Pior caso: sem {ps.index[0]}, p = {p_circ:.4f}")

    resto = [("Retirando cada circuito", p_circ < LIMITE),
             ("Medida alternativa", p_alt < LIMITE),
             ("Fase relativa à corrida", p_rel < LIMITE),
             ("Mesma ordem nas duas eras", corr_eras > 0)]
    item1 = p_sem < LIMITE

    print("\n" + "=" * 72)
    print("REGRA")
    print("=" * 72)
    print(f"1. {'PASSOU' if item1 else 'NÃO PASSOU'}  Sem urbano-técnico")
    for i, (nome, ok) in enumerate(resto, 2):
        print(f"{i}. {'PASSOU' if ok else 'NÃO PASSOU'}  {nome}")
    print()
    if item1 and all(ok for _, ok in resto):
        print("Conclusão: FIRME. A fase da parada rende diferente entre os perfis.")
    elif not item1 and all(ok for _, ok in resto):
        print("Conclusão: SÓ URBANO-TÉCNICO. Pista sobre dois circuitos, não diferença "
              "entre perfis.")
    else:
        print("Conclusão: FRÁGIL. Registrar como não confirmado e não seguir testando.")
    print("\nNada foi salvo.")


if __name__ == "__main__":
    main()
