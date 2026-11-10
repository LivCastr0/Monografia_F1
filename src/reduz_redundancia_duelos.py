"""
Etapa 2 - Elimina subgrupos redundantes da busca de duelos.

Entra: data/processed/base_duelos.csv
       data/processed/subgrupos_duelos.csv            (ou _reduzida com --reduzida)
Sai:   data/processed/subgrupos_duelos_finais.csv     (ou _reduzida_finais)

Não usa internet e não refaz a busca: só organiza os subgrupos firmes que ela já
encontrou. O relatório da MSI II prevê eliminar regras redundantes.

Por que é preciso: um subgrupo como "distância até 0,3 s E equipes diferentes" é
quase o mesmo que "distância até 0,3 s", porque quase todo duelo é entre equipes
diferentes. A condição a mais não acrescenta nada, mas o subgrupo passa do limiar
só por herdar o efeito da distância.

Regra, fixada antes de ver a lista final. Um subgrupo firme fica se, comparado com
CADA versão mais simples dele (todas as combinações de parte das condições dele, e
também a base inteira):
  1. a taxa muda pelo menos MIN_GANHO (3 pontos percentuais) no mesmo sentido em que
     o subgrupo se afasta da média; e
  2. essa mudança passa de 2 erros-padrão, calculados com o tamanho do subgrupo.
As versões mais simples entram mesmo que não tenham sido firmes.

Também sai a parte 3 da busca: subgrupos com perfil que ficarem aqui continuam sem
o perfil acrescentar nada além do restante, se a parte 3 disse isso.
"""
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("data/processed/base_duelos.csv")
ALVO = "passou"
MIN_GANHO = 0.03
MIN_Z = 2.0


def condicoes(descricao):
    return [tuple(c.split("=", 1)) for c in descricao.split(" E ")]


def main():
    p = argparse.ArgumentParser(description="Elimina subgrupos redundantes dos duelos")
    p.add_argument("--reduzida", action="store_true")
    args = p.parse_args()
    sufixo = "_reduzida" if args.reduzida else ""
    entrada = Path(f"data/processed/subgrupos_duelos{sufixo}.csv")
    saida = Path(f"data/processed/subgrupos_duelos{sufixo}_finais.csv")

    df = pd.read_csv(BASE)
    y = df[ALVO].to_numpy(dtype=float)
    media = y.mean()
    colunas = {}

    def mascara(conds):
        m = np.ones(len(df), dtype=bool)
        for c, v in conds:
            if c not in colunas:
                colunas[c] = df["d_" + c].to_numpy()
            m &= colunas[c] == v
        return m

    res = pd.read_csv(entrada)
    firmes = res[res["firme"]].copy()

    linhas = []
    for _, r in firmes.iterrows():
        conds = condicoes(r["descricao"])
        m = mascara(conds)
        n, taxa = m.sum(), y[m].mean()
        sinal = np.sign(taxa - media)
        pior_ganho, pior_z, pior_geral = np.inf, np.inf, None
        for k in range(len(conds)):
            for sub in combinations(conds, k):
                base = y[mascara(sub)].mean() if sub else media
                ganho = sinal * (taxa - base)
                ep = np.sqrt(max(base * (1 - base), 1e-12) / n)
                z = ganho / ep
                if ganho < pior_ganho:
                    pior_ganho, pior_z = ganho, z
                    pior_geral = " E ".join(f"{c}={v}" for c, v in sub) or "base inteira"
        fica = pior_ganho >= MIN_GANHO and pior_z >= MIN_Z
        linhas.append({**r.to_dict(), "menor_ganho": pior_ganho, "z_do_menor_ganho": pior_z,
                       "comparado_com": pior_geral, "fica": fica})

    t = pd.DataFrame(linhas)
    finais = t[t["fica"]].sort_values("z", key=abs, ascending=False)
    print(f"Subgrupos firmes: {len(t)}   ficam: {len(finais)}   saem por redundância: "
          f"{len(t) - len(finais)}")
    print(f"Taxa geral de ultrapassagem: {media:.3f}\n")

    cols = ["descricao", "voltas_de_duelo", "circuitos", "taxa", "z", "menor_ganho",
            "comparado_com"]
    fmt = lambda x: f"{x:+.3f}"
    for nome, parte in [("ACIMA DA MÉDIA", finais[finais["taxa"] > media]),
                        ("ABAIXO DA MÉDIA", finais[finais["taxa"] < media])]:
        print("=" * 78 + f"\n{nome}: {len(parte)}\n" + "=" * 78)
        if len(parte):
            print(parte[cols].to_string(index=False, float_format=fmt, max_colwidth=62))
        print()
    com_perfil = finais[finais["tem_perfil"]]
    print(f"Finais com condição de perfil: {len(com_perfil)}")
    if "perfil_acrescenta" in finais and len(com_perfil):
        print(f"  em quantos o perfil acrescenta (parte 3): "
              f"{int((com_perfil['perfil_acrescenta'] == True).sum())}")
    finais.to_csv(saida, index=False)
    print(f"\nSalvo em {saida}")


if __name__ == "__main__":
    main()
