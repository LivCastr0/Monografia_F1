"""
Etapa 2 - H8: zonas de DRS x ultrapassagem.

Critérios fixados antes de rodar a busca: claude/drs-ultrapassagem-msi2.md (projeto no
claude.ai). Hipótese formulada DEPOIS de ver o efeito de "0,3 a 0,6 s com defensor livre"
por circuito (tipologia_comportamental.py): exploratória declarada.

Desenho da H2 (sd_features_etapa1.py) com uma única feature de circuito: a classe de zonas
de DRS (1 / 2 / 3+), de data/external/zonas_drs_por_circuito.csv, nas posições
baixa / média / alta usadas pelo script.

Previstos (maior taxa de ultrapassagem que a mesma distância nas outras pistas):
  3+ zonas E 0,3 a 0,6 s;  3+ zonas E 0,6 a 1,0 s   (basta um)
Informativo: Spearman entre o número típico de zonas e (i) a taxa de ultrapassagem e
(ii) o efeito D3 da tipologia comportamental.

Uso: python src/sd_drs_duelos.py [--sintetico]
Associação, não causa.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sd_features_etapa1 as sfe  # noqa: E402

PROC = Path("data/processed")
SAIDA = PROC / "sd_drs"
FEAT = "zonas_drs"
F = f"f_{FEAT}"
SEMENTE = 2026
CLASSE = {"1": "baixa", "2": "média", "3+": "alta"}
H8 = dict(base="duelos", alvo="passou", suporte=100,
          condicoes=["d_dif_idade", "d_dif_composto", "d_distancia", "d_trem", "d_largada",
                     "d_fase", "d_relargada", "d_posicao", "d_mesma_equipe"],
          previstos=[{F: "alta", "d_distancia": "0,3 a 0,6 s"},
                     {F: "alta", "d_distancia": "0,6 a 1,0 s"}],
          sentido=+1, feature=FEAT, sentido_cont=+1)


def classes():
    z = pd.read_csv(Path("data/external/zonas_drs_por_circuito.csv"), dtype={"zonas_drs_classe": str})
    z = z.set_index("circuito")
    f = z[["zonas_drs_tipico"]].rename(columns={"zonas_drs_tipico": FEAT})
    t = z[["zonas_drs_classe"]].rename(columns={"zonas_drs_classe": FEAT})
    t[FEAT] = t[FEAT].map(CLASSE)
    return f, t


def informativo(f, rng, n=10_000):
    du = pd.read_csv(PROC / "base_duelos.csv")
    linhas = []
    a = pd.read_csv(PROC / "tipologia_comportamental" / "assinatura_todos.csv", index_col=0)
    for nome, serie in [("taxa de ultrapassagem", du.groupby("circuito")["passou"].mean()),
                        ("efeito D3 (0,3 a 0,6 s com defensor livre)", a["D3"])]:
        x = f[FEAT].reindex(serie.index).to_numpy(dtype=float)
        y = serie.to_numpy(dtype=float)
        obs = spearmanr(x, y).statistic
        nul = np.array([spearmanr(x, rng.permutation(y)).statistic for _ in range(n)])
        p = (np.sum(nul >= obs) + 1) / (n + 1)
        linhas.append(f"   informativo: Spearman zonas de DRS x {nome} = {obs:+.2f}, "
                      f"p unilateral = {p:.4f}, {len(y)} circuitos")
    return linhas


def sintetico(f, t):
    d = pd.read_csv(PROC / "base_duelos.csv")
    rng = np.random.default_rng(5)
    base_p = d.groupby(["d_distancia", "d_trem"])["passou"].transform("mean").to_numpy()
    alta = d["circuito"].map(t[FEAT]).eq("alta").to_numpy()
    meio = (d["d_distancia"] == "0,3 a 0,6 s").to_numpy()
    for efeito in (0.0, 0.08):
        x = d.copy()
        x["passou"] = (rng.random(len(x)) < base_p + efeito * (alta & meio)).astype(int)
        sfe.roda_hipotese(f"sintetico_{int(efeito * 100)}", H8, f, t,
                          np.random.default_rng(SEMENTE), df=x, saida=SAIDA / "sintetico")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sintetico", action="store_true")
    args = ap.parse_args()
    sfe.FEATURES = [FEAT]
    f, t = classes()
    sfe.titulo("CLASSES DE ZONAS DE DRS (26 circuitos)")
    print(pd.concat([f, t.rename(columns={FEAT: "classe"})], axis=1).sort_values(FEAT).to_string())
    if args.sintetico:
        sfe.PERMUTACOES, sfe.PERM_ACR = 200, 500
        sintetico(f, t)
        return
    resumo = sfe.roda_hipotese("H8", H8, f, t, np.random.default_rng(SEMENTE),
                               df=sfe.carrega("duelos"), saida=SAIDA)
    resumo += informativo(f, np.random.default_rng(SEMENTE))
    sfe.titulo("RESUMO")
    print("\n".join(resumo))
    (SAIDA / "resumo.txt").write_text("\n".join(resumo) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
