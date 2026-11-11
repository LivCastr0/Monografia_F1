"""
Etapa 2 - H6: tempo perdido no pit lane x decisão de parar.

Critérios fixados antes de rodar: claude/pit-lane-parada-msi2.md (projeto no claude.ai).
Mesmo desenho das H1-H5 (sd_features_etapa1.py), com uma única feature de circuito:
pit_lane_time_loss_s em tercis entre os 26 circuitos. Hipótese definida depois das
análises anteriores da Etapa 2: exploratória declarada.

Entra: data/processed/base_decisao_parada.csv, features_completas_por_circuito.csv
Sai:   data/processed/sd_pit_lane/<H>_subgrupos.csv, <H>_acrescenta.csv, resumo.txt
Uso:   python src/sd_pit_lane_parada.py               (H6a, H6b e sensibilidade sem urbanos)
       python src/sd_pit_lane_parada.py --sintetico   (teste do procedimento)

Previstos
  H6a  pit lane alto E safety car ou VSC          -> maior taxa de parada
  H6b  pit lane alto E pneu 20–29 voltas          -> menor
       pit lane alto E segundo terço              -> menor (basta um dos dois)
Confirmada: subgrupo previsto firme na busca e que acrescenta no sentido previsto.
Informativo: Spearman entre o pit lane e a taxa de parada / idade do pneu na parada.

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
SAIDA = PROC / "sd_pit_lane"
FEAT = "pit_lane_time_loss_s"
F = f"f_{FEAT}"
URBANOS = {"Monaco", "Singapore"}
SEMENTE = 2026
CONDICOES = ["d_idade_pneu", "d_composto", "d_neutralizacao", "d_rival_frente", "d_rival_tras",
             "d_janela", "d_dist_frente", "d_paradas_feitas", "d_fase", "d_posicao"]
SC = "safety car ou VSC"
HIPOTESES = {
    "H6a": dict(base="parada", alvo="parou", suporte=200, condicoes=CONDICOES,
                previstos=[{F: "alta", "d_neutralizacao": SC}], sentido=+1, feature=FEAT,
                sentido_cont=-1),
    "H6b": dict(base="parada", alvo="parou", suporte=200, condicoes=CONDICOES,
                previstos=[{F: "alta", "d_idade_pneu": "20–29 voltas"},
                           {F: "alta", "d_fase": "segundo terço"}],
                sentido=-1, feature=FEAT, sentido_cont=-1),
}


def tercis():
    f = pd.read_csv(PROC / "features_completas_por_circuito.csv").set_index("circuito")[[FEAT]]
    t = f.rank(method="first").apply(lambda s: pd.cut(s, 3, labels=["baixa", "média", "alta"]))
    return f, t.astype(str)


def apoio(df, f, rng, n=10_000):
    """Informativo: Spearman nos circuitos entre o pit lane e duas medidas de parada."""
    x = f[FEAT]
    linhas = []
    for nome, serie, sentido in [
            ("taxa de parada por volta", df.groupby("circuito")["parou"].mean(), -1),
            ("idade do pneu na parada (mediana)",
             df[df["parou"] == 1].groupby("circuito")["idade_pneu"].median(), +1)]:
        xs = x.reindex(serie.index).to_numpy()
        y = serie.to_numpy()
        obs = spearmanr(xs, y).statistic
        nul = np.array([spearmanr(xs, rng.permutation(y)).statistic for _ in range(n)])
        p = (np.sum(sentido * nul >= sentido * obs) + 1) / (n + 1)
        linhas.append(f"   informativo: Spearman pit lane x {nome} = {obs:+.2f} "
                      f"(previsto {'+' if sentido > 0 else '-'}), p unilateral = {p:.4f}, "
                      f"{len(y)} circuitos")
    return linhas


def configura():
    sfe.FEATURES = [FEAT]          # a classe Busca usa a lista do módulo


def sintetico(f, t):
    """Alvo sintético com as taxas reais por idade do pneu e neutralização (sem efeito de
    pista) e, no segundo cenário, +5 p.p. sob SC/VSC nos circuitos de pit lane alto."""
    d = pd.read_csv(PROC / "base_decisao_parada.csv")
    rng = np.random.default_rng(5)
    base_p = d.groupby(["d_idade_pneu", "d_neutralizacao"])["parou"].transform("mean").to_numpy()
    alta = d["circuito"].map(t[FEAT]).eq("alta").to_numpy()
    sc = (d["d_neutralizacao"] == SC).to_numpy()
    for efeito in (0.0, 0.05):
        x = d.copy()
        x["parou"] = (rng.random(len(x)) < base_p + efeito * (alta & sc)).astype(int)
        sfe.roda_hipotese(f"sintetico_{int(efeito * 100)}", HIPOTESES["H6a"], f, t,
                          np.random.default_rng(SEMENTE), df=x, saida=SAIDA / "sintetico")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sintetico", action="store_true")
    args = ap.parse_args()
    configura()
    f, t = tercis()
    sfe.titulo("TERCIS DO PIT LANE (26 circuitos)")
    print(pd.concat([f.round(1), t.rename(columns={FEAT: "tercil"})], axis=1)
          .sort_values(FEAT).to_string())
    if args.sintetico:
        sfe.PERMUTACOES, sfe.PERM_ACR = 200, 500
        sintetico(f, t)
        return
    df = pd.read_csv(PROC / "base_decisao_parada.csv")
    resumo = []
    for nome, cfg in HIPOTESES.items():
        resumo += sfe.roda_hipotese(nome, cfg, f, t, np.random.default_rng(SEMENTE), df=df,
                                    saida=SAIDA)
    resumo += apoio(df, f, np.random.default_rng(SEMENTE))
    sfe.titulo("SENSIBILIDADE: SEM MÔNACO E SINGAPURA")
    sem = df[~df["circuito"].isin(URBANOS)]
    for nome, cfg in HIPOTESES.items():
        resumo += [f"[sem urbanos] {s}" for s in
                   sfe.roda_hipotese(f"{nome}_sem_urbanos", cfg, f, t,
                                     np.random.default_rng(SEMENTE), df=sem, saida=SAIDA)]
    sfe.titulo("RESUMO")
    print("\n".join(resumo))
    (SAIDA / "resumo.txt").write_text("\n".join(resumo) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
