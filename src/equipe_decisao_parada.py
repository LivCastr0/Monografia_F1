"""
Etapa 2 - H7: comportamento de parada por equipe.

Critérios fixados antes de rodar: claude/equipe-parada-msi2.md (projeto no claude.ai).
Hipótese definida depois das análises anteriores da Etapa 2: exploratória declarada.

  H7a  top 5 E rival de trás parou: taxa de parada da ponta - das outras  (previsto > 0)
  H7b  SC/VSC, estratificado pela posição (Mantel-Haenszel): ponta - outras (previsto > 0)
  Ponta = 3 primeiras equipes do ano pela soma de pontos (base_piloto_corrida).
  Teste: 10.000 permutações sorteando 3 equipes "ponta" por ano (rótulo por equipe-ano);
  p unilateral; Holm para as duas. Confirmada: p Holm < 0,05, mesmo sentido nas duas eras
  e sem a equipe-ano com mais voltas na situação.
  Exploratório: teste de partições (valida_sd_por_cluster) com as 10 equipes como grupos
  e a equipe-ano como unidade.

Entra: data/processed/base_decisao_parada.csv, base_piloto_corrida.csv
Sai:   data/processed/equipe_parada/   (ou .../sintetico/)
Uso:   python src/equipe_decisao_parada.py [--sintetico]

Associação, não causa.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tipologia_comportamental as tc  # noqa: E402
import valida_sd_por_cluster as vsc  # noqa: E402

P = Path("data/processed")
SAI = P / "equipe_parada"
SEMENTE, PERM = 2026, 10_000
SC = "safety car ou VSC"
LINHAGEM = {
    "Toro Rosso": "Racing Bulls", "AlphaTauri": "Racing Bulls", "RB": "Racing Bulls",
    "Racing Bulls": "Racing Bulls",
    "Force India": "Aston Martin", "Racing Point": "Aston Martin", "Aston Martin": "Aston Martin",
    "Sauber": "Sauber", "Alfa Romeo": "Sauber", "Alfa Romeo Racing": "Sauber",
    "Kick Sauber": "Sauber",
    "Renault": "Alpine", "Alpine": "Alpine",
    "Haas F1 Team": "Haas", "Red Bull Racing": "Red Bull",
}
ESTRATOS = ["top 5", "6.º a 10.º", "11.º em diante"]


def carregar():
    pa = pd.read_csv(P / "base_decisao_parada.csv")
    pc = pd.read_csv(P / "base_piloto_corrida.csv", usecols=["ano", "round", "piloto", "equipe",
                                                             "pontos"])
    pc["equipe"] = pc["equipe"].map(lambda e: LINHAGEM.get(e, e))
    pa = pa.merge(pc[["ano", "round", "piloto", "equipe"]], on=["ano", "round", "piloto"],
                  how="left")
    falta = pa["equipe"].isna().sum()
    if falta:
        print(f"Aviso: {falta} voltas sem equipe; saem da análise")
        pa = pa.dropna(subset=["equipe"])
    pa["unidade"] = pa["equipe"] + "_" + pa["ano"].astype(str)
    pontos = pc.groupby(["ano", "equipe"])["pontos"].sum().reset_index()
    pontos["rank"] = pontos.groupby("ano")["pontos"].rank(ascending=False, method="first")
    ponta = set((pontos.loc[pontos["rank"] <= 3, "equipe"] + "_" +
                 pontos.loc[pontos["rank"] <= 3, "ano"].astype(str)))
    return pa, ponta, pontos


def estatisticas(pa):
    """Contagens por unidade nas duas situações, para recombinar permutações rápido."""
    a = pa[(pa["d_posicao"] == "top 5") & (pa["d_rival_tras"] == "rival de trás parou")]
    ta = a.groupby("unidade")["parou"].agg(["size", "sum"])
    b = pa[pa["d_neutralizacao"] == SC]
    tb = {s: b[b["d_posicao"] == s].groupby("unidade")["parou"].agg(["size", "sum"])
          for s in ESTRATOS}
    return ta, tb


def diferenca(tab, rotulo):
    p = rotulo.reindex(tab.index).fillna(False).to_numpy()
    n, s = tab["size"].to_numpy(), tab["sum"].to_numpy()
    n1, n0 = n[p].sum(), n[~p].sum()
    if n1 == 0 or n0 == 0:
        return np.nan, n1, n0
    return s[p].sum() / n1 - s[~p].sum() / n0, n1, n0


def mh(tabs, rotulo):
    num = den = 0.0
    for t in tabs.values():
        d, n1, n0 = diferenca(t, rotulo)
        if np.isfinite(d):
            w = n1 * n0 / (n1 + n0)
            num, den = num + w * d, den + w
    return num / den if den else np.nan


def holm(p):
    o = np.argsort(p)
    adj, acc = np.empty(len(p)), 0.0
    for i, j in enumerate(o):
        acc = max(acc, (len(p) - i) * p[j])
        adj[j] = min(1.0, acc)
    return adj


def confirmatorio(pa, ponta, rng):
    unidades = pa[["unidade", "ano"]].drop_duplicates().set_index("unidade")["ano"]
    real = pd.Series(unidades.index.isin(ponta), index=unidades.index)
    ta, tb = estatisticas(pa)
    obs = np.array([diferenca(ta, real)[0], mh(tb, real)])
    por_ano = {a: g.index.to_numpy() for a, g in unidades.groupby(unidades)}
    nul = np.empty((PERM, 2))
    for i in range(PERM):
        esc = np.concatenate([rng.choice(u, 3, replace=False) for u in por_ano.values()])
        r = pd.Series(unidades.index.isin(esc), index=unidades.index)
        nul[i] = diferenca(ta, r)[0], mh(tb, r)
    p = (np.sum(nul >= obs, axis=0) + 1) / (PERM + 1)
    ph = holm(p)

    linhas = []
    for k, (nome, situacao) in enumerate([("H7a", "top 5 E rival de trás parou"),
                                          ("H7b", "SC/VSC (estratificado pela posição)")]):
        sub = pa.copy()
        if nome == "H7a":
            sub = sub[(sub["d_posicao"] == "top 5") & (sub["d_rival_tras"] == "rival de trás parou")]
        else:
            sub = sub[sub["d_neutralizacao"] == SC]
        eras = {}
        for e in ("2018-21", "2022+"):
            se = pa[pa["d_era"] == e]
            ta_e, tb_e = estatisticas(se)
            eras[e] = diferenca(ta_e, real)[0] if nome == "H7a" else mh(tb_e, real)
        maior = sub["unidade"].value_counts().index[0]
        sem = pa[pa["unidade"] != maior]
        ta_s, tb_s = estatisticas(sem)
        sem_val = diferenca(ta_s, real)[0] if nome == "H7a" else mh(tb_s, real)
        e_ponta = sub["unidade"].isin(ponta)
        firme = bool(ph[k] < 0.05 and all(v > 0 for v in eras.values()) and sem_val > 0)
        linhas.append({
            "hipotese": nome, "situacao": situacao, "voltas": len(sub),
            "voltas_ponta": int(e_ponta.sum()), "voltas_outras": int((~e_ponta).sum()),
            "taxa_ponta": sub.loc[e_ponta, "parou"].mean(),
            "taxa_outras": sub.loc[~e_ponta, "parou"].mean(),
            "diferenca": obs[k], "diferenca_95_acaso": np.quantile(nul[:, k], 0.95),
            "p": p[k], "p_holm": ph[k], "dif_2018_21": eras["2018-21"],
            "dif_2022_mais": eras["2022+"], "sem_maior_unidade": sem_val,
            "maior_unidade": maior, "confirmada": firme})
    return pd.DataFrame(linhas)


def exploratorio(pa, rng):
    df = pa.copy()
    df["circuito"] = df["unidade"]           # a "unidade" do teste de partições
    rot = df.drop_duplicates("unidade").set_index("unidade")["equipe"]
    cfg = vsc.BASES["decisao_parada"]
    r = tc.particoes_tipologia(df, cfg, rot, rng)
    return pd.DataFrame([r])


def sintetico(pa, ponta, rng, efeito_a, efeito_b):
    x = pa.copy()
    base_p = x.groupby(["d_idade_pneu", "d_neutralizacao", "d_posicao", "d_rival_tras"])["parou"] \
        .transform("mean").to_numpy()
    e_ponta = x["unidade"].isin(ponta).to_numpy()
    sa = ((x["d_posicao"] == "top 5") & (x["d_rival_tras"] == "rival de trás parou")).to_numpy()
    sb = (x["d_neutralizacao"] == SC).to_numpy()
    p = base_p + e_ponta * (efeito_a * sa + efeito_b * sb)
    x["parou"] = (rng.random(len(x)) < np.clip(p, 0, 1)).astype(int)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sintetico", action="store_true")
    args = ap.parse_args()
    pa, ponta, pontos = carregar()
    print(f"{len(pa)} voltas, {pa['unidade'].nunique()} equipes-ano, {pa['equipe'].nunique()} equipes")
    top = pontos[pontos["rank"] <= 3].sort_values(["ano", "rank"])
    print("Equipes de ponta por ano: " + "; ".join(
        f"{a}: {', '.join(g['equipe'])}" for a, g in top.groupby("ano")))
    if args.sintetico:
        global PERM
        PERM = 2000
        saida = SAI / "sintetico"
        saida.mkdir(parents=True, exist_ok=True)
        for nome, ea, eb in [("sem_efeito", 0.0, 0.0), ("com_efeito", 0.06, 0.05)]:
            x = sintetico(pa, ponta, np.random.default_rng(5), ea, eb)
            t = confirmatorio(x, ponta, np.random.default_rng(SEMENTE))
            t.to_csv(saida / f"{nome}.csv", index=False)
            print(f"\nSINTÉTICO {nome}:")
            print(t[["hipotese", "taxa_ponta", "taxa_outras", "diferenca", "p", "p_holm",
                     "confirmada"]].round(4).to_string(index=False))
        return
    SAI.mkdir(parents=True, exist_ok=True)
    tc.titulo("CONFIRMATÓRIO")
    t = confirmatorio(pa, ponta, np.random.default_rng(SEMENTE))
    t.to_csv(SAI / "confirmatorio.csv", index=False)
    with pd.option_context("display.width", 220):
        print(t.round(4).T.to_string())
    tc.titulo("EXPLORATÓRIO: AS REGRAS VARIAM ENTRE EQUIPES?")
    e = exploratorio(pa, np.random.default_rng(SEMENTE))
    e.to_csv(SAI / "exploratorio_particoes.csv", index=False)
    print(e.round(4).T.to_string())


if __name__ == "__main__":
    main()
