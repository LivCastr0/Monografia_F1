"""
Etapa 2 - As paradas feitas nas situações que a SD identificou renderam posições?

Liga as regras da decisão de parar (quando as equipes param) ao resultado da
parada (saldo de posições do A1). Transforma regras descritivas em avaliação.

Entra: data/processed/base_decisao_parada.csv (condições na volta da parada)
       data/processed/base_parada.csv         (saldo de cada parada, A1)
       data/processed/regras_globais_parada.csv
Sai:   data/processed/eficacia_regras_parada.csv
Uso:   python src/eficacia_regras_parada.py [--sintetico]

Desenho (fixado antes de rodar)
-------------------------------
  - Unidade: parada em corrida seca, ligada pela volta de entrada no box
    (ano, round, piloto, volta = volta_parada).
  - Alvo: saldo (duelos de parada ganhos menos perdidos, medida do A1). Robustez:
    saldo_alt (vizinhança de 2 posições, janela de 5 voltas).
  - Grupos: as 8 regras de parada acima da média (situações em que se para mais).
    Para cada regra: saldo médio das paradas feitas na situação contra o das outras
    paradas. Regras com "0 paradas" comparam, na prática, primeiras paradas com as
    demais; isso é declarado na leitura.
  - Intervalo de 95% da diferença: reamostragem dos circuitos (1.000 sorteios).
  - Teste: o saldo é embaralhado entre as paradas da MESMA corrida (5.000 vezes);
    p corrigido = proporção de permutações em que o maior |t| entre as 8 regras
    passa do |t| observado da regra.
  - Firme: p corrigido < 0,05, mesmo sentido nas duas eras e no saldo_alt.
  - Teste sintético antes: sem efeito e com +0,5 posição plantado numa regra.
  - Descoberto ao montar, antes de ver qualquer saldo: o A1 só mede paradas em
    bandeira verde (exclui safety car, VSC e vermelha na volta de entrada ou na
    seguinte). A regra de safety car fica sem paradas e não é avaliável; regras com
    menos de 30 paradas também não.

Associação, não causa: quem para numa situação não é sorteado.
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

P = Path("data/processed")
PERM, BOOT, SEMENTE = 5000, 1000, 2026
warnings.filterwarnings("ignore", category=RuntimeWarning)


def mascara(df, regra):
    m = np.ones(len(df), dtype=bool)
    for cond in regra.split(" E "):
        c, v = cond.split("=", 1)
        m &= df["d_" + c].to_numpy() == v
    return m


def monta():
    d = pd.read_csv(P / "base_decisao_parada.csv")
    s = d[d["parou"] == 1].rename(columns={"volta": "volta_parada"})
    p = pd.read_csv(P / "base_parada.csv")[["ano", "round", "piloto", "volta_parada", "saldo",
                                             "saldo_alt", "duelos"]]
    return s.merge(p, on=["ano", "round", "piloto", "volta_parada"], how="inner")


def tstat(y, m):
    a, b = y[m], y[~m]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    return (a.mean() - b.mean()) / np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))


def analisa(df, regras, rng, alvo="saldo", quieto=False):
    df = df.reset_index(drop=True)
    y = df[alvo].to_numpy(dtype=float)
    masc = [mascara(df, r) for r in regras]
    masc = [m if 30 <= m.sum() <= len(m) - 30 else np.zeros(len(m), dtype=bool) for m in masc]
    obs = np.array([tstat(y, m) for m in masc])
    corrida = pd.factorize(df["ano"].astype(str) + "_" + df["round"].astype(str))[0]
    grupos = list(pd.Series(np.arange(len(df))).groupby(corrida).indices.values())
    maximos = np.empty(PERM)
    for i in range(PERM):
        v = y.copy()
        for g in grupos:
            v[g] = y[rng.permutation(g)]
        maximos[i] = np.nanmax(np.abs([tstat(v, m) for m in masc]))
    circ = df["circuito"].to_numpy()
    circs = np.unique(circ)
    idx_c = {c: np.flatnonzero(circ == c) for c in circs}
    sorteios = rng.integers(0, len(circs), (BOOT, len(circs)))
    era = df["d_era"].to_numpy()
    linhas = []
    for r, m, t in zip(regras, masc, obs):
        difs = []
        for s in sorteios:
            ii = np.concatenate([idx_c[circs[k]] for k in s])
            yy, mm = y[ii], m[ii]
            difs.append(yy[mm].mean() - yy[~mm].mean() if mm.any() and (~mm).any() else np.nan)
        eras = {e: y[m & (era == e)].mean() - y[~m & (era == e)].mean() for e in ("2018-21", "2022+")}
        linhas.append({"regra": r, "paradas": int(m.sum()), "saldo_na_situacao": y[m].mean(),
                       "saldo_outras": y[~m].mean(), "diferenca": y[m].mean() - y[~m].mean(),
                       "ic_inf": np.nanpercentile(difs, 2.5), "ic_sup": np.nanpercentile(difs, 97.5),
                       "t": t, "p_corrigido": (np.sum(maximos >= abs(t)) + 1) / (PERM + 1)
                       if np.isfinite(t) else np.nan,
                       "dif_2018_21": eras["2018-21"], "dif_2022_mais": eras["2022+"]})
    return pd.DataFrame(linhas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sintetico", action="store_true")
    args = ap.parse_args()
    regras = pd.read_csv(P / "regras_globais_parada.csv")
    regras = regras.loc[regras["sentido"] == "acima", "regra"].tolist()
    df = monta()
    print(f"Paradas ligadas: {len(df)} em {df.groupby(['ano', 'round']).ngroups} corridas, "
          f"{df['circuito'].nunique()} circuitos; saldo médio {df['saldo'].mean():+.3f}")
    if args.sintetico:
        global PERM
        PERM = 1000
        for efeito in (0.0, 0.5):
            x = df.copy()
            rng = np.random.default_rng(7)
            x["saldo"] = rng.normal(0, x["saldo"].std(), len(x)) + efeito * mascara(x, regras[1])
            t = analisa(x, regras, np.random.default_rng(SEMENTE))
            print(f"\nSINTÉTICO, +{efeito} em '{regras[1]}':")
            print(t[["regra", "paradas", "diferenca", "p_corrigido"]].to_string(index=False,
                  float_format=lambda v: f"{v:+.3f}", max_colwidth=60))
        return
    t = analisa(df, regras, np.random.default_rng(SEMENTE))
    alt = analisa(df.dropna(subset=["saldo_alt"]), regras, np.random.default_rng(SEMENTE), alvo="saldo_alt")
    t["dif_saldo_alt"] = alt["diferenca"].to_numpy()
    t["p_corrigido_alt"] = alt["p_corrigido"].to_numpy()
    s = np.sign(t["diferenca"])
    t["firme"] = ((t["p_corrigido"] < 0.05) & (np.sign(t["dif_2018_21"]) == s)
                  & (np.sign(t["dif_2022_mais"]) == s) & (np.sign(t["dif_saldo_alt"]) == s))
    pd.set_option("display.width", 250)
    print(t.to_string(index=False, float_format=lambda v: f"{v:+.3f}", max_colwidth=60))
    t.to_csv(P / "eficacia_regras_parada.csv", index=False)


if __name__ == "__main__":
    main()
