"""
Etapa 2 - H10: chuva. Quando se troca de pneu e se a troca rende.

Critérios fixados antes de montar as bases: claude/safety-car-chuva-msi2.md (projeto no
claude.ai). Exploratória declarada.

Corridas: as que tiveram uso de pneu intermediário ou de chuva (sem as de disputa nula).
  H10a (entrar): piloto de pneu seco no fim de L-1, com chuva ou algum piloto já de pneu
        de chuva. Alvo: trocou para intermediário/chuva na volta L.
  H10b (sair): piloto de pneu de chuva no fim de L-1. Alvo: trocou para pneu seco em L.
  H10c (rende?): cada troca de H10a/H10b. Alvo: posições ganhas do fim de L-1 ao fim de
        L+3, reordenadas entre os pilotos presentes nos dois momentos.
Fora (H10a/b): volta 1, última volta, volta de saída do box, vermelha no ponto de decisão.

Uso: python src/sd_chuva.py [--sintetico]
Associação, não causa.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import avalia_piloto_a1 as A  # noqa: E402
import sd_eventos as S  # noqa: E402

P = Path("data/processed")
SAIDA = P / "sd_chuva"
SEMENTE = 2026
CHUVA = {"INTERMEDIATE", "WET"}
SECOS = {"HYPERSOFT", "ULTRASOFT", "SUPERSOFT", "SOFT", "MEDIUM", "HARD"}
NEUTRO = {"4", "6", "7"}
DESC_DECISAO = ["d_pelotao", "d_ritmo", "d_chuva", "d_idade_pneu", "d_posicao",
                "d_neutralizacao", "d_fase"]
DESC_TROCA = ["d_direcao", "d_pelotao", "d_ritmo", "d_posicao", "d_neutralizacao"]


def faixa(x, cortes, rotulos):
    for c, r in zip(cortes, rotulos):
        if x <= c:
            return r
    return rotulos[-1]


def tipo(c):
    return "chuva" if c in CHUVA else "seco" if c in SECOS else np.nan


def corrida(ano, rnd, circuito, total):
    v = A.le_voltas(A.VOLTAS / f"{ano}_{rnd:02d}_voltas.csv").dropna(subset=["posicao"]).copy()
    v["tipo"] = v["composto"].map(tipo)
    b = pd.read_csv(A.CORRIDAS / f"{ano}_{rnd:02d}_bandeiras.csv").sort_values("tempo_s")
    bt, bc = b["tempo_s"].to_numpy(), b["codigo"].astype(str).to_numpy()
    cl = pd.read_csv(A.CORRIDAS / f"{ano}_{rnd:02d}_clima.csv").sort_values("tempo_s")
    ct, cc = cl["tempo_s"].to_numpy(), cl["chuva"].astype(bool).to_numpy()

    def bandeira(t):
        if pd.isna(t):
            return "1"
        i = np.searchsorted(bt, t, side="right") - 1
        return bc[max(i, 0)]

    def chove(t):
        if pd.isna(t) or not len(ct):
            return False
        i = np.searchsorted(ct, t, side="right") - 1
        return bool(cc[max(i, 0)])

    ultima = int(v["volta"].max())
    T = v.pivot_table(index="volta", columns="piloto", values="tipo", aggfunc="first")
    POS = v.pivot_table(index="volta", columns="piloto", values="posicao")
    TEMPO = v.assign(t=np.where(v["entrada_box_s"].isna() & v["saida_box_s"].isna(),
                                v["tempo_volta_s"], np.nan)) \
        .pivot_table(index="volta", columns="piloto", values="t", dropna=False).reindex(T.index)
    FIM = v.pivot_table(index="volta", columns="piloto", values="fim_volta_s")
    idx = v.set_index(["piloto", "volta"])
    dec, trocas = [], []
    for (pil, L), r in idx.iterrows():
        if L < 2 or L >= ultima or pd.notna(r["saida_box_s"]):
            continue
        if (pil, L - 1) not in idx.index or L - 1 not in T.index:
            continue
        atual = T.loc[L - 1].get(pil)
        if pd.isna(atual):
            continue
        destino = "seco" if atual == "chuva" else "chuva"
        t_ant = idx.loc[(pil, L - 1), "fim_volta_s"]
        t_dec = r["entrada_box_s"] if pd.notna(r["entrada_box_s"]) else r["fim_volta_s"]
        if bandeira(t_dec) == "5":
            continue
        outros = T.loc[L - 1].drop(pil).dropna()
        frac = (outros == destino).mean() if len(outros) else 0.0
        if atual == "seco" and not (chove(t_ant) or (outros == "chuva").any()):
            continue
        prox = T.loc[L + 1].get(pil) if L + 1 in T.index else np.nan
        trocou = int(pd.notna(r["entrada_box_s"]) and pd.notna(prox) and prox == destino)
        tem = TEMPO.loc[L - 1]
        tp = T.loc[L - 1]
        med_d, med_a = tem[tp == destino].median(), tem[tp == atual].median()
        if pd.isna(med_d) or pd.isna(med_a):
            ritmo = "sem referência"
        else:
            rel = med_d / med_a - 1
            ritmo = "destino mais rápido" if rel < -0.01 else \
                "destino mais lento" if rel > 0.01 else "parecido"
        pos_ant = POS.loc[L - 1].get(pil)
        linha = {
            "ano": ano, "round": rnd, "circuito": circuito, "evento": f"{ano}_{rnd:02d}",
            "piloto": pil, "volta": L, "base": "entrar" if atual == "seco" else "sair",
            "trocou": trocou,
            "d_pelotao": "nenhum" if frac == 0 else faixa(frac, [1 / 3, 2 / 3],
                                                          ["até 1/3", "1/3 a 2/3", "mais de 2/3"]),
            "d_ritmo": ritmo,
            "d_chuva": "chovendo" if chove(t_ant) else "sem chuva",
            "d_idade_pneu": faixa(idx.loc[(pil, L - 1), "vida_pneu"], [9, 19],
                                  ["0–9 voltas", "10–19 voltas", "20+ voltas"])
            if pd.notna(idx.loc[(pil, L - 1), "vida_pneu"]) else np.nan,
            "d_posicao": faixa(pos_ant, [5, 10], ["top 5", "6.º a 10.º", "11.º em diante"]),
            "d_neutralizacao": "safety car ou VSC" if bandeira(t_dec) in NEUTRO else "bandeira verde",
            "d_fase": "primeiro terço" if L <= total / 3 else
            "segundo terço" if L <= 2 * total / 3 else "último terço",
            "d_era": "2022+" if ano >= 2022 else "2018-21"}
        dec.append(linha)
        if trocou and L + 3 <= ultima and L + 3 in POS.index:
            p0, p1 = POS.loc[L - 1].dropna(), POS.loc[L + 3].dropna()
            pres = p0.index.intersection(p1.index)
            if pil in pres:
                g = float(p0[pres].rank()[pil] - p1[pres].rank()[pil])
                trocas.append({**{k: linha[k] for k in
                                  ["ano", "round", "circuito", "evento", "piloto", "volta",
                                   "d_pelotao", "d_ritmo", "d_posicao", "d_neutralizacao",
                                   "d_era"]},
                               "d_direcao": "para chuva" if destino == "chuva" else "para seco",
                               "ganho": g})
    _ = FIM
    return pd.DataFrame(dec), pd.DataFrame(trocas)


def monta():
    base = pd.read_csv(P / "base_piloto_corrida.csv").drop_duplicates(["ano", "round"])
    base = base[~base["corrida_sem_disputa"]]
    dec, tro, usadas = [], [], []
    for _, r in base.iterrows():
        ano, rnd = int(r["ano"]), int(r["round"])
        f = A.VOLTAS / f"{ano}_{rnd:02d}_voltas.csv"
        if not f.exists():
            continue
        comp = pd.read_csv(f, usecols=["composto"])["composto"]
        if not comp.isin(CHUVA).any():
            continue
        d, t = corrida(ano, rnd, r["circuito"], r["voltas_corrida"])
        usadas.append(f"{ano} {r['circuito']}")
        dec.append(d)
        tro.append(t)
    dec, tro = pd.concat(dec, ignore_index=True), pd.concat(tro, ignore_index=True)
    dec.to_csv(P / "base_chuva_decisao.csv", index=False)
    tro.to_csv(P / "base_chuva_trocas.csv", index=False)
    print(f"Corridas ({len(usadas)}): " + "; ".join(usadas))
    for b in ("entrar", "sair"):
        x = dec[dec["base"] == b]
        print(f"{b}: {len(x)} voltas, {int(x['trocou'].sum())} trocas, "
              f"{x['evento'].nunique()} corridas, {x['circuito'].nunique()} circuitos")
    print(f"trocas com resultado: {len(tro)}")
    return dec, tro


def roda(dec, tro, rng_seed, saida, permutacoes=500, imprime=12):
    out = {}
    for nome, b in [("H10a_entrar", "entrar"), ("H10b_sair", "sair")]:
        x = dec[dec["base"] == b]
        res = S.busca(x, "trocou", DESC_DECISAO, "evento", 30, 0.03,
                      np.random.default_rng(rng_seed), permutacoes, min_era=10)
        S.resume(nome, res, imprime)
        out[nome] = res
    res = S.busca(tro, "ganho", DESC_TROCA, "evento", 20, 0.5, np.random.default_rng(rng_seed),
                  permutacoes, min_era=10)
    S.resume("H10c_rende", res, imprime)
    out["H10c_rende"] = res
    if saida is not None:
        saida.mkdir(parents=True, exist_ok=True)
        for nome, r in out.items():
            if not r.empty:
                r.drop(columns="chave").to_csv(saida / f"{nome}_subgrupos.csv", index=False)
    return out


def sintetico(dec, tro):
    rng = np.random.default_rng(5)
    # Nulo puro (acrescentado antes da rodada real): alvo embaralhado dentro da corrida.
    x = dec.copy()
    x["trocou"] = x.groupby(["base", "evento"])["trocou"].transform(
        lambda s: rng.permutation(s.to_numpy()))
    t = tro.copy()
    t["ganho"] = t.groupby("evento")["ganho"].transform(lambda s: rng.permutation(s.to_numpy()))
    print("\n=== SINTÉTICO nulo_puro (alvo embaralhado dentro da corrida) ===")
    roda(x, t, SEMENTE, None, permutacoes=200, imprime=4)
    for nome, efeito in [("sem_efeito", 0.0), ("com_efeito", 0.15)]:
        x = dec.copy()
        base_p = x.groupby(["base", "d_ritmo"])["trocou"].transform("mean").to_numpy()
        alvo = (x["d_pelotao"] == "mais de 2/3").to_numpy()
        x["trocou"] = (rng.random(len(x)) < np.clip(base_p + efeito * alvo, 0, 1)).astype(int)
        t = tro.copy()
        t["ganho"] = t.groupby("evento")["ganho"].transform(lambda s: rng.permutation(s.to_numpy()))
        print(f"\n=== SINTÉTICO {nome} ===")
        roda(x, t, SEMENTE, None, permutacoes=200, imprime=4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sintetico", action="store_true")
    args = ap.parse_args()
    dec, tro = monta()
    if args.sintetico:
        sintetico(dec, tro)
        return
    roda(dec, tro, SEMENTE, SAIDA, imprime=20)


if __name__ == "__main__":
    main()
