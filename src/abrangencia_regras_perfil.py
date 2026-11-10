"""
Etapa 2 - Tarefa 3: abrangência das regras globais por perfil de circuito.

Pergunta: onde as regras globais (tarefa 2) aparecem e se o efeito delas muda entre os
perfis da Etapa 1. Não se busca regra própria de cada perfil.

Entra: data/processed/regras_globais_duelos.csv  + base_duelos.csv
       data/processed/regras_globais_parada.csv  + base_decisao_parada.csv
Sai:   data/processed/abrangencia/<analise>_por_perfil.csv   (regra × perfil)
       data/processed/abrangencia/<analise>_testes.csv       (regra: testes de partição)

Uso:   python src/abrangencia_regras_perfil.py
       python src/abrangencia_regras_perfil.py --sintetico   (teste do script)

Desenho fixado antes de rodar
-----------------------------
Para cada regra e cada perfil:
  - frequência: % das voltas do perfil que caem na regra;
  - efeito: taxa na regra contra a taxa do próprio perfil, em p.p. e como razão, com
    intervalo de 95% por reamostragem dos circuitos do perfil (1.000 sorteios);
  - retirando um circuito do perfil de cada vez: menor e maior diferença;
  - diferença em cada era (2018-21 e 2022+) dentro do perfil;
  - "abaixo do suporte" quando a regra tem menos voltas no perfil que o suporte da
    busca (100 nos duelos, 200 na parada).
  Urbano-técnico (2 circuitos): só descritivo, sem intervalo e fora dos testes, como na
  validação por cluster.

Testes de partição (três perfis testados; mesma lógica de valida_sd_por_cluster.py):
  - frequência: Q/gl da diferença de frequência entre os perfis;
  - efeito: Q/gl da diferença de efeito (taxa na regra − taxa do perfil), ponderada pelo
    inverso da variância; um perfil só entra se a regra tem o suporte nele.
  Referência: 2.000 partições aleatórias dos 26 circuitos com os tamanhos reais
  (2/5/8/11), o grupo de 2 fora. p por regra; p corrigido para todas as regras da
  análise comparando com o máximo entre regras em cada partição.
  O circuito é a unidade embaralhada, então a variação natural entre circuitos já está
  na referência: o teste pergunta se a tipologia separa mais que um agrupamento ao acaso.

Teste sintético (--sintetico): base com circuitos de taxas e efeitos próprios; sem
efeito de perfil, e com a regra mais frequente e mais forte no stop-and-go.

Associação, não causa.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

P = Path("data/processed")
PASTA = P / "abrangencia"
URBANO = "urbano-técnico"
ERAS = ("2018-21", "2022+")
SORTEIOS, PARTICOES, SEMENTE = 1000, 2000, 2026

ANALISES = {
    "duelos": dict(base="base_duelos.csv", regras="regras_globais_duelos.csv",
                   alvo="passou", suporte=100),
    "parada": dict(base="base_decisao_parada.csv", regras="regras_globais_parada.csv",
                   alvo="parou", suporte=200),
}


def mascara(df, descricao):
    m = np.ones(len(df), dtype=bool)
    for cond in descricao.split(" E "):
        c, v = cond.split("=", 1)
        m &= df["d_" + c].to_numpy() == v
    return m


def somas(df, m, alvo, circuitos):
    """Por circuito: voltas, positivos, voltas na regra, positivos na regra (e por era)."""
    y = df[alvo].to_numpy(dtype=float)
    t = pd.DataFrame({"c": df["circuito"].to_numpy(), "e": df["d_era"].to_numpy(),
                      "n": 1.0, "y": y, "m": m.astype(float), "ym": y * m})
    tot = t.groupby("c")[["n", "y", "m", "ym"]].sum().reindex(circuitos).fillna(0)
    era = {e: t[t["e"] == e].groupby("c")[["n", "y", "m", "ym"]].sum().reindex(circuitos).fillna(0)
           for e in ERAS}
    return tot.to_numpy(), {e: v.to_numpy() for e, v in era.items()}


def efeito(a):
    """a: linhas = circuitos, colunas n, y, m, ym (já somadas ou não)."""
    n, y, m, ym = a[..., 0].sum(-1), a[..., 1].sum(-1), a[..., 2].sum(-1), a[..., 3].sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return ym / m - y / n, (ym / m) / (y / n), m / n


def q_por_gl(grupos_somas, suporte):
    """Heterogeneidade da frequência e do efeito entre grupos. grupos_somas: G × 4."""
    n, y, m, ym = grupos_somas.T
    # frequência
    pi = m.sum() / n.sum()
    q_f = (n * (m / n - pi) ** 2).sum() / max(pi * (1 - pi), 1e-12) / (len(n) - 1)
    # efeito
    ok = m >= suporte
    if ok.sum() < 2:
        return q_f, np.nan
    p_g = y[ok] / n[ok]
    d = ym[ok] / m[ok] - p_g
    w = m[ok] / np.maximum(p_g * (1 - p_g), 1e-9)
    dbar = (w * d).sum() / w.sum()
    return q_f, (w * (d - dbar) ** 2).sum() / (ok.sum() - 1)


def analisa(nome, df, regras, alvo, suporte, rng):
    circuitos = np.array(sorted(df["circuito"].unique()))
    perfil_c = df.drop_duplicates("circuito").set_index("circuito")["d_perfil"] \
        .reindex(circuitos).to_numpy()
    perfis = pd.Series(perfil_c).value_counts()
    testados = sorted(g for g in perfis.index if g != URBANO)
    rotulos_base = np.concatenate([[g] * k for g, k in perfis.items()])
    parts = np.array([rng.permutation(rotulos_base) for _ in range(PARTICOES)])
    y_all = df[alvo].to_numpy(dtype=float)
    print(f"\n{'=' * 78}\n{nome.upper()}: {len(df)} voltas, taxa geral {100 * y_all.mean():.2f}%; "
          f"{len(regras)} regras\n{'=' * 78}")

    linhas, testes, q_parts = [], [], []
    for r in regras.itertuples():
        m = mascara(df, r.regra)
        tot, era = somas(df, m, alvo, circuitos)
        for g in list(testados) + ([URBANO] if URBANO in perfis else []):
            sel = perfil_c == g
            a = tot[sel]
            d, raz, freq = efeito(a)
            linha = {"ordem": r.ordem, "regra": r.regra, "sentido": r.sentido, "perfil": g,
                     "circuitos_perfil": int(sel.sum()),
                     "circuitos_com_regra": int((a[:, 2] > 0).sum()),
                     "voltas_perfil": int(a[:, 0].sum()), "voltas_regra": int(a[:, 2].sum()),
                     "frequencia_pct": 100 * freq,
                     "taxa_perfil_pct": 100 * a[:, 1].sum() / a[:, 0].sum(),
                     "taxa_regra_pct": 100 * a[:, 3].sum() / max(a[:, 2].sum(), 1),
                     "dif_pp": 100 * d, "razao": raz,
                     "abaixo_do_suporte": a[:, 2].sum() < suporte}
            if g != URBANO:
                idx = rng.integers(0, len(a), size=(SORTEIOS, len(a)))
                db, rb, _ = efeito(a[idx])
                linha.update(dif_pp_ic_inf=100 * np.nanpercentile(db, 2.5),
                             dif_pp_ic_sup=100 * np.nanpercentile(db, 97.5),
                             razao_ic_inf=np.nanpercentile(rb, 2.5),
                             razao_ic_sup=np.nanpercentile(rb, 97.5))
            loo = [efeito(np.delete(a, k, axis=0))[0] for k in range(len(a))] if len(a) > 1 else [np.nan]
            linha.update(dif_pp_sem_1_min=100 * np.nanmin(loo), dif_pp_sem_1_max=100 * np.nanmax(loo))
            for e in ERAS:
                ae = era[e][sel]
                linha[f"dif_pp_{e}"] = 100 * efeito(ae)[0] if ae[:, 2].sum() > 0 else np.nan
                linha[f"voltas_regra_{e}"] = int(ae[:, 2].sum())
            linhas.append(linha)

        def q(rot):
            gs = np.array([tot[rot == g].sum(0) for g in testados])
            return q_por_gl(gs, suporte)
        qf, qe = q(perfil_c)
        qp = np.array([q(rot) for rot in parts])          # PARTICOES × 2
        q_parts.append(qp)
        testes.append({"ordem": r.ordem, "regra": r.regra, "sentido": r.sentido,
                       "Q_frequencia": qf,
                       "p_frequencia": (np.sum(qp[:, 0] >= qf) + 1) / (PARTICOES + 1),
                       "Q_efeito": qe,
                       "p_efeito": (np.sum(qp[:, 1] >= qe) + 1) / (PARTICOES + 1)
                       if not np.isnan(qe) else np.nan})

    # correção para todas as regras: máximo entre regras em cada partição
    qp = np.stack(q_parts)                                  # regras × partições × 2
    max_f = np.nanmax(qp[:, :, 0], axis=0)
    max_e = np.nanmax(qp[:, :, 1], axis=0)
    for t in testes:
        t["p_frequencia_corrigido"] = (np.sum(max_f >= t["Q_frequencia"]) + 1) / (PARTICOES + 1)
        t["p_efeito_corrigido"] = ((np.sum(max_e >= t["Q_efeito"]) + 1) / (PARTICOES + 1)
                                   if not np.isnan(t["Q_efeito"]) else np.nan)
    por_perfil, testes = pd.DataFrame(linhas), pd.DataFrame(testes)
    return por_perfil, testes


def mostra(por_perfil, testes):
    pd.set_option("display.width", 250)
    for r in testes.itertuples():
        t = por_perfil[por_perfil["ordem"] == r.ordem]
        print(f"\n[{r.ordem}] {r.regra} ({r.sentido})")
        print(f"    frequência entre perfis: p = {r.p_frequencia:.3f} (corrigido {r.p_frequencia_corrigido:.3f})"
              f" | efeito entre perfis: p = {r.p_efeito:.3f} (corrigido {r.p_efeito_corrigido:.3f})")
        for x in t.itertuples():
            ic = (f"[{x.dif_pp_ic_inf:+.1f}; {x.dif_pp_ic_sup:+.1f}]"
                  if not np.isnan(getattr(x, "dif_pp_ic_inf", np.nan)) else "(descritivo)")
            print(f"    {x.perfil:15s} freq {x.frequencia_pct:5.1f}%  {x.voltas_regra:6d} v  "
                  f"taxa {x.taxa_regra_pct:5.1f}% vs {x.taxa_perfil_pct:5.1f}%  "
                  f"dif {x.dif_pp:+5.1f} {ic:16s} ×{x.razao:4.2f}  "
                  f"sem 1 circ {x.dif_pp_sem_1_min:+.1f} a {x.dif_pp_sem_1_max:+.1f}  "
                  f"eras {x.dif_pp_2018_21:+.1f} / {x.dif_pp_2022_mais:+.1f}"
                  f"{'  ABAIXO DO SUPORTE' if x.abaixo_do_suporte else ''}")


def renomeia_eras(t):
    return t.rename(columns={"dif_pp_2018-21": "dif_pp_2018_21", "dif_pp_2022+": "dif_pp_2022_mais",
                             "voltas_regra_2018-21": "voltas_regra_2018_21",
                             "voltas_regra_2022+": "voltas_regra_2022_mais"})


def sintetico(rng, com_efeito):
    base = pd.read_csv(P / "base_decisao_parada.csv", usecols=["circuito", "d_perfil"]) \
        .drop_duplicates("circuito")
    partes = []
    for r, (c, g) in enumerate(base.itertuples(index=False)):
        base_c, ef_c, freq_c = rng.normal(-3.0, 0.4), rng.normal(1.0, 0.3), rng.uniform(0.3, 0.5)
        if com_efeito and g == "stop-and-go":
            ef_c += 0.9
            freq_c += 0.2
        for ano in range(2018, 2025):
            n = 600
            a = (rng.random(n) < freq_c).astype(int)
            lp = base_c + ef_c * a + rng.normal(0, 0.3)
            partes.append(pd.DataFrame({"circuito": c, "d_perfil": g, "ano": ano, "round": r,
                                        "d_era": "2018-21" if ano <= 2021 else "2022+",
                                        "d_a": a.astype(str), "y": (rng.random(n) < 1 / (1 + np.exp(-lp))).astype(int)}))
    return pd.concat(partes, ignore_index=True)


def main():
    global PARTICOES
    ap = argparse.ArgumentParser()
    ap.add_argument("--sintetico", action="store_true")
    args = ap.parse_args()
    rng = np.random.default_rng(SEMENTE)
    if args.sintetico:
        PARTICOES = 1000
        regras = pd.DataFrame({"ordem": [1], "regra": ["a=1"], "sentido": ["acima"]})
        for com in (False, True):
            df = sintetico(rng, com)
            pp, tt = analisa(f"sintetico_efeito_{com}", df, regras, "y", 200, rng)
            mostra(renomeia_eras(pp), tt)
        return
    PASTA.mkdir(parents=True, exist_ok=True)
    for nome, cfg in ANALISES.items():
        df = pd.read_csv(P / cfg["base"])
        regras = pd.read_csv(P / cfg["regras"])
        pp, tt = analisa(nome, df, regras, cfg["alvo"], cfg["suporte"], rng)
        pp = renomeia_eras(pp)
        pp.to_csv(PASTA / f"{nome}_por_perfil.csv", index=False)
        tt.to_csv(PASTA / f"{nome}_testes.csv", index=False)
        mostra(pp, tt)
        print(f"\nSalvo em {PASTA}/{nome}_por_perfil.csv e {nome}_testes.csv")


if __name__ == "__main__":
    main()
