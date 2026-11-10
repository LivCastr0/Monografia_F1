"""
Etapa 2 - Piloto 2 do bloco de setores: a análise por setor é válida?

O piloto 1 (`piloto_setores.py`, 4 corridas de 2024) mostrou que a MEDIDA de
degradação por setor funciona. Este piloto pergunta se o BLOCO de setores serve
para a monografia: se o setor carrega informação estável, que o circuito inteiro
perde, ligada às características que a Etapa 1 mede.

Entra: FastF1 (cache em data/cache; precisa de internet na primeira vez)
       data/processed/base_duelos.csv   (lista de corridas secas e perfil)
Sai:   data/processed/piloto_setores_validade/
         corridas/<ano>_<round>_{setores,stints}.csv  (uma vez por corrida; o script
                                                       pula as que já existem)
         criterios_medida.csv, desvios_setor.csv, resumo.txt
Uso:   python src/piloto_setores_validade.py              (coleta e avalia)
       python src/piloto_setores_validade.py --so-avaliar (só avalia o que já coletou)
       python src/piloto_setores_validade.py --sintetico  (teste dos critérios)

A medida, os filtros e o efeito da volta são os do piloto 1, versão 2
(importados de piloto_setores.py, sem mudança): inclinação Theil-Sen do tempo de
setor contra a vida do pneu, por stint x setor, em % do tempo típico do setor,
com o efeito da volta estimado na corrida por ajuste cruzado.

Corridas (sorteio fixado antes de rodar)
----------------------------------------
default_rng(2026) entre as corridas secas da base de duelos, sem 2022 (a API não
tem a classificação de 2022, e é dela que saem as features do setor) e sem os
circuitos do piloto 1 (Sakhir, Barcelona, Monza, Singapura). Dois circuitos de
cada perfil testado (alto downforce, stop-and-go, potência), entre os que têm
corrida seca nas duas eras; para cada circuito, uma corrida sorteada de 2018-21 e
uma de 2023-25. Total: 12 corridas, 6 circuitos, 18 pares de setor entre eras.
Urbano-técnico fica fora (só Mônaco sobra fora do piloto 1).

Critérios, fixados antes de rodar (mudar só antes de ver a saída)
-----------------------------------------------------------------
Desvio do setor = mediana da degradação dos stints no setor menos a média dos
três setores na mesma corrida (o nível da corrida sai, fica o perfil entre setores).

  M  medida: em pelo menos 10 das 12 corridas, cobertura >= 85% (C1 do piloto 1) e
     setor acima do acaso dentro do stint, p < 0,05 (C3 do piloto 1).
  V1 estabilidade: Spearman entre o desvio do setor na corrida de 2018-21 e na de
     2023-25 do mesmo circuito, nos pares de setor elegíveis, >= 0,50, com p < 0,05
     por permutação dos setores dentro de cada circuito. Par elegível: comprimento
     do setor muda no máximo 3% entre os dois anos (mudança de traçado ou de
     fronteira sai).
  V2 o setor acrescenta: na variação da degradação dos stint-setores, a parcela
     entre setores da mesma corrida >= 10% (no piloto 1, 15,7%).
  V3 ligação com a Etapa 1: Spearman entre o desvio do setor e o desvio de
     zonas_freio_km do setor (também centrado na corrida) >= 0,40, com p < 0,05 por
     permutação dos rótulos de setor dentro de cada corrida. A feature foi escolhida
     no piloto 1 (Spearman 0,71 em 12 setores já vistos); aqui é testada em
     corridas novas. pct_full_throttle e vel_media são só informativas.
  V4 caracterização: KMeans k = 4 (como na Etapa 1) sobre as features dos 36
     setores (vel_media, vel_minima, zonas_freio_km, pico_desac, densidade_curvas),
     padronizadas. Silhouette >= 0,20 e ARI médio em 200 reamostragens >= 0,70.
     Se passar, a tipologia de setores substitui a projeção nos centroides, que
     não funcionou no piloto 1.

Regra de decisão
  M, V1, V2 e V3 passam      -> bloco de setores válido; montar a base completa.
  M passa, V1 ou V2 falha    -> o perfil entre setores não é estável ou não
                                acrescenta; não vale o bloco.
  só V3 falha                -> o setor importa, mas não pelo que a Etapa 1 mede;
                                o bloco segue sem a ponte com a tipologia.
  V4 decide só como caracterizar os setores na base completa.

Teste sintético antes: perfis entre setores sem efeito, e perfis estáveis ligados a
zonas_freio_km. Associação, não causa.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))

BASE = Path("data/processed/base_duelos.csv")
SAIDA = Path("data/processed/piloto_setores_validade")
SEMENTE = 2026
PERFIS = ["alto downforce", "stop-and-go", "potência"]
USADOS_PILOTO1 = {"Sakhir", "Barcelona", "Monza", "Singapore"}
SEM_ANO = {2022}
COL = "modelo_cf"
FEATURES_V4 = ["vel_media", "vel_minima", "zonas_freio_km", "pico_desac", "densidade_curvas"]
FEATURE_V3 = "zonas_freio_km"
INFORMATIVAS = ["pct_full_throttle", "vel_media"]

M_CORRIDAS, M_COB, M_P = 10, 0.85, 0.05
V1_RHO, V1_COMP = 0.50, 0.03
V2_MIN = 0.10
V3_RHO = 0.40
V4_SIL, V4_ARI, V4_K, V4_BOOT = 0.20, 0.70, 4, 200
N_PERM = 5000


def titulo(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


# ---------------------------------------------------------------------------
def sorteia():
    b = pd.read_csv(BASE, usecols=["ano", "round", "circuito", "d_perfil"]) \
        .drop_duplicates(["ano", "round"])
    b = b[~b["ano"].isin(SEM_ANO) & ~b["circuito"].isin(USADOS_PILOTO1)]
    b["era"] = np.where(b["ano"] <= 2021, "2018-21", "2023-25")
    b = b.sort_values(["circuito", "ano", "round"]).reset_index(drop=True)
    rng = np.random.default_rng(SEMENTE)
    escolha = []
    for perfil in PERFIS:
        g = b[b["d_perfil"] == perfil]
        ok = g.groupby("circuito")["era"].nunique()
        circs = sorted(ok[ok == 2].index)
        for c in rng.permutation(circs)[:2]:
            for era in ("2018-21", "2023-25"):
                h = g[(g["circuito"] == c) & (g["era"] == era)]
                escolha.append(h.iloc[rng.integers(len(h))])
    return pd.DataFrame(escolha).reset_index(drop=True)


# ---------------------------------------------------------------------------
def coleta(corridas):
    """Roda a parte do piloto 1 que usa a FastF1, corrida a corrida, e salva."""
    import fastf1
    import piloto_setores as P
    Path(P.CACHE).mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(P.CACHE)
    pasta = SAIDA / "corridas"
    pasta.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEMENTE)
    crit = []
    for _, c in corridas.iterrows():
        ano, rnd, circ = int(c["ano"]), int(c["round"]), c["circuito"]
        nome = f"{ano}_{rnd:02d}"
        arq_set, arq_st, arq_cr = (pasta / f"{nome}_setores.csv", pasta / f"{nome}_stints.csv",
                                   pasta / f"{nome}_criterios.csv")
        if arq_set.exists() and arq_st.exists() and arq_cr.exists():
            print(f"{circ} {ano}: já coletada")
            continue
        print(f"\n=== {circ} {ano} (r{rnd}) ===")
        try:
            q = P.carregar(ano, rnd, "Q", laps=True, telemetry=True, weather=False)
            setores, c2, _ = P.setores_da_classificacao(q)
            r = P.carregar(ano, rnd, "R", laps=True, telemetry=False, weather=True)
        except Exception as e:
            print(f"    FALHA ao carregar: {type(e).__name__}: {e}")
            continue
        chuva = float(r.weather_data["Rainfall"].mean()) if r.weather_data is not None else 0.0
        laps = pd.DataFrame(r.laps)
        n_voltas = int(laps["LapNumber"].max())
        longo, c1, _ = P.voltas_limpas(laps)
        rep = P.reparticao_combustivel(setores, longo)
        longo = P.corrigir_volta(longo, rng)
        incl = P.inclinacoes(longo, rep, n_voltas, ano)
        stat, p, n_st = P.teste_sinal_setores(incl, P.VERSAO_PRINCIPAL, rng)
        print(f"    cobertura {c1:.1%}; setor explica {stat:.1%} dentro do stint (p = {p:.4f})")
        for df in (setores, incl):
            df.insert(0, "circuito", circ)
            df.insert(1, "ano", ano)
            df.insert(2, "round", rnd)
        setores.to_csv(arq_set, index=False)
        incl.to_csv(arq_st, index=False)
        pd.DataFrame([{"circuito": circ, "ano": ano, "round": rnd, "chuva": chuva,
                       "cobertura": c1, "fronteira": c2, "setor_explica": stat, "p": p,
                       "n_stints": n_st}]).to_csv(arq_cr, index=False)


def carrega_coletado():
    pasta = SAIDA / "corridas"
    le = lambda suf: pd.concat([pd.read_csv(a) for a in sorted(pasta.glob(f"*_{suf}.csv"))],
                               ignore_index=True)
    return le("setores"), le("stints"), le("criterios")


# ---------------------------------------------------------------------------
def desvios(setores, stints):
    m = stints.groupby(["circuito", "ano", "setor"])[COL].median().rename("deg").reset_index()
    m = m.merge(setores, on=["circuito", "ano", "setor"], how="left")
    corrida = m.groupby(["circuito", "ano"])
    m["dev"] = m["deg"] - corrida["deg"].transform("mean")
    for f in set(FEATURES_V4) | set(INFORMATIVAS):
        if f in m:
            m[f"dev_{f}"] = m[f] - corrida[f].transform("mean")
    m["era"] = np.where(m["ano"] <= 2021, "2018-21", "2023-25")
    return m


def perm_spearman(x, y, grupos, rng, n=N_PERM):
    """Spearman e p unilateral (sentido positivo) permutando y dentro dos grupos."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    obs = spearmanr(x, y).correlation
    idx = [np.flatnonzero(grupos == g) for g in np.unique(grupos)]
    maior = 0
    for _ in range(n):
        yp = y.copy()
        for i in idx:
            yp[i] = y[rng.permutation(i)]
        maior += spearmanr(x, yp).correlation >= obs
    return obs, (maior + 1) / (n + 1)


def v1(m, rng):
    a = m[m["era"] == "2018-21"].set_index(["circuito", "setor"])
    b = m[m["era"] == "2023-25"].set_index(["circuito", "setor"])
    j = a[["dev", "comprimento_m"]].join(b[["dev", "comprimento_m"]], lsuffix="_a", rsuffix="_b",
                                          how="inner")
    j["mudou"] = (j["comprimento_m_b"] / j["comprimento_m_a"] - 1).abs() > V1_COMP
    e = j[~j["mudou"]].reset_index()
    if len(e) < 6:
        return np.nan, np.nan, len(e), len(j)
    rho, p = perm_spearman(e["dev_a"], e["dev_b"], e["circuito"].to_numpy(), rng)
    return rho, p, len(e), len(j)


def v2(stints):
    s = stints.dropna(subset=[COL])
    y = s[COL]
    tot = ((y - y.mean()) ** 2).sum()
    corrida = s.groupby(["circuito", "ano"])[COL].transform("mean")
    setor = s.groupby(["circuito", "ano", "setor"])[COL].transform("mean")
    return (((corrida - y.mean()) ** 2).sum() / tot, ((setor - corrida) ** 2).sum() / tot,
            ((y - setor) ** 2).sum() / tot)


def v4(m, rng):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, adjusted_rand_score
    X = m[FEATURES_V4].dropna()
    if len(X) < V4_K * 3:
        return np.nan, np.nan, None
    Z = ((X - X.mean()) / X.std(ddof=0)).to_numpy()
    km = KMeans(V4_K, n_init=50, random_state=0).fit(Z)
    sil = silhouette_score(Z, km.labels_)
    aris = []
    for _ in range(V4_BOOT):
        i = rng.integers(0, len(Z), len(Z))
        kb = KMeans(V4_K, n_init=10, random_state=int(rng.integers(1e9))).fit(Z[i])
        aris.append(adjusted_rand_score(km.labels_, kb.predict(Z)))
    return sil, float(np.mean(aris)), pd.Series(km.labels_, index=X.index)


def avalia(setores, stints, crit, rng, n_esperado=12):
    titulo("M - A MEDIDA EM CADA CORRIDA (critérios do piloto 1)")
    print(crit[["circuito", "ano", "chuva", "cobertura", "setor_explica", "p", "n_stints"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    nm = int(((crit["cobertura"] >= M_COB) & (crit["p"] < M_P)).sum())
    ok_m = nm >= M_CORRIDAS
    if len(crit) < n_esperado:
        print(f"AVISO: {len(crit)} de {n_esperado} corridas coletadas")

    m = desvios(setores, stints)
    rho1, p1, n1, n1_tot = v1(m, rng)
    entre, dentro, resid = v2(stints)
    ok3 = m[["dev", f"dev_{FEATURE_V3}"]].dropna()
    rho3, p3 = perm_spearman(m.loc[ok3.index, f"dev_{FEATURE_V3}"], m.loc[ok3.index, "dev"],
                             (m.loc[ok3.index, "circuito"] + m.loc[ok3.index, "ano"].astype(str)).to_numpy(),
                             rng)
    sil, ari, rot = v4(m, rng)

    crits = [
        ("M  medida", f"{nm} de {len(crit)} corridas", f">= {M_CORRIDAS}", ok_m),
        ("V1 estabilidade entre eras", f"Spearman {rho1:+.2f}, p = {p1:.4f} ({n1} de {n1_tot} pares)",
         f">= {V1_RHO} e p < 0,05", bool(rho1 >= V1_RHO and p1 < 0.05)),
        ("V2 o setor acrescenta", f"entre setores {dentro:.1%} (entre corridas {entre:.1%}, resíduo {resid:.1%})",
         f">= {V2_MIN:.0%}", bool(dentro >= V2_MIN)),
        ("V3 ligação com a Etapa 1", f"Spearman {rho3:+.2f} com {FEATURE_V3}, p = {p3:.4f} ({len(ok3)} setores)",
         f">= {V3_RHO} e p < 0,05", bool(rho3 >= V3_RHO and p3 < 0.05)),
        ("V4 tipologia de setores", f"silhouette {sil:.2f}, ARI {ari:.2f}",
         f">= {V4_SIL} e >= {V4_ARI}", bool(sil >= V4_SIL and ari >= V4_ARI)),
    ]
    titulo("CRITÉRIOS")
    linhas = []
    for nome, valor, lim, ok in crits:
        s = f"{nome:28s} {'PASSA' if ok else 'FALHA':6s} {valor}   (limite {lim})"
        print(s)
        linhas.append(s)

    titulo("INFORMATIVO (não decide)")
    for f in INFORMATIVAS:
        c = f"dev_{f}"
        if c in m:
            d = m[["dev", c]].dropna()
            print(f"Spearman desvio x {f}: {spearmanr(d[c], d['dev']).correlation:+.2f} ({len(d)} setores)")

    ok = dict(zip(["M", "V1", "V2", "V3", "V4"], [c[-1] for c in crits]))
    if ok["M"] and ok["V1"] and ok["V2"] and ok["V3"]:
        dec = "bloco de setores válido; montar a base completa"
    elif ok["M"] and not (ok["V1"] and ok["V2"]):
        dec = "o perfil entre setores não é estável ou não acrescenta; não vale o bloco"
    elif ok["M"] and ok["V1"] and ok["V2"]:
        dec = "o setor importa, mas não pelo que a Etapa 1 mede; bloco sem a ponte com a tipologia"
    else:
        dec = "a medida não funcionou nestas corridas; rever antes de qualquer conclusão"
    dec += ("; tipologia de setores por KMeans" if ok["V4"] else
            "; caracterizar setores pelas features como descritores")
    print("\nDecisão pela regra fixada:", dec)

    m.drop(columns=[c for c in m.columns if c.startswith("Unnamed")], errors="ignore") \
        .to_csv(SAIDA / "desvios_setor.csv", index=False)
    crit.to_csv(SAIDA / "criterios_medida.csv", index=False)
    (SAIDA / "resumo.txt").write_text("\n".join(linhas + ["", "Decisão: " + dec]) + "\n", encoding="utf-8")
    return ok


# ---------------------------------------------------------------------------
def sintetico(rng, efeito):
    """6 circuitos x 2 anos x 3 setores; 40 stints por corrida. Com efeito, o desvio
    do setor = 0,03 x desvio padronizado de zonas_freio_km + ruído de ano pequeno."""
    sets, sts, crs = [], [], []
    for ci in range(6):
        base_f = rng.normal(0, 1, 3)                       # feature fixa do setor
        comp = rng.uniform(1200, 2500, 3)
        for ano in (2019, 2024):
            nivel = rng.normal(0.08, 0.02)
            f = base_f + rng.normal(0, 0.05, 3)
            dev = (0.03 * (f - f.mean()) if efeito else np.zeros(3)) + rng.normal(0, 0.005 if efeito else 0.0, 3)
            for s in range(3):
                sets.append({"circuito": f"C{ci}", "ano": ano, "round": 1, "setor": s + 1,
                             "comprimento_m": comp[s] * (1 + rng.normal(0, 0.005)),
                             "zonas_freio_km": 1.5 + 0.3 * f[s], "vel_media": 200 - 20 * f[s],
                             "vel_minima": 100 - 15 * f[s] + rng.normal(0, 5),
                             "pico_desac": -30 + rng.normal(0, 2), "densidade_curvas": 2.5 + 0.4 * f[s],
                             "pct_full_throttle": 70 - 8 * f[s]})
            for st in range(40):
                lvl = nivel + rng.normal(0, 0.03)
                for s in range(3):
                    sts.append({"circuito": f"C{ci}", "ano": ano, "round": 1, "setor": s + 1,
                                "piloto": f"P{st}", "stint": 1, COL: lvl + dev[s] + rng.normal(0, 0.04)})
            crs.append({"circuito": f"C{ci}", "ano": ano, "round": 1, "chuva": 0.0, "cobertura": 1.0,
                        "fronteira": 0.001, "setor_explica": np.nan, "p": 0.001, "n_stints": 40})
    return pd.DataFrame(sets), pd.DataFrame(sts), pd.DataFrame(crs)


def main():
    global SAIDA
    ap = argparse.ArgumentParser()
    ap.add_argument("--so-avaliar", action="store_true")
    ap.add_argument("--sintetico", action="store_true")
    args = ap.parse_args()
    rng = np.random.default_rng(SEMENTE)
    if args.sintetico:
        SAIDA = Path("data/processed/piloto_setores_validade/sintetico")
        SAIDA.mkdir(parents=True, exist_ok=True)
        for efeito in (False, True):
            titulo(f"SINTÉTICO: perfil entre setores ligado à feature = {efeito}")
            avalia(*sintetico(np.random.default_rng(3), efeito), rng)
        return
    corridas = sorteia()
    titulo("CORRIDAS DO PILOTO 2 (sorteio com default_rng(2026))")
    print(corridas.to_string(index=False))
    SAIDA.mkdir(parents=True, exist_ok=True)
    corridas.to_csv(SAIDA / "corridas_sorteadas.csv", index=False)
    if not args.so_avaliar:
        coleta(corridas)
    setores, stints, crit = carrega_coletado()
    avalia(setores, stints, crit, rng, n_esperado=len(corridas))


if __name__ == "__main__":
    main()
