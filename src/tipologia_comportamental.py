"""
Etapa 2 - Tipologia comportamental dos circuitos.

Pergunta: se os circuitos forem agrupados pelo que acontece nas corridas (e não pelo
traçado), surge uma divisão estável? Se surgir, ela difere da tipologia física, o que
a explica e se ela organiza os padrões da SD melhor que a física.

Critérios fixados antes de rodar e aprovados pela Lívia em 2026-11-11:
claude/tipologia-comportamental-msi2.md (projeto no claude.ai).

Entra: data/processed/base_duelos.csv, base_decisao_parada.csv, base_stint.csv,
       base_corrida.csv, base_parada.csv, regras_globais_duelos.csv,
       regras_globais_parada.csv, clusters_circuitos.csv,
       features_completas_por_circuito.csv, pirelli_agregado_por_circuito.csv
Sai:   data/processed/tipologia_comportamental/   (ou .../sintetico_<cenario>/)
Uso:   python src/tipologia_comportamental.py               (base real)
       python src/tipologia_comportamental.py --sintetico   (os dois cenários sintéticos)

Passos
------
1. Assinatura de cada circuito em cada metade (A, B) e com todos os dados:
   8 níveis (N1-N8) e os efeitos das 19 regras globais, encolhidos para o efeito
   global (Bayes empírico). Regra só entra com mediana de 30+ linhas por
   circuito-metade.
   Metades: em cada circuito, corridas secas ordenadas por ano e round e
   distribuídas alternadamente em A e B, a primeira por sorteio (default_rng(2026)).
   Circuitos com 2+ corridas secas (Sochi fica fora).
2. Confiabilidade: Spearman A x B nos 25 circuitos; entra com rho >= 0,50 e
   p < 0,05 (unilateral, 10.000 permutações). Para se menos de 3 medidas passarem.
3. KMeans (n_init 50, semente 2026) nas medidas aprovadas padronizadas; k = 4
   (principal) e k de 2 a 6 pelo silhouette (secundário). Versões A, B e todos os
   dados, com e sem Mônaco e Singapura.
4. Replicação: ARI(A, B) sem os urbanos >= 0,40 e acima de 95% de 1.000
   permutações dos rótulos de B. Para se não replicar. Eras: informativo.
5. Comparação com a física: tabela cruzada, ARI com p por 10.000 permutações,
   circuitos que mudam de grupo.
6. SD explicativa: alvo "circuito do grupo X", descritores do circuito em tercis,
   profundidade 1 e 2, 3+ circuitos, WRAcc, p por 1.000 permutações dos grupos.
7. Tipologia da metade A x padrões da metade B: teste de partições de
   valida_sd_por_cluster.py (Q/gl médio, 2.000 partições com os mesmos tamanhos;
   grupos com menos de 4 circuitos ficam fora, como o urbano-técnico). Mesmo
   teste com a tipologia física, para comparar.

Teste sintético (antes da base real; ver o registro no projeto):
  sem estrutura -> para no passo 2 (nenhuma medida confiável).
  estrutura com efeitos de cerca de 70% da dispersão real entre circuitos (--forca 1)
  -> para no passo 2 (2 medidas); com 1,5x (--forca 1.5) ou mais -> replica no
  passo 4. O passo 7 não detectou a interação plantada (+13 p.p. numa regra) em
  nenhuma força, coerente com o poder medido na tarefa 4.
  A força dos efeitos plantados foi ajustada depois da primeira rodada sintética,
  antes de qualquer rodada na base real.

Associação, não causa.
"""
import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import valida_sd_por_cluster as vsc  # noqa: E402

P = Path("data/processed")
SEMENTE = 2026
URBANOS = {"Monaco", "Singapore"}
LIM_RHO, MIN_MEDIDAS, LIM_ARI, MIN_LINHAS_REGRA = 0.50, 3, 0.40, 30
PERM_RHO, PERM_ARI, PERM_FIS, PERM_SD, PARTICOES = 10_000, 1_000, 10_000, 1_000, 2_000
MIN_CIRC_GRUPO = 4
NOMES_FIS = {0: "alto downforce", 1: "stop-and-go", 2: "potência", 3: "urbano-técnico"}
TREM = "defensor em trem de DRS"
NIVEIS = {
    "N1": "taxa de ultrapassagem nos duelos",
    "N2": "% duelos com defensor em trem de DRS",
    "N3": "correlação largada x chegada (mediana)",
    "N4": "trocas por volta em bandeira verde (mediana)",
    "N5": "taxa de parada por volta",
    "N6": "idade do pneu na parada (mediana)",
    "N7": "degradação (mediana dos stints)",
    "N8": "saldo médio por parada",
}


def titulo(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------
def carregar():
    du = pd.read_csv(P / "base_duelos.csv")
    secas = du[["ano", "round", "circuito"]].drop_duplicates()
    pa = pd.read_csv(P / "base_decisao_parada.csv")
    st = pd.read_csv(P / "base_stint.csv", usecols=["ano", "round", "circuito", "deg_pct"])
    co = pd.read_csv(P / "base_corrida.csv", usecols=["ano", "round", "circuito", "rho_largada",
                                                       "trocas_por_volta"])
    pr = pd.read_csv(P / "base_parada.csv", usecols=["ano", "round", "circuito", "saldo"])
    k = ["ano", "round"]
    st, co, pr = (d.merge(secas[k]) for d in (st, co, pr))
    regras = {"duelos": pd.read_csv(P / "regras_globais_duelos.csv")["regra"].tolist(),
              "parada": pd.read_csv(P / "regras_globais_parada.csv")["regra"].tolist()}
    return dict(du=du, pa=pa, st=st, co=co, pr=pr, secas=secas), regras


def fisica():
    c = pd.read_csv(P / "clusters_circuitos.csv")
    return c.set_index("circuito")["cluster_kmeans_so_dinamicas_k4"].map(NOMES_FIS)


def metades(secas, rng):
    n = secas.groupby("circuito").size()
    validos = n[n >= 2].index
    s = secas[secas["circuito"].isin(validos)].sort_values(["circuito", "ano", "round"]).copy()
    partes = []
    for c, g in s.groupby("circuito", sort=True):
        inicio = int(rng.integers(2))
        g = g.copy()
        g["metade"] = np.where((np.arange(len(g)) + inicio) % 2 == 0, "A", "B")
        partes.append(g)
    s = pd.concat(partes)
    s["era"] = np.where(s["ano"] <= 2021, "2018-21", "2022+")
    return s


def recorta(dados, corridas):
    k = corridas[["ano", "round"]]
    return {n: d.merge(k) for n, d in dados.items() if n != "secas"}


def mascara(df, regra):
    m = np.ones(len(df), dtype=bool)
    for cond in regra.split(" E "):
        col, val = cond.split("=", 1)
        m &= df[f"d_{col}"].to_numpy() == val
    return m


# ---------------------------------------------------------------------------
# Passo 1: assinatura
# ---------------------------------------------------------------------------
def efeitos_regras(df, alvo, regras, prefixo, circuitos):
    y = df[alvo].to_numpy(dtype=float)
    circ = df["circuito"].to_numpy()
    taxa_c = pd.Series(y).groupby(circ).mean().reindex(circuitos)
    efe, cont = {}, {}
    for i, r in enumerate(regras, 1):
        m = mascara(df, r)
        n_r = pd.Series(m).groupby(circ).sum().reindex(circuitos).fillna(0)
        s_r = pd.Series(y * m).groupby(circ).sum().reindex(circuitos).fillna(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            t_r = s_r / n_r
        bruto = t_r - taxa_c
        glob = y[m].mean() - y.mean() if m.any() else 0.0
        p_ref = y[m].mean() if m.any() else 0.5
        v = np.where(n_r > 0, max(p_ref * (1 - p_ref), 1e-6) / n_r.clip(lower=1), np.inf)
        ok = n_r > 0
        tau2 = max(0.0, np.nanvar(bruto[ok]) - np.mean(v[ok])) if ok.sum() > 2 else 0.0
        w = np.where(np.isfinite(v), tau2 / (tau2 + v) if tau2 > 0 else 0.0, 0.0)
        efe[f"{prefixo}{i}"] = glob + w * (bruto.fillna(glob).to_numpy() - glob)
        cont[f"{prefixo}{i}"] = n_r.to_numpy()
    return pd.DataFrame(efe, index=circuitos), pd.DataFrame(cont, index=circuitos)


def assinatura(d, regras, circuitos):
    g = lambda df: df.groupby("circuito")                           # noqa: E731
    niv = pd.DataFrame(index=circuitos)
    niv["N1"] = g(d["du"])["passou"].mean()
    niv["N2"] = (d["du"]["d_trem"] == TREM).groupby(d["du"]["circuito"]).mean()
    niv["N3"] = g(d["co"])["rho_largada"].median()
    niv["N4"] = g(d["co"])["trocas_por_volta"].median()
    niv["N5"] = g(d["pa"])["parou"].mean()
    niv["N6"] = g(d["pa"][d["pa"]["parou"] == 1])["idade_pneu"].median()
    niv["N7"] = g(d["st"])["deg_pct"].median()
    niv["N8"] = g(d["pr"])["saldo"].mean()
    ed, cd = efeitos_regras(d["du"], "passou", regras["duelos"], "D", circuitos)
    ep, cp = efeitos_regras(d["pa"], "parou", regras["parada"], "P", circuitos)
    return pd.concat([niv, ed, ep], axis=1), pd.concat([cd, cp], axis=1)


# ---------------------------------------------------------------------------
# Passos 2-4
# ---------------------------------------------------------------------------
def rho_perm(a, b, rng, n=PERM_RHO):
    ok = a.notna() & b.notna()
    a, b = a[ok].to_numpy(), b[ok].to_numpy()
    if len(a) < 5:
        return np.nan, np.nan, len(a)
    if np.ptp(a) == 0 or np.ptp(b) == 0:                     # medida constante (encolhida)
        return np.nan, np.nan, len(a)
    r = spearmanr(a, b).statistic
    ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    perm = np.array([np.corrcoef(ra, rng.permutation(rb))[0, 1] for _ in range(n)])
    return r, (np.sum(perm >= r) + 1) / (n + 1), len(a)


def padroniza(S, cols):
    X = S[cols].astype(float)
    X = X.fillna(X.median())
    return ((X - X.mean()) / X.std(ddof=0).replace(0, 1)).to_numpy()


def agrupa(S, cols, k):
    return KMeans(n_clusters=k, n_init=50, random_state=SEMENTE).fit_predict(padroniza(S, cols))


def melhor_k(S, cols):
    X = padroniza(S, cols)
    sil = {}
    for k in range(2, 7):
        if k < len(S):
            lab = KMeans(n_clusters=k, n_init=50, random_state=SEMENTE).fit_predict(X)
            sil[k] = silhouette_score(X, lab)
    return max(sil, key=sil.get), sil


def ari_perm(a, b, rng, n):
    obs = adjusted_rand_score(a, b)
    perm = np.array([adjusted_rand_score(a, rng.permutation(b)) for _ in range(n)])
    return obs, float(np.quantile(perm, 0.95)), (np.sum(perm >= obs) + 1) / (n + 1)


# ---------------------------------------------------------------------------
# Passo 6: SD explicativa no grão circuito
# ---------------------------------------------------------------------------
def descritores_circuito(circuitos):
    f = pd.read_csv(P / "features_completas_por_circuito.csv").set_index("circuito")
    pi = pd.read_csv(P / "pirelli_agregado_por_circuito.csv").set_index("circuito")
    t = pd.DataFrame(index=circuitos)
    for c in ["vel_media", "vel_minima", "zonas_freio", "densidade_curvas",
              "pit_lane_time_loss_s", "altitude_m"]:
        t[c] = f[c]
    t["frenagem_forte"] = -f["pico_desac"]                   # mais alto = desaceleração maior
    t["abrasao"] = pi["#7_asphalt_abrasion_1_5"]
    t["estresse_pneu"] = pi["pirelli_tyre_stress_1_5"]
    t["carga_lateral"] = pi["pirelli_lateral_1_5"]
    t = t.reindex(circuitos)
    out = pd.DataFrame(index=circuitos)
    for c in t:
        r = t[c].rank(method="average")
        out[c] = pd.cut(r, bins=[0, len(r) / 3, 2 * len(r) / 3, len(r) + 1],
                        labels=["baixa", "média", "alta"]).astype(str)
    return out


def sd_circuitos(rotulos, D, rng):
    N = len(rotulos)
    sel = [(c, v) for c in D for v in ["baixa", "média", "alta"]]
    descs = [(s,) for s in sel] + [(a, b) for a, b in combinations(sel, 2) if a[0] != b[0]]
    M = np.array([np.all([D[c].to_numpy() == v for c, v in d], axis=0) for d in descs])
    n = M.sum(1)
    validas = n >= 3
    M, n, descs = M[validas], n[validas], [d for d, ok in zip(descs, validas) if ok]
    linhas = []
    for g in sorted(set(rotulos)):
        y = (rotulos == g).astype(float)
        wr = (n / N) * (M @ y / n - y.mean())
        maximos = np.array([((n / N) * (M @ (yp := rng.permutation(y)) / n - yp.mean())).max()
                            for _ in range(PERM_SD)])
        for j in np.argsort(-wr)[:5]:
            linhas.append({"grupo": g, "descricao": " E ".join(f"{c}={v}" for c, v in descs[j]),
                           "circuitos": int(n[j]), "do_grupo": int(M[j] @ y),
                           "tamanho_grupo": int(y.sum()), "wracc": wr[j],
                           "p": (np.sum(maximos >= wr[j]) + 1) / (PERM_SD + 1),
                           "lista": ", ".join(np.array(D.index)[M[j]])})
    return pd.DataFrame(linhas)


# ---------------------------------------------------------------------------
# Passo 7: teste de partições com uma tipologia qualquer
# ---------------------------------------------------------------------------
def particoes_tipologia(df, cfg, rotulo_circ, rng):
    alvo, suporte = cfg["alvo"], cfg["suporte"]
    df = df[df["circuito"].isin(rotulo_circ.index)]
    df = df.sort_values(["circuito", "ano", "round"], kind="stable").reset_index(drop=True)
    descs = vsc.descricoes(df, cfg["descritores"])
    cob = vsc.cobertura(df, descs)
    nomes, n_dc, s_dc, n_c, s_c = vsc.estatisticas_por_circuito(df, cob, alvo)
    rot = rotulo_circ.reindex(nomes).astype(str).to_numpy()
    tam = pd.Series(rot).value_counts()
    testados = sorted(g for g, t in tam.items() if t >= MIN_CIRC_GRUPO)
    q, _, _ = vsc.heterogeneidade(rot, testados, n_dc, s_dc, n_c, s_c, suporte)
    obs = np.nanmean(q)
    base = np.concatenate([[g] * t for g, t in tam.items()])
    med = np.empty(PARTICOES)
    for i in range(PARTICOES):
        qq, _, _ = vsc.heterogeneidade(rng.permutation(base), testados, n_dc, s_dc, n_c, s_c,
                                       suporte)
        med[i] = np.nanmean(qq)
    return dict(grupos_testados=len(testados), tamanhos="/".join(map(str, sorted(tam.values))),
                descricoes=int(np.isfinite(q).sum()), Q_medio_real=obs,
                Q_medio_mediana_aleatoria=float(np.median(med)),
                Q_medio_95_aleatoria=float(np.quantile(med, 0.95)),
                p=(np.sum(med >= obs) + 1) / (PARTICOES + 1))


# ---------------------------------------------------------------------------
def roda(dados, regras, saida, rng):
    saida.mkdir(parents=True, exist_ok=True)
    corr = metades(dados["secas"], rng)
    circuitos = sorted(corr["circuito"].unique())
    fis = fisica().reindex(circuitos)
    sem_urb = [c for c in circuitos if c not in URBANOS]
    corr.to_csv(saida / "metades.csv", index=False)
    print(f"{len(circuitos)} circuitos, {len(corr)} corridas secas "
          f"(A {int((corr.metade == 'A').sum())}, B {int((corr.metade == 'B').sum())})")

    # ---- passo 1
    titulo("PASSO 1: ASSINATURA DE CADA CIRCUITO")
    SA, CA = assinatura(recorta(dados, corr[corr.metade == "A"]), regras, circuitos)
    SB, CB = assinatura(recorta(dados, corr[corr.metade == "B"]), regras, circuitos)
    ST, _ = assinatura(recorta(dados, corr), regras, circuitos)
    e1 = corr[corr.era == "2018-21"]
    e2 = corr[corr.era == "2022+"]
    dois = sorted(set(e1.circuito) & set(e2.circuito))
    SE1, _ = assinatura(recorta(dados, e1), regras, circuitos)
    SE2, _ = assinatura(recorta(dados, e2), regras, circuitos)
    for nome, S in [("A", SA), ("B", SB), ("todos", ST), ("2018_21", SE1), ("2022_mais", SE2)]:
        S.to_csv(saida / f"assinatura_{nome}.csv")
    med_linhas = pd.concat([CA, CB]).median()
    regras_ok = [r for r in med_linhas.index if med_linhas[r] >= MIN_LINHAS_REGRA]
    textos = {f"D{i}": r for i, r in enumerate(regras["duelos"], 1)}
    textos.update({f"P{i}": r for i, r in enumerate(regras["parada"], 1)})
    print(f"Regras com mediana de {MIN_LINHAS_REGRA}+ linhas por circuito-metade: "
          f"{len(regras_ok)} de {len(med_linhas)}")
    for r in med_linhas.index:
        print(f"  {r} {'entra' if r in regras_ok else 'fora '} (mediana {med_linhas[r]:.0f}) "
              f"{textos[r]}")

    # ---- passo 2
    titulo("PASSO 2: CONFIABILIDADE (SPEARMAN ENTRE METADES)")
    candidatas = list(NIVEIS) + regras_ok
    conf = []
    for m in candidatas:
        r, p, n = rho_perm(SA[m], SB[m], rng)
        r_su, _, _ = rho_perm(SA.loc[sem_urb, m], SB.loc[sem_urb, m], rng, 1)
        r_er, p_er, n_er = rho_perm(SE1.loc[dois, m], SE2.loc[dois, m], rng, 2000)
        conf.append({"medida": m, "descricao": NIVEIS.get(m, textos.get(m)), "rho_metades": r,
                     "p_metades": p, "circuitos": n, "rho_sem_urbanos": r_su,
                     "rho_eras": r_er, "p_eras": p_er, "circuitos_eras": n_er,
                     "aprovada": bool(r >= LIM_RHO and p < 0.05)})
    conf = pd.DataFrame(conf)
    conf.to_csv(saida / "confiabilidade.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_colwidth", 45):
        print(conf.round(3).to_string(index=False))
    aprov = conf.loc[conf.aprovada, "medida"].tolist()
    print(f"\nAprovadas: {len(aprov)} ({', '.join(aprov) or 'nenhuma'})")
    resumo = {"circuitos": len(circuitos), "medidas_aprovadas": len(aprov),
              "aprovadas": ", ".join(aprov)}
    if len(aprov) < MIN_MEDIDAS:
        print(f"PARADA: menos de {MIN_MEDIDAS} medidas confiáveis. A análise termina aqui.")
        resumo["parou_em"] = "passo 2"
        pd.Series(resumo).to_csv(saida / "resumo.csv")
        return resumo

    # ---- passo 3
    titulo("PASSO 3: AGRUPAMENTO")
    tip = pd.DataFrame(index=circuitos)
    tip["fisica"] = fis
    for nome, S in [("A", SA), ("B", SB), ("todos", ST)]:
        tip[f"{nome}_k4"] = agrupa(S, aprov, 4)
        lab = pd.Series(agrupa(S.loc[sem_urb], aprov, 4), index=sem_urb)
        tip[f"{nome}_k4_sem_urbanos"] = lab.reindex(circuitos)
        k, sil = melhor_k(S, aprov)
        tip[f"{nome}_k_sil"] = agrupa(S, aprov, k)
        print(f"{nome}: silhouette k=2..6 " +
              ", ".join(f"{kk}:{v:.3f}" for kk, v in sil.items()) + f" -> k = {k}")
    tip.to_csv(saida / "tipologias.csv")

    # ---- passo 4
    titulo("PASSO 4: REPLICAÇÃO ENTRE METADES")
    rep = []
    for versao, cols in [("k4_sem_urbanos", "k4_sem_urbanos"), ("k4_com_urbanos", "k4"),
                         ("k_silhouette", "k_sil")]:
        a = tip[f"A_{cols}"].dropna().astype(int)
        b = tip[f"B_{cols}"].reindex(a.index).astype(int)
        obs, q95, p = ari_perm(a.to_numpy(), b.to_numpy(), rng, PERM_ARI)
        rep.append({"versao": versao, "circuitos": len(a), "ARI": obs, "ARI_95_acaso": q95,
                    "p": p})
    du_su = [c for c in dois if c not in URBANOS]
    l1 = agrupa(SE1.loc[du_su], aprov, 4)
    l2 = agrupa(SE2.loc[du_su], aprov, 4)
    obs, q95, p = ari_perm(l1, l2, rng, PERM_ARI)
    rep.append({"versao": "eras_k4_sem_urbanos (informativo)", "circuitos": len(du_su),
                "ARI": obs, "ARI_95_acaso": q95, "p": p})
    rep = pd.DataFrame(rep)
    rep.to_csv(saida / "replicacao.csv", index=False)
    print(rep.round(3).to_string(index=False))
    principal = rep.iloc[0]
    replica = bool(principal.ARI >= LIM_ARI and principal.ARI > principal.ARI_95_acaso)
    resumo.update(ARI_metades_sem_urbanos=round(principal.ARI, 3), replica=replica)
    print(f"\nReplica (ARI >= {LIM_ARI} e acima de 95% do acaso, sem urbanos): {replica}")
    if not replica:
        print("PARADA: a divisão não se repete entre metades. A análise termina aqui.")
        resumo["parou_em"] = "passo 4"
        pd.Series(resumo).to_csv(saida / "resumo.csv")
        return resumo

    # ---- passo 5
    titulo("PASSO 5: COMPARAÇÃO COM A TIPOLOGIA FÍSICA")
    comp = tip["todos_k4"]
    cruz = pd.crosstab(tip["fisica"], comp)
    print(cruz.to_string())
    obs, q95, p = ari_perm(fis.to_numpy(), comp.to_numpy(), rng, PERM_FIS)
    print(f"ARI física x comportamental: {obs:.3f} (95% do acaso {q95:.3f}; p = {p:.4f})")
    lin, col = linear_sum_assignment(-cruz.to_numpy())
    corresp = {cruz.columns[c]: cruz.index[r] for r, c in zip(lin, col)}
    Z = pd.DataFrame(padroniza(ST, aprov), index=circuitos, columns=aprov)
    mov = []
    for c in circuitos:
        alvo_fis = corresp.get(comp[c])
        if alvo_fis != fis[c]:
            dz = (Z.loc[c] - Z.loc[fis == fis[c]].mean()).abs().sort_values(ascending=False)
            mov.append({"circuito": c, "fisica": fis[c], "grupo_comportamental": int(comp[c]),
                        "grupo_parecido_com": alvo_fis,
                        "medidas_que_afastam": ", ".join(f"{m} ({Z.loc[c, m]:+.1f})"
                                                         for m in dz.index[:2])})
    mov = pd.DataFrame(mov)
    mov.to_csv(saida / "circuitos_que_mudam.csv", index=False)
    print(f"\nCircuitos que mudam de grupo: {len(mov)}")
    if len(mov):
        print(mov.to_string(index=False))
    perfis = ST[aprov].groupby(comp.to_numpy()).median()
    perfis.to_csv(saida / "perfil_grupos_comportamentais.csv")
    print("\nMediana das medidas por grupo comportamental:\n" + perfis.round(3).to_string())
    resumo.update(ARI_fisica=round(obs, 3), p_ARI_fisica=round(p, 4), circuitos_que_mudam=len(mov))

    # ---- passo 6
    titulo("PASSO 6: SD EXPLICATIVA (QUE CARACTERÍSTICAS DESCREVEM CADA GRUPO)")
    sdx = sd_circuitos(comp.to_numpy(), descritores_circuito(circuitos), rng)
    sdx.to_csv(saida / "sd_explicativa.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_colwidth", 60):
        print(sdx.round(3).to_string(index=False))

    # ---- passo 7
    titulo("PASSO 7: TIPOLOGIA DA METADE A x PADRÕES DA METADE B")
    dB = recorta(dados, corr[corr.metade == "B"])
    comp_A = tip["A_k4"].astype(int).astype(str)
    part = []
    for nome, chave in [("decisao_parada", "pa"), ("duelos", "du")]:
        cfg = vsc.BASES[nome]
        for tnome, rot in [("comportamental (metade A)", comp_A), ("física", fis)]:
            r = particoes_tipologia(dB[chave], cfg, rot, rng)
            r.update(analise=nome, tipologia=tnome)
            part.append(r)
            print(f"{nome:15s} {tnome:26s} grupos {r['tamanhos']:>12s}: Q/gl médio "
                  f"{r['Q_medio_real']:.2f} contra mediana {r['Q_medio_mediana_aleatoria']:.2f}"
                  f" (95% {r['Q_medio_95_aleatoria']:.2f}); p = {r['p']:.4f}")
    part = pd.DataFrame(part)
    part.to_csv(saida / "particoes_metade_B.csv", index=False)
    for _, r in part.iterrows():
        resumo[f"p_particoes_{r.analise}_{r.tipologia.split()[0]}"] = round(r.p, 4)
    resumo["parou_em"] = "completo"
    pd.Series(resumo).to_csv(saida / "resumo.csv")
    return resumo


# ---------------------------------------------------------------------------
# Sintético
# ---------------------------------------------------------------------------
def sintetico(dados, cenario, rng, forca=1.0):
    """Embaralha qual circuito cada corrida é (mantendo o número de corridas por circuito),
    o que apaga a identidade do circuito. Em 'estrutura', planta 3 grupos de circuitos."""
    secas = dados["secas"].copy()
    novo = secas.assign(circuito=rng.permutation(secas["circuito"].to_numpy()))
    mapa = novo.set_index(["ano", "round"])["circuito"]
    d = {}
    for n, df in dados.items():
        df = df.copy()
        df["circuito"] = mapa.reindex(pd.MultiIndex.from_frame(df[["ano", "round"]])).to_numpy()
        d[n] = df
    if cenario == "estrutura":
        circs = sorted(d["secas"]["circuito"].unique())
        grupo = dict(zip(circs, rng.permutation(np.arange(len(circs)) % 3)))
        g = lambda df: df["circuito"].map(grupo).to_numpy()   # noqa: E731
        # tamanhos da ordem das diferenças reais entre circuitos (desvio entre circuitos)
        du = d["du"]
        g_du = g(du)
        nao = (du["passou"] == 0).to_numpy()
        perto = (du["d_distancia"] == "até 0,3 s").to_numpy()
        sorteio = rng.random(len(du))
        du.loc[(g_du == 0) & nao & perto & (sorteio < min(1, 0.30 * forca)), "passou"] = 1   # regra D1
        du.loc[(g_du == 0) & nao & ~perto & (sorteio < 0.06 * forca), "passou"] = 1  # N1 (+cerca de 5 p.p.)
        co = d["co"]
        co.loc[g(co) == 0, "rho_largada"] -= 0.15 * forca                         # N3
        pa = d["pa"]
        alvo = (g(pa) == 1) & (pa["parou"] == 0)
        pa.loc[alvo & (rng.random(len(pa)) < 0.010 * forca), "parou"] = 1         # N5 (+1 p.p.)
        st = d["st"]
        st.loc[g(st) == 2, "deg_pct"] += 0.04 * forca                             # N7
        print("Plantado: grupo 0: até 0,3 s +30% das não-ultrapassagens, demais duelos +6%, "
              "rho -0,15; grupo 1: parada +1 p.p.; grupo 2: degradação +0,04")
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sintetico", action="store_true")
    ap.add_argument("--forca", type=float, default=1.0,
                    help="multiplica os efeitos plantados no cenário com estrutura")
    args = ap.parse_args()
    dados, regras = carregar()
    base = P / "tipologia_comportamental"
    if args.sintetico:
        for cenario in ("sem_estrutura", "estrutura"):
            titulo(f"SINTÉTICO: {cenario}")
            rng = np.random.default_rng(SEMENTE)
            d = sintetico(dados, cenario, rng, args.forca)
            pasta = base / (f"sintetico_{cenario}" + ("" if args.forca == 1 else f"_forca{args.forca:g}"))
            print(roda(d, regras, pasta, np.random.default_rng(SEMENTE)))
        return
    print(roda(dados, regras, base, np.random.default_rng(SEMENTE)))


if __name__ == "__main__":
    main()
