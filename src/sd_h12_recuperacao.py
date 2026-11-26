"""
Etapa 2 - H12: recuperação de quem largou fora de posição.

Critérios fixados antes de rodar: claude/h12-recuperacao-fora-posicao-msi2.md
(projeto no claude.ai). Exploratória declarada.

Parte A: o déficit de largada (grid atrás do ritmo esperado do carro) explica
         o ganho de posições? (descritiva; testa a explicação levantada na H11)
Parte B: SD entre quem largou 2+ posições atrás do esperado: em que condições
         recupera mais que o normal para o mesmo déficit?
Parte C: recuperação por circuito contra a taxa de ultrapassagem e as features
         da Etapa 1 (descritiva).

Uso:
  python src/sd_h12_recuperacao.py               base real
  python src/sd_h12_recuperacao.py --sintetico   os dois testes sintéticos
"""
import sys
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pandas as pd

BASE_SD = Path("data/processed/base_analise_sd.csv")
BASE_PC = Path("data/processed/base_piloto_corrida.csv")
BASE_RITMO = Path("data/processed/base_ritmo.csv")
ASSINATURA = Path("data/processed/tipologia_comportamental/assinatura_todos.csv")
FEATURES = Path("data/processed/features_completas_por_circuito.csv")
SAIDA = Path("data/processed/sd_h12_recuperacao")

CORRIDA = ["ano", "round"]
MIN_OUTRAS = 4
MIN_PILOTOS = 10
DEFICIT_MIN = 2
DESCRITORES = ["d_deficit", "d_grid", "d_paradas", "d_momento",
               "d_composto", "d_safety_car", "d_ultrapassagem"]
SUPORTE = 40
PROFUNDIDADE = 2
MIN_CIRCUITOS = 4
MAX_CIRCUITO = 0.40
PERMUTACOES = 1000
MIN_EFEITO = 0.3
MIN_LINHAS_CIRCUITO = 10
PLANTADO = {"d_ultrapassagem": "ultrapassagem alta", "d_deficit": "déficit 4-6,5"}
FEATS_ETAPA1 = ["vel_media", "vel_minima", "zonas_freio", "pico_desac", "densidade_curvas"]


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------
def ritmo_esperado(df):
    """Média do carro nas outras corridas secas da temporada (4+ outras)."""
    r = pd.read_csv(BASE_RITMO)
    out = []
    for (eq, ano), g in r.groupby(["equipe", "ano"]):
        for rnd in df.loc[(df.equipe == eq) & (df.ano == ano), "round"].unique():
            outras = g[g["round"] != rnd]["ritmo"]
            if len(outras) >= MIN_OUTRAS:
                out.append({"equipe": eq, "ano": ano, "round": rnd,
                            "ritmo_esp": outras.mean()})
    return pd.DataFrame(out)


def grid_esperado(df):
    """Verificação: grid médio do carro nas outras corridas da temporada."""
    pc = pd.read_csv(BASE_PC)
    pc = pc[~pc["nao_largou"]].copy()
    pc["g"] = pc["grid"].replace(0, 20)
    carro = pc.groupby(["equipe", "ano", "round"])["g"].mean().reset_index()
    out = []
    for (eq, ano), g in carro.groupby(["equipe", "ano"]):
        for rnd in df.loc[(df.equipe == eq) & (df.ano == ano), "round"].unique():
            outras = g[g["round"] != rnd]["g"]
            if len(outras) >= MIN_OUTRAS:
                out.append({"equipe": eq, "ano": ano, "round": rnd,
                            "grid_esp": outras.mean()})
    return pd.DataFrame(out)


def faixa_ultrapassagem():
    a = pd.read_csv(ASSINATURA).rename(columns={"Unnamed: 0": "circuito"})
    a = a[["circuito", "N1"]].dropna()
    t = pd.qcut(a["N1"].rank(method="first"), 3,
                labels=["ultrapassagem baixa", "ultrapassagem média", "ultrapassagem alta"])
    return a.assign(d_ultrapassagem=t.astype(str))


def reordena(df, chave_esp, nome):
    df = df.copy()
    df["grid_ord"] = df["grid"].replace(0, 99)
    df["grid_r"] = df.groupby(CORRIDA)["grid_ord"].rank(method="first")
    df["cheg_r"] = df.groupby(CORRIDA)["posicao_final"].rank(method="first")
    df[nome] = df["grid_r"] - df.groupby(CORRIDA)[chave_esp].rank(method="average")
    df["ganho"] = df["grid_r"] - df["cheg_r"]
    return df


def monta_base():
    sd = pd.read_csv(BASE_SD)
    sd = sd[sd["d_chuva"] == "seco"].copy()
    n0 = sd.groupby(CORRIDA).ngroups
    df = sd.merge(ritmo_esperado(sd), on=["equipe", "ano", "round"], how="inner")
    n_pil = df.groupby(CORRIDA)["piloto"].transform("size")
    saiu = df[n_pil < MIN_PILOTOS].groupby(CORRIDA).ngroups
    df = df[n_pil >= MIN_PILOTOS]
    df = reordena(df, "ritmo_esp", "deficit")
    df = df.merge(faixa_ultrapassagem()[["circuito", "N1", "d_ultrapassagem"]],
                  on="circuito", how="left")
    df["era"] = np.where(df["ano"] <= 2021, "2018-21", "2022+")
    print(f"Base: {len(sd)} linhas secas em {n0} corridas -> {len(df)} linhas em "
          f"{df.groupby(CORRIDA).ngroups} corridas ({saiu} corridas com menos de "
          f"{MIN_PILOTOS} pilotos medidos saíram); {df.circuito.nunique()} circuitos")
    return df.reset_index(drop=True)


def faixa_deficit(d):
    return pd.cut(d, [-np.inf, -2, 1.5, 3.5, 6.5, np.inf],
                  labels=["≤ -2", "-1,5 a +1,5", "2 a 3,5", "4 a 6,5", "7+"])


# ---------------------------------------------------------------------------
# Parte A
# ---------------------------------------------------------------------------
def spearman_perm(df, x, y, rng, n=2000):
    from scipy.stats import spearmanr
    rho = spearmanr(df[x], df[y]).statistic
    corridas = list(df.groupby(CORRIDA).indices.values())
    yy = df[y].to_numpy()
    xr = df[x].rank().to_numpy()
    cont = 0
    for _ in range(n):
        e = yy.copy()
        for l in corridas:
            e[l] = yy[rng.permutation(l)]
        if abs(np.corrcoef(xr, pd.Series(e).rank())[0, 1]) >= abs(rho):
            cont += 1
    return rho, (cont + 1) / (n + 1)


def parte_a(df):
    rng = np.random.default_rng(2026)
    print(f"\n{'=' * 78}\nPARTE A - o déficit explica o ganho?\n{'=' * 78}")
    rho, p = spearman_perm(df, "deficit", "ganho", rng)
    print(f"Spearman déficit (ritmo) x ganho: {rho:+.3f} (p = {p:.4f})")
    df = df.assign(faixa=faixa_deficit(df["deficit"]))
    t = df.groupby("faixa", observed=True)["ganho"].agg(["size", "mean"])
    print("\nGanho médio por faixa de déficit:\n" + t.to_string(float_format="{:+.2f}".format))
    no_lugar = df["deficit"].abs() <= 1.5
    t2 = pd.DataFrame({
        "toda a base": df.groupby("d_grid")["ganho"].mean(),
        "n": df.groupby("d_grid").size(),
        "largou no lugar": df[no_lugar].groupby("d_grid")["ganho"].mean(),
        "n no lugar": df[no_lugar].groupby("d_grid").size()})
    print("\nGanho médio por faixa de largada:\n" + t2.to_string(float_format="{:+.2f}".format))

    # verificação: déficit pelo grid médio do carro
    g = df.drop(columns=["grid_ord", "grid_r", "cheg_r", "ganho"]).merge(
        grid_esperado(df), on=["equipe", "ano", "round"], how="inner")
    g = reordena(g, "grid_esp", "deficit_grid")
    rho2, p2 = spearman_perm(g, "deficit_grid", "ganho", rng)
    print(f"\nVerificação (déficit pelo grid médio do carro), {len(g)} linhas: "
          f"Spearman {rho2:+.3f} (p = {p2:.4f})")
    t3 = g.assign(faixa=faixa_deficit(g["deficit_grid"])).groupby(
        "faixa", observed=True)["ganho"].agg(["size", "mean"])
    print(t3.to_string(float_format="{:+.2f}".format))
    print(f"Concordância entre as duas medidas de déficit (Spearman): "
          f"{g['deficit'].corr(g['deficit_grid'], method='spearman'):+.3f}")
    SAIDA.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"medida": "ritmo", "linhas": len(df), "spearman": rho, "p": p},
                  {"medida": "grid", "linhas": len(g), "spearman": rho2, "p": p2}]
                 ).to_csv(SAIDA / "parte_a_spearman.csv", index=False)
    t.to_csv(SAIDA / "parte_a_por_faixa_deficit.csv")
    t2.to_csv(SAIDA / "parte_a_por_faixa_largada.csv")


# ---------------------------------------------------------------------------
# Parte B
# ---------------------------------------------------------------------------
def base_b(df):
    b = df[df["deficit"] >= DEFICIT_MIN].copy()
    b["d_deficit"] = pd.cut(b["deficit"], [1.9, 3.5, 6.5, np.inf],
                            labels=["déficit 2-3,5", "déficit 4-6,5", "déficit 7+"]).astype(str)
    chave = b["deficit"].round().clip(upper=8)
    b["alvo_rec"] = b["ganho"] - b.groupby(chave)["ganho"].transform("mean")
    return b.reset_index(drop=True)


def descricoes(df):
    sel = [(d, v) for d in DESCRITORES for v in sorted(df[d].dropna().unique())]
    masc = {s: (df[s[0]] == s[1]).to_numpy() for s in sel}
    lista, mascs = [], []
    for k in range(1, PROFUNDIDADE + 1):
        for attrs in combinations(DESCRITORES, k):
            vals = [[s for s in sel if s[0] == a] for a in attrs]
            for comb in product(*vals):
                m = np.logical_and.reduce([masc[s] for s in comb])
                if m.sum() >= SUPORTE:
                    lista.append(comb)
                    mascs.append(m)
    return lista, np.array(mascs)


def texto(comb):
    return " E ".join(v for _, v in comb)


def elegiveis(df, cobre):
    circ = pd.factorize(df["circuito"])[0]
    oh = np.eye(circ.max() + 1)[circ]
    contagem = cobre.astype(float) @ oh
    n = cobre.sum(axis=1)
    return ((contagem > 0).sum(axis=1) >= MIN_CIRCUITOS) & \
           (contagem.max(axis=1) / n <= MAX_CIRCUITO)


def zs(cobre, y):
    n = cobre.sum(axis=1)
    return cobre @ (y - y.mean()) / np.sqrt(n) / y.std(ddof=1)


def limiar_busca(df, cobre, eleg, y, rng):
    corridas = list(df.groupby(CORRIDA).indices.values())
    c = cobre[eleg].astype(float)
    maxs = np.empty(PERMUTACOES)
    for i in range(PERMUTACOES):
        e = y.copy()
        for linhas in corridas:
            e[linhas] = y[rng.permutation(linhas)]
        maxs[i] = np.abs(zs(c, e)).max()
    return np.quantile(maxs, 0.95)


def rodada(df, rotulo, semente=2026, salvar=True):
    rng = np.random.default_rng(semente)
    y = df["alvo_rec"].to_numpy()
    lista, cobre = descricoes(df)
    eleg = elegiveis(df, cobre)
    z = zs(cobre.astype(float), y)
    lim = limiar_busca(df, cobre, eleg, y, rng)
    linhas = []
    for i, comb in enumerate(lista):
        sub = df[cobre[i]]
        maior = sub["circuito"].value_counts()
        sem_maior = sub[sub["circuito"] != maior.index[0]]["alvo_rec"].mean()
        eras = sub.groupby("era")["alvo_rec"].mean()
        media = sub["alvo_rec"].mean()
        sinal = np.sign(media)
        firme = (eleg[i] and abs(z[i]) > lim
                 and len(eras) == 2 and (np.sign(eras) == sinal).all()
                 and np.sign(sem_maior) == sinal)
        linhas.append({
            "i": i, "descricao": texto(comb), "condicoes": len(comb),
            "linhas": len(sub), "media": media, "ganho_bruto": sub["ganho"].mean(),
            "deficit_medio": sub["deficit"].mean(), "z": z[i],
            "elegivel": bool(eleg[i]), "acima_limiar": bool(abs(z[i]) > lim),
            "media_2018_21": eras.get("2018-21", np.nan),
            "media_2022": eras.get("2022+", np.nan),
            "media_sem_maior_circuito": sem_maior,
            "circuitos": sub["circuito"].nunique(),
            "maior_circuito_pct": maior.iloc[0] / len(sub),
            "firme": bool(firme)})
    res = pd.DataFrame(linhas)
    firmes = res[res["firme"]]
    se = y.std(ddof=1)
    final = []
    for _, r in firmes.iterrows():
        comb = lista[r["i"]]
        ok = True
        for k in range(1, len(comb)):
            for sub in combinations(comb, k):
                j = next((x for x, c in enumerate(lista) if c == sub), None)
                if j is None or not res.loc[j, "firme"]:
                    continue
                dif = abs(r["media"] - res.loc[j, "media"])
                if dif < MIN_EFEITO or dif < 2 * se / np.sqrt(r["linhas"]):
                    ok = False
        final.append(ok)
    res["final"] = False
    res.loc[firmes.index, "final"] = final

    print(f"\n{'=' * 78}\n{rotulo}\n{'=' * 78}")
    print(f"Linhas {len(df)} ({df.groupby(CORRIDA).ngroups} corridas, "
          f"{df.circuito.nunique()} circuitos); ganho médio {df.ganho.mean():+.2f}; "
          f"desvio do alvo {se:.2f}; descrições {len(lista)} ({int(eleg.sum())} elegíveis)")
    print(f"Limiar |z| {lim:.2f}; acima {int((res.elegivel & res.acima_limiar).sum())}; "
          f"firmes {int(res.firme.sum())}; finais {int(res.final.sum())}")
    cols = ["descricao", "linhas", "media", "ganho_bruto", "deficit_medio", "z",
            "media_2018_21", "media_2022", "media_sem_maior_circuito", "circuitos"]
    fin = res[res["final"]].sort_values("z", key=abs, ascending=False)
    if len(fin):
        print(fin[cols].to_string(index=False, float_format=lambda x: f"{x:+.2f}",
                                  max_colwidth=60))
    top = res[res.elegivel].sort_values("z", key=abs, ascending=False).head(8)
    print("\nMaiores |z| entre as elegíveis (informativo):")
    print(top[cols + ["firme"]].to_string(index=False, float_format=lambda x: f"{x:+.2f}",
                                          max_colwidth=60))
    if salvar:
        SAIDA.mkdir(parents=True, exist_ok=True)
        nome = rotulo.lower().replace(" ", "_")
        res.drop(columns="i").sort_values("z", key=abs, ascending=False).to_csv(
            SAIDA / f"subgrupos_{nome}.csv", index=False)
        pd.DataFrame([{"versao": rotulo, "linhas": len(df), "descricoes": len(lista),
                       "elegiveis": int(eleg.sum()), "limiar_z": lim,
                       "firmes": int(res.firme.sum()), "finais": int(res.final.sum())}]
                     ).to_csv(SAIDA / f"resumo_{nome}.csv", index=False)
    return res


# ---------------------------------------------------------------------------
# Parte C
# ---------------------------------------------------------------------------
def parte_c(b):
    print(f"\n{'=' * 78}\nPARTE C - recuperação por circuito x Etapa 1 (descritiva)\n{'=' * 78}")
    g = b.groupby("circuito").agg(linhas=("ganho", "size"), ganho=("ganho", "sum"),
                                  deficit=("deficit", "sum"), perfil=("d_perfil", "first"),
                                  N1=("N1", "first"))
    g = g[g["linhas"] >= MIN_LINHAS_CIRCUITO].copy()
    g["recuperacao"] = g["ganho"] / g["deficit"]
    f = pd.read_csv(FEATURES)
    chave = "circuito" if "circuito" in f.columns else f.columns[0]
    g = g.join(f.set_index(chave)[FEATS_ETAPA1], how="left")
    rng = np.random.default_rng(2026)
    linhas = []
    rec = g["recuperacao"].rank().to_numpy()
    for x in ["N1"] + FEATS_ETAPA1:
        ok = g[x].notna().to_numpy()
        xr = g.loc[ok, x].rank().to_numpy()
        rr = pd.Series(g.loc[ok, "recuperacao"]).rank().to_numpy()
        rho = np.corrcoef(xr, rr)[0, 1]
        perm = np.array([np.corrcoef(xr, rng.permutation(rr))[0, 1] for _ in range(10000)])
        p = (np.sum(np.abs(perm) >= abs(rho)) + 1) / 10001
        linhas.append({"medida": x, "circuitos": int(ok.sum()), "spearman": rho, "p": p})
    t = pd.DataFrame(linhas).sort_values("p")
    m = len(t)
    t["p_holm"] = np.minimum(1, np.maximum.accumulate((m - np.arange(m)) * t["p"].to_numpy()))
    print(f"{len(g)} circuitos com {MIN_LINHAS_CIRCUITO}+ linhas de déficit >= {DEFICIT_MIN}")
    print(t.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
    print("\nRecuperação por circuito (fração do déficit recuperada):")
    print(g.sort_values("recuperacao", ascending=False)[["perfil", "linhas", "recuperacao", "N1"]]
          .to_string(float_format=lambda x: f"{x:.2f}"))
    pp = b.groupby("d_perfil").agg(linhas=("ganho", "size"), circuitos=("circuito", "nunique"),
                                   ganho=("ganho", "sum"), deficit=("deficit", "sum"))
    pp["recuperacao"] = pp["ganho"] / pp["deficit"]
    print("\nPor perfil (descritivo):\n" + pp.to_string(float_format=lambda x: f"{x:.2f}"))
    t.to_csv(SAIDA / "parte_c_spearman.csv", index=False)
    g.to_csv(SAIDA / "parte_c_por_circuito.csv")
    pp.to_csv(SAIDA / "parte_c_por_perfil.csv")


# ---------------------------------------------------------------------------
def sintetico():
    b = base_b(monta_base())
    rng = np.random.default_rng(7)
    a = b.copy()
    y = a["alvo_rec"].to_numpy().copy()
    for linhas in a.groupby(CORRIDA).indices.values():
        y[linhas] = y[rng.permutation(linhas)]
    a["alvo_rec"] = y
    rodada(a, "SINTETICO sem efeito", semente=11)
    c = a.copy()
    m = np.logical_and.reduce([(c[k] == v).to_numpy() for k, v in PLANTADO.items()])
    c.loc[m, "alvo_rec"] += 1.5
    print(f"\nPlantado +1,5 em {m.sum()} linhas")
    rodada(c, "SINTETICO com efeito", semente=12)


def real():
    df = monta_base()
    parte_a(df)
    b = base_b(df)
    rodada(b, "PARTE B - BASE REAL")
    parte_c(b)


if __name__ == "__main__":
    sintetico() if "--sintetico" in sys.argv else real()
