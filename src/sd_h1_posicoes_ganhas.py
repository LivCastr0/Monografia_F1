"""
Etapa 2 - Hipótese (i) original do relatório: estratégia -> posições ganhas.

Critérios fixados antes de rodar: claude/h1-posicoes-ganhas-msi2.md (projeto no
claude.ai). Exploratória declarada: definida depois das outras análises da
Etapa 2, na mesma base.

Alvo: posições ganhas (largada - chegada), as duas reordenadas entre os pilotos
que terminaram a corrida, menos a média de quem largou na mesma posição do grid.
Descritores: os 7 da rodada 2. Busca de profundidade 1 a 3, suporte 50, limiar
por permutação dentro da corrida, firmeza por eras e sem o circuito mais
frequente, redundância, e o teste "o perfil acrescenta".

Uso:
  python src/sd_h1_posicoes_ganhas.py               base real
  python src/sd_h1_posicoes_ganhas.py --sintetico   os dois testes sintéticos
"""
import sys
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pandas as pd

BASE_SD = Path("data/processed/base_analise_sd.csv")
BASE_PC = Path("data/processed/base_piloto_corrida.csv")
SAIDA = Path("data/processed/sd_h1_posicoes_ganhas")

DESCRITORES = ["d_perfil", "d_grid", "d_paradas", "d_momento",
               "d_composto", "d_chuva", "d_safety_car"]
SUPORTE = 50
PROFUNDIDADE = 3
MIN_CIRCUITOS = 4
MAX_CIRCUITO = 0.40
PERMUTACOES = 1000
MIN_EFEITO = 0.3
CORRIDA = ["ano", "round"]
PAROU_MAIS = "parou mais que o pelotão"
PLANTADO = {"d_perfil": "stop-and-go", "d_momento": "parou antes do pelotão"}


# ---------------------------------------------------------------------------
# Base e alvo
# ---------------------------------------------------------------------------
def monta_base():
    sd = pd.read_csv(BASE_SD)
    pc = pd.read_csv(BASE_PC)
    fin = pc[pc["terminou"]].copy()
    # largada do pit lane (grid 0) conta como última
    fin["grid_ord"] = fin["grid"].replace(0, 99)
    fin["grid_reord"] = fin.groupby(CORRIDA)["grid_ord"].rank(method="first")
    fin["cheg_reord"] = fin.groupby(CORRIDA)["posicao_final"].rank(method="first")
    fin["ganho_reord"] = fin["grid_reord"] - fin["cheg_reord"]
    df = sd.merge(fin[CORRIDA + ["piloto", "ganho_reord", "posicoes_ganhas"]],
                  on=CORRIDA + ["piloto"], how="left", validate="1:1")
    assert df["ganho_reord"].notna().all()
    df["era"] = np.where(df["ano"] <= 2021, "2018-21", "2022+")
    return df.reset_index(drop=True)


def ajusta_largada(df):
    """Desconta a média de quem largou na mesma posição (base inteira)."""
    df = df.copy()
    df["alvo_h1"] = df["ganho_reord"] - df.groupby("grid")["ganho_reord"].transform("mean")
    df["alvo_h1_cru"] = (df["posicoes_ganhas"]
                         - df.groupby("grid")["posicoes_ganhas"].transform("mean"))
    return df


# ---------------------------------------------------------------------------
# Descrições
# ---------------------------------------------------------------------------
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
    return " E ".join(f"{a[2:]}={v}" for a, v in comb)


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


# ---------------------------------------------------------------------------
# O perfil acrescenta?
# ---------------------------------------------------------------------------
def prepara_acrescenta(df, lista, eleg):
    """Descrições elegíveis com perfil: índice, perfil P e máscara do resto (C)."""
    itens = []
    for i, comb in enumerate(lista):
        if not eleg[i] or not any(a == "d_perfil" for a, _ in comb):
            continue
        resto = [s for s in comb if s[0] != "d_perfil"]
        base_c = (np.logical_and.reduce([(df[a] == v).to_numpy() for a, v in resto])
                  if resto else np.ones(len(df), bool))
        itens.append((i, dict(comb)["d_perfil"], base_c))
    return itens


def z_acrescenta(itens, perfil, y):
    """C E perfil P contra C nos outros perfis, para cada item."""
    sd = y.std(ddof=1)
    out = {}
    for i, p, base_c in itens:
        dentro = base_c & (perfil == p)
        fora = base_c & (perfil != p)
        n1, n2 = dentro.sum(), fora.sum()
        if n1 < 2 or n2 < 2:
            continue
        out[i] = (y[dentro].mean() - y[fora].mean()) / (sd * np.sqrt(1 / n1 + 1 / n2))
    return out


def limiar_acrescenta(df, itens, y, rng):
    """Perfis permutados entre circuitos (mesmos tamanhos), alvo fixo."""
    mapa = df.groupby("circuito")["d_perfil"].first()
    circ_idx = pd.Categorical(df["circuito"], categories=mapa.index).codes
    rot = mapa.to_numpy()
    maxs = np.empty(PERMUTACOES)
    for k in range(PERMUTACOES):
        perfil = rng.permutation(rot)[circ_idx]
        z = z_acrescenta(itens, perfil, y)
        maxs[k] = np.abs(list(z.values())).max()
    return np.quantile(maxs, 0.95)


# ---------------------------------------------------------------------------
# Rodada completa
# ---------------------------------------------------------------------------
def rodada(df, rotulo, semente=2026, perm_perfil=True, salvar=True):
    rng = np.random.default_rng(semente)
    y = df["alvo_h1"].to_numpy()
    lista, cobre = descricoes(df)
    eleg = elegiveis(df, cobre)
    z = zs(cobre.astype(float), y)
    lim = limiar_busca(df, cobre, eleg, y, rng)

    linhas = []
    for i, comb in enumerate(lista):
        sub = df[cobre[i]]
        maior = sub["circuito"].value_counts()
        sem_maior = sub[sub["circuito"] != maior.index[0]]["alvo_h1"].mean()
        eras = sub.groupby("era")["alvo_h1"].mean()
        media = sub["alvo_h1"].mean()
        sinal = np.sign(media)
        firme = (eleg[i] and abs(z[i]) > lim
                 and len(eras) == 2 and (np.sign(eras) == sinal).all()
                 and np.sign(sem_maior) == sinal)
        linhas.append({
            "i": i, "descricao": texto(comb), "condicoes": len(comb),
            "linhas": len(sub), "media": media, "z": z[i],
            "elegivel": bool(eleg[i]), "acima_limiar": bool(abs(z[i]) > lim),
            "media_2018_21": eras.get("2018-21", np.nan),
            "media_2022": eras.get("2022+", np.nan),
            "media_sem_maior_circuito": sem_maior,
            "media_alvo_cru": sub["alvo_h1_cru"].mean(),
            "circuitos": sub["circuito"].nunique(),
            "maior_circuito_pct": maior.iloc[0] / len(sub),
            "firme": bool(firme),
        })
    res = pd.DataFrame(linhas)

    # redundância entre firmes
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

    # perfil acrescenta
    res["z_acrescenta"] = np.nan
    res["perfil_acrescenta"] = False
    lim_acr = np.nan
    itens = prepara_acrescenta(df, lista, eleg)
    za = z_acrescenta(itens, df["d_perfil"].to_numpy(), y)
    for i, v in za.items():
        res.loc[i, "z_acrescenta"] = v
    if perm_perfil and za:
        lim_acr = limiar_acrescenta(df, itens, y, rng)
        res["perfil_acrescenta"] = res["firme"] & (res["z_acrescenta"].abs() > lim_acr)

    print(f"\n{'=' * 78}\n{rotulo}\n{'=' * 78}")
    print(f"Linhas {len(df)}; desvio do alvo {y.std(ddof=1):.2f}; "
          f"descrições {len(lista)} ({int(eleg.sum())} elegíveis)")
    print(f"Limiar |z| {lim:.2f}; acima {int((res.elegivel & res.acima_limiar).sum())}; "
          f"firmes {int(res.firme.sum())}; finais {int(res.final.sum())}")
    print(f"Descrições com perfil testadas no 'acrescenta': {len(za)}; "
          f"limiar |z_acr| {lim_acr:.2f}; maior |z_acr| "
          f"{np.nanmax(np.abs(list(za.values()))) if za else np.nan:.2f}; "
          f"firmes em que o perfil acrescenta: {int(res.perfil_acrescenta.sum())}")
    cols = ["descricao", "linhas", "media", "z", "media_2018_21", "media_2022",
            "media_sem_maior_circuito", "media_alvo_cru", "circuitos"]
    fin = res[res["final"]].sort_values("z", key=abs, ascending=False)
    print(fin[cols].to_string(index=False, float_format=lambda x: f"{x:+.2f}",
                              max_colwidth=70))
    if salvar:
        SAIDA.mkdir(parents=True, exist_ok=True)
        nome = rotulo.lower().replace(" ", "_").replace("'", "")
        res.drop(columns="i").sort_values("z", key=abs, ascending=False).to_csv(
            SAIDA / f"subgrupos_{nome}.csv", index=False)
        pd.DataFrame([{"versao": rotulo, "linhas": len(df), "descricoes": len(lista),
                       "elegiveis": int(eleg.sum()), "limiar_z": lim,
                       "firmes": int(res.firme.sum()), "finais": int(res.final.sum()),
                       "limiar_z_acrescenta": lim_acr,
                       "perfil_acrescenta": int(res.perfil_acrescenta.sum())}]
                     ).to_csv(SAIDA / f"resumo_{nome}.csv", index=False)
    return res


def sintetico():
    df = ajusta_largada(monta_base())
    rng = np.random.default_rng(7)
    # sem efeito: alvo real embaralhado dentro da corrida
    a = df.copy()
    y = a["alvo_h1"].to_numpy().copy()
    for linhas in a.groupby(CORRIDA).indices.values():
        y[linhas] = y[rng.permutation(linhas)]
    a["alvo_h1"] = y
    rodada(a, "SINTETICO sem efeito", semente=11, salvar=True)
    # efeito plantado
    b = a.copy()
    m = np.logical_and.reduce([(b[k] == v).to_numpy() for k, v in PLANTADO.items()])
    b.loc[m, "alvo_h1"] += 1.5
    print(f"\nPlantado +1,5 em {m.sum()} linhas")
    rodada(b, "SINTETICO com efeito", semente=12, salvar=True)


def real():
    df = ajusta_largada(monta_base())
    print(f"Alvo: média {df.alvo_h1.mean():+.3f}; Spearman com a largada "
          f"antes {df.ganho_reord.corr(df.grid, method='spearman'):+.3f}, depois "
          f"{df.alvo_h1.corr(df.grid, method='spearman'):+.3f}")
    rodada(df, "BASE REAL")
    sem = ajusta_largada(df[df["d_paradas"] != PAROU_MAIS].reset_index(drop=True))
    rodada(sem, "SEM PAROU MAIS")


if __name__ == "__main__":
    sintetico() if "--sintetico" in sys.argv else real()
