"""
Etapa 2 - Descoberta de subgrupos para bases organizadas por evento (período de safety
car, corrida com chuva). Usada por sd_safety_car.py e sd_chuva.py.

Procedimento (o mesmo de duelos e decisão de parar; critérios em
claude/safety-car-chuva-msi2.md):
  - descrições de profundidade 1 e 2 sobre os descritores dados;
  - elegível: suporte mínimo, 4+ circuitos, nenhum circuito acima de 40% das linhas;
  - z = (média no subgrupo - média geral) * sqrt(n) / desvio;
  - limiar: 95% do máximo |z| em N permutações do alvo dentro do evento;
  - firme: limiar + mesmo sentido sem o circuito mais frequente + mesmo sentido nas
    duas eras (com min_era, a checagem das eras só vale quando o subgrupo tem min_era
    linhas em cada era);
  - redundância: um subgrupo de 2 condições fica se a média difere em pelo menos
    min_red e 2 erros-padrão da média de cada condição sozinha.
"""
from itertools import combinations

import numpy as np
import pandas as pd

MIN_CIRCUITOS, MAX_CIRCUITO = 4, 0.40


def descricoes(df, descritores):
    sel = [(c, v) for c in descritores for v in sorted(df[c].dropna().astype(str).unique())]
    return [(s,) for s in sel] + [(a, b) for a, b in combinations(sel, 2) if a[0] != b[0]]


def texto(d):
    return " E ".join(f"{c[2:]}={v}" for c, v in d)


def busca(df, alvo, descritores, evento, suporte, min_red, rng, permutacoes=500,
          min_era=None):
    df = df.dropna(subset=[alvo]).reset_index(drop=True)
    y = df[alvo].to_numpy(dtype=float)
    circ = df["circuito"].to_numpy()
    era = df["d_era"].to_numpy()
    descs = descricoes(df, descritores)
    col = {c: df[c].astype(str).to_numpy() for c in descritores}
    M = np.array([np.all([col[c] == v for c, v in d], axis=0) for d in descs])
    n = M.sum(1)
    codigos, ci = np.unique(circ, return_inverse=True)
    cont = np.zeros((len(descs), len(codigos)), dtype=int)
    for j in range(len(codigos)):
        cont[:, j] = M[:, ci == j].sum(1)
    ok = (n >= suporte) & ((cont > 0).sum(1) >= MIN_CIRCUITOS) & \
         (cont.max(1) <= MAX_CIRCUITO * np.maximum(n, 1))
    idx = np.flatnonzero(ok)
    Mf = M[idx].astype(np.float64)
    media, dp = y.mean(), y.std(ddof=1)
    z = (Mf @ (y - media)) / np.sqrt(n[idx]) / dp

    grupos = list(pd.Series(np.arange(len(y))).groupby(df[evento].to_numpy()).indices.values())
    maximos = np.empty(permutacoes)
    for i in range(permutacoes):
        v = y.copy()
        for g in grupos:
            v[g] = y[rng.permutation(g)]
        zp = (Mf @ (v - v.mean())) / np.sqrt(n[idx]) / v.std(ddof=1)
        maximos[i] = np.abs(zp).max() if len(zp) else 0.0
    limiar = float(np.quantile(maximos, 0.95))

    linhas = []
    for j, k in enumerate(idx):
        m = M[k]
        s = np.sign(y[m].mean() - media)
        top = codigos[cont[k].argmax()]
        sem = m & (circ != top)
        sem_ok = sem.any() and np.sign(y[sem].mean() - media) == s
        por_era = {e: m & (era == e) for e in ("2018-21", "2022+")}
        n_era = {e: int(v.sum()) for e, v in por_era.items()}
        if min_era is None:
            eras_ok = all(n_era[e] > 0 and np.sign(y[por_era[e]].mean() - media) == s
                          for e in por_era)
            eras_checada = True
        elif all(n_era[e] >= min_era for e in por_era):
            eras_ok = all(np.sign(y[por_era[e]].mean() - media) == s for e in por_era)
            eras_checada = True
        else:
            eras_ok, eras_checada = True, False
        passou = abs(z[j]) > limiar
        linhas.append({
            "chave": descs[k], "descricao": texto(descs[k]), "condicoes": len(descs[k]),
            "linhas": int(n[k]), "circuitos": int((cont[k] > 0).sum()),
            "eventos": int(df.loc[m, evento].nunique()), "media": y[m].mean(),
            "diferenca": y[m].mean() - media, "ep": y[m].std(ddof=1) / np.sqrt(n[k]),
            "z": float(z[j]), "acima_do_limiar": bool(passou),
            "eras_checada": eras_checada,
            "media_2018_21": y[por_era["2018-21"]].mean() if n_era["2018-21"] else np.nan,
            "media_2022_mais": y[por_era["2022+"]].mean() if n_era["2022+"] else np.nan,
            "firme": bool(passou and sem_ok and eras_ok)})
    res = pd.DataFrame(linhas)
    if res.empty:
        res.attrs.update(limiar=limiar, media=media, linhas_base=len(df), elegiveis=0)
        return res
    medias = {r["chave"]: r["media"] for _, r in res.iterrows()}
    simples = {d: (y[M[descs.index(d)]].mean()) for d in descs if len(d) == 1}

    def nao_redundante(r):
        if r["condicoes"] == 1:
            return True
        for s in r["chave"]:
            pai = simples.get((s,), np.nan)
            d = abs(r["media"] - pai)
            if not (d >= min_red and d >= 2 * r["ep"]):
                return False
        return True

    res["nao_redundante"] = res.apply(nao_redundante, axis=1)
    res["final"] = res["firme"] & res["nao_redundante"]
    res = res.sort_values("z", key=abs, ascending=False)
    res.attrs.update(limiar=limiar, media=media, linhas_base=len(df), elegiveis=len(idx))
    _ = medias
    return res


def resume(nome, res, n_mostrar=12, fmt="{:+.3f}"):
    a = res.attrs
    print(f"\n[{nome}] {a['linhas_base']} linhas; média geral {a['media']:.4f}; elegíveis "
          f"{a['elegiveis']}; limiar |z| {a['limiar']:.2f}")
    if res.empty:
        print("  nenhuma descrição elegível")
        return
    print(f"  acima do limiar {int(res['acima_do_limiar'].sum())}; firmes "
          f"{int(res['firme'].sum())}; finais (firmes e não redundantes) {int(res['final'].sum())}")
    cols = ["descricao", "linhas", "circuitos", "eventos", "media", "diferenca", "z", "firme",
            "final"]
    with pd.option_context("display.width", 220, "display.max_colwidth", 70):
        print(res[cols].head(n_mostrar).to_string(index=False,
                                                  float_format=lambda x: fmt.format(x)))
