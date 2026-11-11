"""
Etapa 2 - Perfil de circuito x padrões do safety car (H9 e H11).

Critérios fixados antes de rodar: claude/safety-car-chuva-msi2.md, seção "Perfil de
circuito x H9 e H11". Pedido depois dos resultados; exploratório.

Teste de partições (como valida_sd_por_cluster.py) para alvo numérico: Q/gl médio da
heterogeneidade do efeito das descrições entre os perfis com 4+ circuitos, contra 2.000
partições aleatórias dos circuitos com os mesmos tamanhos. Poder: +1 e +2 posições
plantadas em "parou no SC" no perfil com mais circuitos.

Uso: python src/perfil_safety_car.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sd_eventos as S  # noqa: E402
import sd_safety_car as SC  # noqa: E402

P = Path("data/processed")
SAIDA = P / "perfil_safety_car"
NOMES = {0: "alto downforce", 1: "stop-and-go", 2: "potência", 3: "urbano-técnico"}
SUPORTE, PARTICOES, SEMENTE, MIN_CIRC = 15, 2000, 2026, 4


def perfis():
    c = pd.read_csv(P / "clusters_circuitos.csv").set_index("circuito")
    return c["cluster_kmeans_so_dinamicas_k4"].map(NOMES)


def estatisticas(df, alvo):
    df = df.sort_values("circuito").reset_index(drop=True)
    descs = S.descricoes(df, SC.DESCRITORES)
    col = {c: df[c].astype(str).to_numpy() for c in SC.DESCRITORES}
    M = np.array([np.all([col[c] == v for c, v in d], axis=0) for d in descs])
    y = df[alvo].to_numpy(float)
    circ = df["circuito"].to_numpy()
    nomes = np.unique(circ)
    idx = [circ == c for c in nomes]
    n_dc = np.stack([M[:, i].sum(1) for i in idx], 1).astype(float)
    s_dc = np.stack([(M[:, i] * y[i]).sum(1) for i in idx], 1)
    n_c = np.array([i.sum() for i in idx], float)
    s_c = np.array([y[i].sum() for i in idx])
    q_c = np.array([(y[i] ** 2).sum() for i in idx])
    return descs, nomes, n_dc, s_dc, n_c, s_c, q_c


def heterogeneidade(rot, grupos, n_dc, s_dc, n_c, s_c, q_c):
    L, W = [], []
    for g in grupos:
        m = rot == g
        n_d, s_d = n_dc[:, m].sum(1), s_dc[:, m].sum(1)
        n_g, s_g, q_g = n_c[m].sum(), s_c[m].sum(), q_c[m].sum()
        mu = s_g / n_g
        var = max(q_g / n_g - mu ** 2, 1e-9)
        ok = n_d >= SUPORTE
        with np.errstate(invalid="ignore", divide="ignore"):
            L.append(np.where(ok, s_d / n_d - mu, 0.0))
        W.append(np.where(ok, n_d / var, 0.0))
    L, W = np.array(L), np.array(W)
    k = (W > 0).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        lbar = (W * L).sum(0) / W.sum(0)
        q = (W * (L - lbar) ** 2).sum(0) / (k - 1)
    q[k < 2] = np.nan
    return q, L


def teste(df, alvo, rng, rotulo=None):
    descs, nomes, n_dc, s_dc, n_c, s_c, q_c = estatisticas(df, alvo)
    rot = (rotulo if rotulo is not None else perfis()).reindex(nomes).to_numpy()
    tam = pd.Series(rot).value_counts()
    grupos = sorted(g for g, t in tam.items() if t >= MIN_CIRC)
    q, L = heterogeneidade(rot, grupos, n_dc, s_dc, n_c, s_c, q_c)
    obs = np.nanmean(q)
    base = np.concatenate([[g] * t for g, t in tam.items()])
    med = np.array([np.nanmean(heterogeneidade(rng.permutation(base), grupos, n_dc, s_dc,
                                               n_c, s_c, q_c)[0]) for _ in range(PARTICOES)])
    p = (np.sum(med >= obs) + 1) / (PARTICOES + 1)
    tab = pd.DataFrame({"descricao": [S.texto(d) for d in descs], "Q_por_gl": q})
    for j, g in enumerate(grupos):
        apoio = n_dc[:, rot == g].sum(1) >= SUPORTE          # sem suporte no perfil: vazio
        tab[f"efeito_{g}"] = np.where(np.isfinite(q) & apoio, L[j], np.nan)
    tab = tab.dropna(subset=["Q_por_gl"]).sort_values("Q_por_gl", ascending=False)
    return dict(tamanhos="/".join(str(tam[g]) for g in grupos) + f" (fora: {', '.join(g for g in tam.index if g not in grupos)})",
                descricoes=int(np.isfinite(q).sum()), Q_medio_real=obs,
                Q_mediana_acaso=float(np.median(med)), Q_95_acaso=float(np.quantile(med, 0.95)),
                p=p), tab


def main():
    SAIDA.mkdir(parents=True, exist_ok=True)
    h9 = pd.read_csv(P / "base_safety_car.csv")
    h11 = pd.read_csv(P / "sd_resultado_final" / "base_safety_car_final.csv")
    pf = perfis()
    print("Circuitos por perfil na base do SC:",
          pf.reindex(h9["circuito"].unique()).value_counts().to_dict())
    print("Períodos por perfil:", h9.drop_duplicates("evento")["circuito"].map(pf)
          .value_counts().to_dict())
    linhas = []
    for nome, df, alvo in [("H9 (3 voltas)", h9, "ganho"), ("H11 (fim da corrida)", h11, "ganho_final")]:
        r, tab = teste(df, alvo, np.random.default_rng(SEMENTE))
        r["analise"] = nome
        linhas.append(r)
        tab.to_csv(SAIDA / f"heterogeneidade_{alvo}.csv", index=False)
        print(f"\n{nome}: perfis {r['tamanhos']}; {r['descricoes']} descrições; Q/gl médio "
              f"{r['Q_medio_real']:.2f} contra mediana {r['Q_mediana_acaso']:.2f} "
              f"(95% {r['Q_95_acaso']:.2f}); p = {r['p']:.4f}")
        print(tab.head(6).round(3).to_string(index=False))
    # poder
    maior = pf.reindex(h9["circuito"].unique()).value_counts().index[0]
    for efeito in (1.0, 2.0):
        x = h9.copy()
        alvo = (x["circuito"].map(pf) == maior) & (x["d_parou"] == "parou no SC")
        x.loc[alvo, "ganho"] += efeito
        r, _ = teste(x, "ganho", np.random.default_rng(SEMENTE))
        r["analise"] = f"poder: +{efeito:g} em parou no SC só em {maior}"
        linhas.append(r)
        print(f"\n{r['analise']}: Q/gl médio {r['Q_medio_real']:.2f} contra 95% "
              f"{r['Q_95_acaso']:.2f}; p = {r['p']:.4f}")
    pd.DataFrame(linhas).to_csv(SAIDA / "teste_particoes.csv", index=False)
    d = h9.assign(perfil=h9["circuito"].map(pf))
    desc = d.pivot_table(index="perfil", columns="d_parou", values="ganho", aggfunc="mean")
    desc["periodos"] = d.drop_duplicates("evento")["perfil"].value_counts()
    desc.to_csv(SAIDA / "descritivo_parou_por_perfil.csv")
    print("\nDescritivo (H9): ganho médio por perfil e decisão\n" + desc.round(2).to_string())


if __name__ == "__main__":
    main()
