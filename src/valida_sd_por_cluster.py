"""
Etapa 2 - SD dentro de cada cluster (uma busca por perfil) e as duas validações.

Pergunta: os padrões de comportamento de corrida mudam conforme o tipo de circuito?
O cluster da Etapa 1 NÃO entra como descritor: ele define onde cada busca acontece.

Entra: data/processed/base_decisao_parada.csv  (alvo parou)
       data/processed/base_duelos.csv          (alvo passou)
Sai:   data/processed/sd_por_cluster/<base>_subgrupos.csv   (global e cada cluster)
       data/processed/sd_por_cluster/<base>_heterogeneidade.csv

Uso:   python src/valida_sd_por_cluster.py            (as duas bases)
       python src/valida_sd_por_cluster.py --sintetico  (teste do script)

Desenho fixado antes de rodar
-----------------------------
Busca (igual no global e em cada cluster):
  - descritores das análises originais, SEM perfil e SEM era (a era fica só no
    diagnóstico, como previsto no relatório);
  - todas as descrições de profundidade 1 e 2;
  - suporte mínimo absoluto igual ao da análise original (200 voltas na parada,
    100 voltas de duelo), 4+ circuitos e nenhum circuito acima de 40% das linhas;
  - medida z = (média no subgrupo - média da busca) * sqrt(n) / desvio da busca.
    No cluster, a referência é a média DO CLUSTER;
  - limiar: 95% do máximo |z| em 500 permutações do alvo dentro da corrida;
  - firme: passa do limiar, mesmo sentido nas duas eras e sem o circuito mais
    frequente.
  - Urbano-técnico (2 circuitos) é só descritivo: não há busca nem teste.

Validação 1 - local × global:
  para cada cluster, quantos firmes locais também são firmes no global com o mesmo
  sentido, quantos são só locais, e a correlação de Spearman entre o efeito local
  (média no subgrupo - média do cluster) e o efeito global das mesmas descrições.

Validação 2 - permutação de partições:
  para cada descrição, Q = heterogeneidade do efeito entre os três clusters testados
  (soma ponderada pelo inverso da variância dos desvios em torno do efeito médio).
  Estatísticas: média de Q/gl e máximo de Q/gl entre as descrições. Referência:
  2000 partições aleatórias dos 26 circuitos com os tamanhos reais (2/5/8/11); o
  grupo de 2 circuitos fica fora do cálculo, como o urbano-técnico real.
  Uma descrição só entra em Q nos clusters onde tem pelo menos o suporte mínimo, e
  só se aparecer em 2+ clusters. Por descrição: Q real contra o máximo das
  permutações (correção para todas as descrições).
  Os circuitos são a unidade embaralhada, então corridas repetidas do mesmo circuito
  não contam como observações independentes.

Descoberta de subgrupos descreve associação, não causa.
"""
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

PASTA = Path("data/processed/sd_por_cluster")
URBANO = "urbano-técnico"
PERMUTACOES, PARTICOES, SEMENTE = 500, 2000, 2026
MIN_CIRCUITOS, MAX_CIRCUITO = 4, 0.40

BASES = {
    "decisao_parada": dict(
        arquivo="data/processed/base_decisao_parada.csv", alvo="parou", suporte=200,
        descritores=["d_idade_pneu", "d_composto", "d_neutralizacao", "d_rival_frente",
                     "d_rival_tras", "d_janela", "d_dist_frente", "d_paradas_feitas",
                     "d_fase", "d_posicao"]),
    "duelos": dict(
        arquivo="data/processed/base_duelos.csv", alvo="passou", suporte=100,
        descritores=["d_dif_idade", "d_dif_composto", "d_distancia", "d_trem", "d_largada",
                     "d_fase", "d_relargada", "d_posicao", "d_mesma_equipe"]),
}


def titulo(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


# ---------------------------------------------------------------------------
# Descrições e cobertura
# ---------------------------------------------------------------------------
def descricoes(df, descritores):
    sel = [(c, v) for c in descritores for v in sorted(df[c].dropna().unique())]
    unicos = [(s,) for s in sel]
    pares = [(a, b) for a, b in combinations(sel, 2) if a[0] != b[0]]
    return unicos + pares


def cobertura(df, descs):
    cols = {c: df[c].to_numpy() for c in {c for d in descs for c, _ in d}}
    base = {}
    m = np.empty((len(descs), len(df)), dtype=bool)
    for k, d in enumerate(descs):
        r = np.ones(len(df), dtype=bool)
        for s in d:
            if s not in base:
                base[s] = cols[s[0]] == s[1]
            r &= base[s]
        m[k] = r
    return m


def texto(d):
    return " E ".join(f"{c[2:]}={v}" for c, v in d)


# ---------------------------------------------------------------------------
# Busca com limiar por permutação (uma "busca" = um recorte da base)
# ---------------------------------------------------------------------------
def busca(df, cob, descs, alvo, suporte, rng):
    """df ordenado por circuito, ano, round. Devolve a tabela de descrições elegíveis."""
    y = df[alvo].to_numpy(dtype=float)
    circ = df["circuito"].to_numpy()
    inicios = np.flatnonzero(np.r_[True, circ[1:] != circ[:-1]])
    cont = np.add.reduceat(cob, inicios, axis=1, dtype=np.int32)      # desc x circuito
    n = cont.sum(axis=1)
    ok = ((n >= suporte) & ((cont > 0).sum(axis=1) >= MIN_CIRCUITOS)
          & (cont.max(axis=1) <= MAX_CIRCUITO * np.maximum(n, 1)))
    idx = np.flatnonzero(ok)
    if len(idx) == 0:
        return pd.DataFrame()
    c = cob[idx].astype(np.float32)
    media, dp = y.mean(), y.std(ddof=1)
    z = (c @ (y - media).astype(np.float32)) / np.sqrt(n[idx]) / dp

    corrida = pd.factorize(df["ano"].astype(str) + "_" + df["round"].astype(str))[0]
    ordem_corrida = np.argsort(corrida, kind="stable")
    maximos = np.empty(PERMUTACOES)
    for i in range(PERMUTACOES):
        p = np.lexsort((rng.random(len(df)), corrida[ordem_corrida]))
        v = np.empty_like(y)
        v[ordem_corrida] = y[ordem_corrida][p]                       # embaralha dentro da corrida
        zp = (c @ (v - v.mean()).astype(np.float32)) / np.sqrt(n[idx]) / v.std(ddof=1)
        maximos[i] = np.abs(zp).max()
    limiar = float(np.quantile(maximos, 0.95))

    era = df["d_era"].to_numpy()
    linhas = []
    for j, k in enumerate(idx):
        m = cob[k]
        s = np.sign(y[m].mean() - media)
        conta = pd.Series(circ[m]).value_counts()
        sem = m & (circ != conta.index[0])
        por_era = {e: y[m & (era == e)].mean() for e in ("2018-21", "2022+") if (m & (era == e)).any()}
        eras_ok = len(por_era) == 2 and all(np.sign(v - media) == s for v in por_era.values())
        sem_ok = sem.any() and np.sign(y[sem].mean() - media) == s
        passou = abs(z[j]) > limiar
        linhas.append({"chave": descs[k], "descricao": texto(descs[k]), "linhas": int(n[k]),
                       "circuitos": len(conta), "media": y[m].mean(), "efeito": y[m].mean() - media,
                       "z": float(z[j]), "acima_do_limiar": passou,
                       "firme": bool(passou and eras_ok and sem_ok),
                       "media_2018_21": por_era.get("2018-21", np.nan),
                       "media_2022_mais": por_era.get("2022+", np.nan)})
    res = pd.DataFrame(linhas).sort_values("z", key=abs, ascending=False)
    res.attrs.update(limiar=limiar, media=media, linhas_base=len(df))
    return res


# ---------------------------------------------------------------------------
# Validação 2: heterogeneidade entre clusters e permutação de partições
# ---------------------------------------------------------------------------
def estatisticas_por_circuito(df, cob, alvo):
    """Contagens e somas por descrição x circuito, para recombinar partições rápido."""
    y = df[alvo].to_numpy(dtype=float)
    circ = df["circuito"].to_numpy()
    inicios = np.flatnonzero(np.r_[True, circ[1:] != circ[:-1]])
    nomes = circ[inicios]
    n_dc = np.add.reduceat(cob, inicios, axis=1, dtype=np.int64).astype(float)
    s_dc = np.add.reduceat(cob * y, inicios, axis=1)
    n_c = np.add.reduceat(np.ones(len(df)), inicios)
    s_c = np.add.reduceat(y, inicios)
    return nomes, n_dc, s_dc, n_c, s_c


def heterogeneidade(rotulos, grupos, n_dc, s_dc, n_c, s_c, suporte):
    """Q/gl por descrição entre os grupos dados (rótulo por circuito)."""
    L, W = [], []
    for g in grupos:
        sel = rotulos == g
        n_d, s_d = n_dc[:, sel].sum(1), s_dc[:, sel].sum(1)
        n_g, s_g = n_c[sel].sum(), s_c[sel].sum()
        p_g = s_g / n_g
        var = max(p_g * (1 - p_g), 1e-9)
        valido = n_d >= suporte
        with np.errstate(invalid="ignore", divide="ignore"):
            L.append(np.where(valido, s_d / n_d - p_g, 0.0))
        W.append(np.where(valido, n_d / var, 0.0))
    L, W = np.array(L), np.array(W)
    k = (W > 0).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        lbar = (W * L).sum(0) / W.sum(0)
        q = (W * (L - lbar) ** 2).sum(0) / (k - 1)
    q[k < 2] = np.nan
    return q, L, k


def particoes(df, cob, descs, alvo, suporte, rng):
    nomes, n_dc, s_dc, n_c, s_c = estatisticas_por_circuito(df, cob, alvo)
    perfil = df.drop_duplicates("circuito").set_index("circuito")["d_perfil"].reindex(nomes).to_numpy()
    tamanhos = pd.Series(perfil).value_counts()
    testados = sorted(g for g in tamanhos.index if g != URBANO)

    q_real, L_real, k_real = heterogeneidade(perfil, testados, n_dc, s_dc, n_c, s_c, suporte)
    validas = ~np.isnan(q_real)
    obs_media, obs_max = np.nanmean(q_real), np.nanmax(q_real)

    # partições aleatórias com os mesmos tamanhos; o grupo de 2 circuitos fica de fora
    rotulos_base = np.concatenate([[g] * n for g, n in tamanhos.items()])
    medias, maximos = np.empty(PARTICOES), np.empty(PARTICOES)
    for i in range(PARTICOES):
        r = rng.permutation(rotulos_base)
        q, _, _ = heterogeneidade(r, testados, n_dc, s_dc, n_c, s_c, suporte)
        medias[i], maximos[i] = np.nanmean(q), np.nanmax(q)
    p_media = (np.sum(medias >= obs_media) + 1) / (PARTICOES + 1)
    p_max = (np.sum(maximos >= obs_max) + 1) / (PARTICOES + 1)
    limiar_desc = float(np.quantile(maximos, 0.95))

    linhas = []
    for k in np.flatnonzero(validas):
        linha = {"descricao": texto(descs[k]), "Q_por_gl": q_real[k],
                 "clusters_com_suporte": int(k_real[k]),
                 "depende_do_cluster": q_real[k] > limiar_desc}
        for j, g in enumerate(testados):
            linha[f"efeito_{g}"] = L_real[j, k] if n_dc[k, perfil == g].sum() >= suporte else np.nan
        linhas.append(linha)
    tab = pd.DataFrame(linhas).sort_values("Q_por_gl", ascending=False)
    tab.attrs.update(obs_media=obs_media, obs_max=obs_max, p_media=p_media, p_max=p_max,
                     limiar_desc=limiar_desc, ref_media=np.quantile(medias, [0.5, 0.95]),
                     ref_max=np.quantile(maximos, [0.5, 0.95]), testados=testados,
                     n_descricoes=int(validas.sum()))
    return tab


# ---------------------------------------------------------------------------
def roda(nome, df, cfg, rng):
    alvo, suporte, descritores = cfg["alvo"], cfg["suporte"], cfg["descritores"]
    df = df.sort_values(["circuito", "ano", "round"], kind="stable").reset_index(drop=True)
    descs = descricoes(df, descritores)
    print(f"Base {nome}: {len(df)} linhas, {df.groupby(['ano', 'round']).ngroups} corridas, "
          f"{df['circuito'].nunique()} circuitos; taxa geral {df[alvo].mean():.4f}; "
          f"{len(descs)} descrições de profundidade 1-2")
    PASTA.mkdir(parents=True, exist_ok=True)

    titulo(f"[{nome}] BUSCA GLOBAL (mesma configuração das buscas por cluster)")
    cob = cobertura(df, descs)
    glob = busca(df, cob, descs, alvo, suporte, rng)
    print(f"Elegíveis {len(glob)}, limiar |z| {glob.attrs['limiar']:.2f}, "
          f"firmes {int(glob['firme'].sum())}")
    del cob

    resultados = {"global": glob}
    perfis = df["d_perfil"].value_counts()
    titulo(f"[{nome}] BUSCA DENTRO DE CADA CLUSTER")
    for perfil in perfis.index:
        parte = df[df["d_perfil"] == perfil].reset_index(drop=True)
        print(f"\n--- {perfil}: {len(parte)} linhas, {parte['circuito'].nunique()} circuitos, "
              f"taxa {parte[alvo].mean():.4f}")
        if perfil == URBANO:
            fg = glob[glob["firme"]].head(10)
            cobp = cobertura(parte, list(fg["chave"]))
            y = parte[alvo].to_numpy(dtype=float)
            desc = pd.DataFrame({"descricao": fg["descricao"].to_numpy(),
                                 "efeito_global": fg["efeito"].to_numpy(),
                                 "linhas_urbano": cobp.sum(1),
                                 "efeito_urbano": [y[m].mean() - y.mean() if m.any() else np.nan
                                                   for m in cobp]})
            print("Só descritivo (sem teste). Os 10 maiores firmes do global, vistos aqui:")
            print(desc.to_string(index=False, float_format=lambda x: f"{x:+.4f}", max_colwidth=60))
            desc.to_csv(PASTA / f"{nome}_urbano_descritivo.csv", index=False)
            continue
        cobp = cobertura(parte, descs)
        res = busca(parte, cobp, descs, alvo, suporte, rng)
        del cobp
        if res.empty:
            print("Nenhuma descrição elegível.")
            continue
        print(f"Elegíveis {len(res)}, limiar |z| {res.attrs['limiar']:.2f}, "
              f"firmes {int(res['firme'].sum())}")
        resultados[perfil] = res

    titulo(f"[{nome}] VALIDAÇÃO 1: LOCAL × GLOBAL")
    gfirme = {d: np.sign(e) for d, e, f in glob[["descricao", "efeito", "firme"]].itertuples(index=False) if f}
    gefeito = glob.set_index("descricao")["efeito"]
    linhas = []
    for perfil, res in resultados.items():
        if perfil == "global":
            continue
        f = res[res["firme"]]
        mesmo = sum(gfirme.get(d) == np.sign(e) for d, e in zip(f["descricao"], f["efeito"]))
        oposto = sum(d in gfirme and gfirme[d] != np.sign(e) for d, e in zip(f["descricao"], f["efeito"]))
        comum = res.set_index("descricao")["efeito"].to_frame("local").join(gefeito.rename("global"), how="inner")
        rho = comum["local"].corr(comum["global"], method="spearman")
        linhas.append({"cluster": perfil, "firmes_locais": len(f), "tambem_firmes_no_global": mesmo,
                       "sentido_oposto_ao_global": oposto, "so_locais": len(f) - mesmo - oposto,
                       "firmes_globais_nao_achados": len(gfirme) - mesmo,
                       "descricoes_em_comum": len(comum), "spearman_efeito": rho})
        so_locais = f[~f["descricao"].isin(gfirme)]
        if len(so_locais):
            print(f"\n{perfil}: firmes só locais (até 10)")
            print(so_locais.head(10)[["descricao", "linhas", "circuitos", "media", "efeito", "z"]]
                  .to_string(index=False, float_format=lambda x: f"{x:+.4f}", max_colwidth=60))
    comp = pd.DataFrame(linhas)
    print("\n" + comp.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    todos = []
    for perfil, res in resultados.items():
        todos.append(res.drop(columns="chave").assign(busca=perfil))
    pd.concat(todos).to_csv(PASTA / f"{nome}_subgrupos.csv", index=False)
    comp.to_csv(PASTA / f"{nome}_local_x_global.csv", index=False)

    titulo(f"[{nome}] VALIDAÇÃO 2: PERMUTAÇÃO DE PARTIÇÕES ({PARTICOES} partições 2/5/8/11)")
    cob = cobertura(df, descs)
    het = particoes(df, cob, descs, alvo, suporte, rng)
    del cob
    a = het.attrs
    print(f"Clusters testados: {', '.join(a['testados'])}; descrições com suporte em 2+: "
          f"{a['n_descricoes']}")
    print(f"Heterogeneidade média (Q/gl): real {a['obs_media']:.2f}; partições aleatórias "
          f"mediana {a['ref_media'][0]:.2f}, 95% {a['ref_media'][1]:.2f}; p = {a['p_media']:.4f}")
    print(f"Heterogeneidade máxima (Q/gl): real {a['obs_max']:.2f}; partições aleatórias "
          f"mediana {a['ref_max'][0]:.2f}, 95% {a['ref_max'][1]:.2f}; p = {a['p_max']:.4f}")
    dep = het[het["depende_do_cluster"]]
    print(f"Descrições cujo efeito depende do cluster (Q/gl > {a['limiar_desc']:.2f}, corrigido): "
          f"{len(dep)}")
    print("\nMaiores Q/gl:")
    print(het.head(12).to_string(index=False, float_format=lambda x: f"{x:+.4f}", max_colwidth=55))
    het.to_csv(PASTA / f"{nome}_heterogeneidade.csv", index=False)
    return comp, het


def sintetico(rng, efeito_cluster):
    """Base parecida com a de parada: 26 circuitos com perfis reais, 6 corridas cada,
    20 linhas por corrida-volta. Diferença real entre circuitos sempre presente; o
    efeito de cluster (se houver) faz o descritor A=1 valer mais no stop-and-go."""
    cl = pd.read_csv("data/processed/clusters_circuitos.csv")
    mapa = df_perfis(cl)
    linhas = []
    for r, (circuito, perfil) in enumerate(mapa.items()):
        base_c = rng.normal(-3.0, 0.4)                           # cada circuito tem sua taxa
        efeito_a_c = rng.normal(0.8, 0.25)                       # e sua própria força do efeito A
        if efeito_cluster and perfil == "stop-and-go":
            efeito_a_c += 0.9
        for ano in range(2018, 2024):
            n = 600
            a = rng.integers(0, 2, n)
            b = rng.integers(0, 3, n)
            c = rng.integers(0, 2, n)
            lp = base_c + efeito_a_c * a + 0.3 * (b == 2) + rng.normal(0, 0.3)
            y = rng.random(n) < 1 / (1 + np.exp(-lp))
            linhas.append(pd.DataFrame({"ano": ano, "round": r + 1, "circuito": circuito,
                                        "d_a": a.astype(str), "d_b": b.astype(str), "d_c": c.astype(str),
                                        "y": y.astype(int), "d_perfil": perfil,
                                        "d_era": "2018-21" if ano <= 2021 else "2022+"}))
    return pd.concat(linhas, ignore_index=True)


def df_perfis(cl):
    col = [c for c in cl.columns if c != "circuito"][0]
    if cl[col].dtype == object:
        return cl.set_index("circuito")[col].to_dict()
    nomes = pd.read_csv("data/processed/base_decisao_parada.csv",
                        usecols=["circuito", "d_perfil"]).drop_duplicates("circuito")
    return nomes.set_index("circuito")["d_perfil"].to_dict()


def main():
    global PERMUTACOES, PARTICOES
    ap = argparse.ArgumentParser()
    ap.add_argument("--sintetico", action="store_true")
    ap.add_argument("--base", choices=list(BASES))
    args = ap.parse_args()
    rng = np.random.default_rng(SEMENTE)
    if args.sintetico:
        global PASTA
        PASTA = Path("data/processed/sd_por_cluster/sintetico")
        PERMUTACOES, PARTICOES = 200, 1000
        cfg = dict(alvo="y", suporte=100, descritores=["d_a", "d_b", "d_c"])
        for efeito in (False, True):
            titulo(f"SINTÉTICO, efeito de cluster = {efeito}")
            roda(f"sintetico_{int(efeito)}", sintetico(rng, efeito), cfg, rng)
        return
    for nome, cfg in BASES.items():
        if args.base and nome != args.base:
            continue
        df = pd.read_csv(cfg["arquivo"])
        roda(nome, df, cfg, rng)


if __name__ == "__main__":
    main()
