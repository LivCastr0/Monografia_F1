"""
Etapa 2 - Piloto da degradação por setor (estratégia por segmentos do circuito).

Pergunta do piloto: a perda de tempo com a idade do pneu se distribui de forma
diferente entre os três setores oficiais, a ponto de ser medida acima do ruído?
Se não, a abordagem por segmentos não se sustenta e não vale montar a base completa.

Desenho (decidido antes de rodar):
  - Cronometragem: setores oficiais da CORRIDA (Sector1/2/3Time, laços de
    cronometragem, precisão de milésimos).
  - Caracterização: telemetria da CLASSIFICAÇÃO do mesmo fim de semana (o cache
    da Etapa 1). Os limites de setor saem de Sector1/2SessionTime cruzados com a
    telemetria; as features por setor usam a mesma receita da Etapa 1.
  - Medida: inclinação Theil-Sen do tempo de setor contra a vida do pneu, por
    stint x setor, em % do tempo típico do setor. Mesmos filtros da análise de
    degradação por volta (`monta_base_stint.py`), com o tráfego medido na
    entrada de cada setor, não só na linha de chegada.
  - Combustível: suposição externa (0,03 s/kg; sensibilidade 0,02 e 0,04),
    repartida entre setores de duas formas (proporcional ao tempo do setor ou ao
    tempo de pé embaixo). Um modelo com efeitos fixos de piloto, que separa vida
    do pneu e número da volta usando pilotos em estratégias diferentes, serve de
    checagem independente dessa repartição.

Critérios de decisão (fixados antes de rodar):
  C1 cobertura   : >= 85% das voltas limpas têm os três tempos de setor
  C2 fronteiras  : desvio-padrão do limite de setor entre pilotos <= 1% da volta
  C3 sinal       : perfil de degradação entre setores acima do acaso
                   (permutação dentro do stint, p < 0,05) em >= 3 de 4 corridas
  C4 combustível : perfil entre setores estável ao variar K (0,02 e 0,04 s/kg)
                   na repartição principal (Spearman mínimo >= 0,80)
  C5 repartição  : se o perfil muda entre repartição por tempo e por pé
                   embaixo (Spearman < 0,80), o modelo cruzado decide qual
                   delas os dados sustentam; passa se apontar a mesma em
                   >= 3 de 4 corridas
  D6 (diagnóstico, não decide): correlação vida x volta no modelo cruzado

Rodar da raiz do repositório:  python piloto_setores.py
"""
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.stats import theilslopes, spearmanr

warnings.filterwarnings("ignore", category=FutureWarning)

# --------------------------------------------------------------------------
# Configuração
# --------------------------------------------------------------------------
CACHE = "data/cache"
SAIDA = Path("data/processed/piloto_setores")
ARQ_CLUSTERS = "data/processed/clusters_circuitos.csv"
ARQ_FEATURES = "data/processed/features_completas_por_circuito.csv"
ARQ_COMPRIMENTO = "data/external/circuit_length.csv"

# Uma corrida seca por perfil, com estratégias variadas (necessário para o C5).
CORRIDAS = [
    (2024, 1, "Sakhir"),        # stop-and-go
    (2024, 10, "Barcelona"),    # alto downforce / técnico misto
    (2024, 16, "Monza"),        # power circuits
    (2024, 18, "Singapore"),    # urbano-técnico
]

COL_CLUSTER = "cluster_kmeans_so_dinamicas_k4"
NOMES_CLUSTER = {0: "Alto downforce/técnico misto", 1: "Stop-and-go",
                 2: "Power circuits", 3: "Urbano-técnico"}

# Filtros (iguais aos da degradação por volta)
GAP_MIN_S = 1.5             # distância mínima ao carro da frente, na entrada do setor
VOLTAS_DESCARTE_STINT = 2   # primeiras voltas de cada stint
MIN_VOLTAS_LIMPAS = 8
MIN_SPAN_VIDA = 6
TETO_TEMPO_SETOR = 1.07     # descarta setor > 107% da mediana limpa (erro, bandeira local)

# Combustível
K_COMB = [0.02, 0.03, 0.04]          # s por kg
REPARTICOES = ["tempo", "acel"]
VERSAO_PRINCIPAL = "K0.03_tempo"

# Critérios
LIM_C1, LIM_C2, LIM_C3_P, LIM_C3_N, LIM_C4 = 0.85, 0.01, 0.05, 3, 0.80
N_PERM, N_BOOT, SEMENTE = 2000, 2000, 42

SETORES = [1, 2, 3]


def combustivel_inicial(ano):
    return 105.0 if ano <= 2018 else 110.0


# --------------------------------------------------------------------------
# Parte 1 - Setores pela telemetria da classificação
# --------------------------------------------------------------------------
def _s(td):
    """Timedelta (escalar ou série) -> segundos float."""
    if isinstance(td, pd.Series):
        return td.dt.total_seconds()
    return np.nan if pd.isna(td) else td.total_seconds()


def features_setores_da_volta(car, t1, t2):
    """
    Features da Etapa 1 calculadas dentro de cada setor de UMA volta.
    `car` precisa de SessionTime, Speed, Throttle, Brake e Distance.
    A aceleração é calculada na volta inteira e só depois recortada, para o
    corte de setor não criar derivada espúria na fronteira.
    """
    st = _s(car["SessionTime"]).to_numpy()
    speed = car["Speed"].to_numpy(dtype=float)
    throttle = car["Throttle"].to_numpy(dtype=float)
    throttle = np.where(throttle > 100, np.nan, throttle)
    brake = np.asarray(car["Brake"].to_numpy(), dtype=bool)
    dist = car["Distance"].to_numpy(dtype=float)

    dt = np.diff(st)
    dv = np.diff(speed) * (1000 / 3600)
    with np.errstate(divide="ignore", invalid="ignore"):
        acc = dv / dt
    acc = np.r_[np.nan, acc]
    acc[~np.isfinite(acc)] = np.nan
    acc = pd.Series(acc).rolling(5, center=True, min_periods=1).median().to_numpy()
    inicio_freio = np.r_[False, (~brake[:-1]) & brake[1:]]

    d1, d2 = np.interp(t1, st, dist), np.interp(t2, st, dist)
    limites = {1: (dist[0], d1), 2: (d1, d2), 3: (d2, dist[-1])}
    mascaras = {1: st < t1, 2: (st >= t1) & (st < t2), 3: st >= t2}

    linhas = []
    for s in SETORES:
        m = mascaras[s]
        if m.sum() < 10:
            return None
        comp = limites[s][1] - limites[s][0]
        a = acc[m]
        linhas.append({
            "setor": s,
            "inicio_m": limites[s][0], "fim_m": limites[s][1], "comprimento_m": comp,
            "vel_media": np.nanmean(speed[m]),
            "vel_minima": np.nanpercentile(speed[m], 1),
            "pct_full_throttle": np.nanmean(throttle[m] >= 99) * 100,
            "pct_freando": np.mean(brake[m]) * 100,
            "zonas_freio": int(inicio_freio[m].sum()),
            "pico_desac": np.nanpercentile(a, 1) if np.isfinite(a).any() else np.nan,
        })
    return linhas


def setores_da_classificacao(q):
    """Mediana entre pilotos das features por setor + estabilidade dos limites."""
    linhas = []
    for drv in q.laps["Driver"].unique():
        try:
            volta = q.laps[q.laps["Driver"] == drv].pick_fastest()
            if volta is None or pd.isna(volta["Sector1SessionTime"]) \
                    or pd.isna(volta["Sector2SessionTime"]):
                continue
            car = volta.get_car_data().add_distance()
            res = features_setores_da_volta(
                car, _s(volta["Sector1SessionTime"]), _s(volta["Sector2SessionTime"]))
        except Exception:
            continue
        if res:
            for r in res:
                r["piloto"] = drv
            linhas.extend(res)
    por_piloto = pd.DataFrame(linhas)
    if por_piloto.empty:
        raise RuntimeError("nenhuma volta de classificação com setores completos")

    comp_volta = por_piloto.groupby("piloto")["comprimento_m"].sum().median()
    fronteiras = por_piloto[por_piloto["setor"] < 3].groupby("setor")["fim_m"].std()
    setores = por_piloto.groupby("setor").median(numeric_only=True)
    setores["zonas_freio_km"] = setores["zonas_freio"] / (setores["comprimento_m"] / 1000)
    setores["n_pilotos"] = por_piloto.groupby("setor")["piloto"].nunique()

    # Curvas por setor: get_circuit_info mede distância na volta mais rápida
    # da sessão, então os limites usados aqui saem dessa mesma volta.
    try:
        ref = q.laps.pick_fastest()
        car_ref = ref.get_car_data().add_distance()
        st = _s(car_ref["SessionTime"]).to_numpy()
        d1 = np.interp(_s(ref["Sector1SessionTime"]), st, car_ref["Distance"])
        d2 = np.interp(_s(ref["Sector2SessionTime"]), st, car_ref["Distance"])
        dc = q.get_circuit_info().corners["Distance"].to_numpy()
        n = {1: (dc < d1).sum(), 2: ((dc >= d1) & (dc < d2)).sum(), 3: (dc >= d2).sum()}
        setores["n_curvas"] = pd.Series(n)
        setores["densidade_curvas"] = setores["n_curvas"] / (setores["comprimento_m"] / 1000)
    except Exception as e:
        print(f"    curvas indisponíveis: {type(e).__name__}")
        setores["n_curvas"] = np.nan
        setores["densidade_curvas"] = np.nan

    c2 = float((fronteiras / comp_volta).max())
    return setores.reset_index(), c2, comp_volta


# --------------------------------------------------------------------------
# Parte 2 - Voltas de corrida, filtros e inclinações por stint x setor
# --------------------------------------------------------------------------
def gaps_por_linha(laps):
    """
    Distância (s) ao carro que cruzou a mesma linha imediatamente antes,
    em cada uma das três entradas de setor. Conta carros retardatários também:
    o que importa é proximidade física, não posição.
    """
    linhas = {1: "LapStartTime", 2: "Sector1SessionTime", 3: "Sector2SessionTime"}
    saida = pd.DataFrame(index=laps.index)
    for s, col in linhas.items():
        cruz = laps[[col]].copy()
        if s == 1:  # LapStartTime da volta de saída é cruzamento no pit lane
            cruz.loc[laps["PitOutTime"].notna(), col] = pd.NaT
        cruz = cruz.dropna().sort_values(col)
        gap = _s(cruz[col]).diff()
        saida[f"gap_s{s}"] = gap.reindex(laps.index)
    return saida


def voltas_limpas(laps):
    """Filtros de volta inteira + filtros por setor. Devolve formato longo."""
    d = laps.copy()
    d = d.join(gaps_por_linha(d))
    d["pos_stint"] = d.groupby(["Driver", "Stint"])["LapNumber"].rank(method="first")

    base = (
        (d["IsAccurate"] == True)
        & (d["TrackStatus"].astype(str) == "1")
        & d["PitInTime"].isna() & d["PitOutTime"].isna()
        & (d["LapNumber"] > 1)
        & (d["pos_stint"] > VOLTAS_DESCARTE_STINT)
        & d["Compound"].isin(["SOFT", "MEDIUM", "HARD"])
        & d["TyreLife"].notna()
    )
    d = d[base].copy()
    cobertura = float(d[[f"Sector{s}Time" for s in SETORES]].notna().all(axis=1).mean()) \
        if len(d) else np.nan

    longo = []
    for s in SETORES:
        x = d[["Driver", "Team", "Stint", "Compound", "TyreLife", "LapNumber"]].copy()
        x["setor"] = s
        x["tempo"] = _s(d[f"Sector{s}Time"])
        x["gap"] = d[f"gap_s{s}"]
        longo.append(x)
    longo = pd.concat(longo, ignore_index=True).dropna(subset=["tempo"])
    longo = longo[longo["gap"].fillna(np.inf) >= GAP_MIN_S]
    teto = longo.groupby("setor")["tempo"].transform("median") * TETO_TEMPO_SETOR
    longo = longo[longo["tempo"] <= teto]
    return longo, cobertura, len(d)


def reparticao_combustivel(setores, longo):
    """Fração do efeito do peso atribuída a cada setor, nas duas versões."""
    tempo_tip = longo.groupby("setor")["tempo"].median()
    r_tempo = tempo_tip / tempo_tip.sum()
    pe = tempo_tip * setores.set_index("setor")["pct_full_throttle"]
    r_acel = pe / pe.sum()
    return {"tempo": r_tempo, "acel": r_acel}


def inclinacoes(longo, reparticoes, n_voltas, ano):
    """Theil-Sen por stint x setor, em % do tempo típico, para as 6 variantes."""
    comb0 = combustivel_inicial(ano)
    longo = longo.copy()
    longo["comb_kg"] = comb0 * (1 - (longo["LapNumber"] - 1) / n_voltas)

    linhas = []
    for (drv, stint, s), g in longo.groupby(["Driver", "Stint", "setor"]):
        if len(g) < MIN_VOLTAS_LIMPAS or np.ptp(g["TyreLife"]) < MIN_SPAN_VIDA:
            continue
        x = g["TyreLife"].to_numpy(float)
        ref = g["tempo"].median()
        r = {"piloto": drv, "equipe": g["Team"].iloc[0], "stint": int(stint),
             "setor": s, "composto": g["Compound"].iloc[0], "n_voltas": len(g),
             "vida_ini": x.min(), "vida_fim": x.max(), "tempo_tipico": ref}
        for K in K_COMB:
            for rep, frac in reparticoes.items():
                y = g["tempo"].to_numpy() - K * g["comb_kg"].to_numpy() * frac[s]
                r[f"K{K:.2f}_{rep}"] = theilslopes(y, x)[0] / ref * 100
        linhas.append(r)
    return pd.DataFrame(linhas)


# --------------------------------------------------------------------------
# Parte 3 - Critérios
# --------------------------------------------------------------------------
def matriz_stints(incl, col):
    """Stints com os três setores válidos -> matriz stints x setores."""
    w = incl.pivot_table(index=["piloto", "stint"], columns="setor", values=col)
    return w.dropna()


def teste_sinal_setores(incl, col, rng):
    """
    C3. Tira a média de cada stint (o nível de degradação do stint) e mede
    quanto da variação restante é explicada pelo setor. Nulo: embaralhar os
    rótulos de setor dentro de cada stint.
    """
    X = matriz_stints(incl, col).to_numpy()
    if len(X) < 5:
        return np.nan, np.nan, len(X)
    Xc = X - X.mean(axis=1, keepdims=True)

    def estat(M):
        return len(M) * (M.mean(axis=0) ** 2).sum() / (M ** 2).sum()

    obs = estat(Xc)
    nulos = np.array([estat(rng.permuted(Xc, axis=1)) for _ in range(N_PERM)])
    p = (1 + (nulos >= obs).sum()) / (N_PERM + 1)
    return obs, p, len(X)


def ic_bootstrap(incl, col, rng):
    w = matriz_stints(incl, col)
    X = w.to_numpy()
    if len(X) < 5:
        return pd.DataFrame()
    idx = rng.integers(0, len(X), (N_BOOT, len(X)))
    medias = X[idx].mean(axis=1)
    return pd.DataFrame({"setor": w.columns, "media": X.mean(axis=0),
                         "ic_lo": np.percentile(medias, 2.5, axis=0),
                         "ic_hi": np.percentile(medias, 97.5, axis=0),
                         "n_stints": len(X)})


def estabilidade_combustivel(incl_todas):
    """C4. Perfil centrado por corrida (12 valores) sob cada variante."""
    variantes = [f"K{K:.2f}_{r}" for K in K_COMB for r in REPARTICOES]
    perfis = {}
    for v in variantes:
        m = incl_todas.groupby(["circuito", "setor"])[v].median()
        perfis[v] = m - m.groupby(level=0).transform("mean")
    principal = perfis[VERSAO_PRINCIPAL]
    rhos = {v: spearmanr(principal, perfis[v])[0] for v in variantes if v != VERSAO_PRINCIPAL}
    rep_princ = VERSAO_PRINCIPAL.split("_")[1]
    rho_k = min(r for v, r in rhos.items() if v.endswith(rep_princ))
    rho_rep = min(r for v, r in rhos.items() if not v.endswith(rep_princ))
    return rho_k, rho_rep, rhos


def modelo_cruzado(longo, rep):
    """
    C5/D6. Por setor: tempo ~ vida + volta + efeitos fixos de piloto e composto.
    Vida e volta só se separam porque pilotos param em voltas diferentes.
    `b_volta` junta combustível e evolução da pista; a fração dele em cada
    setor testa a repartição proporcional ao tempo.
    """
    out = []
    for s, g in longo.groupby("setor"):
        D = pd.get_dummies(g[["Driver", "Compound"]], drop_first=True, dtype=float)
        X = np.column_stack([np.ones(len(g)), g["TyreLife"], g["LapNumber"], D.to_numpy()])
        b, *_ = np.linalg.lstsq(X, g["tempo"].to_numpy(), rcond=None)
        ref = g["tempo"].median()
        out.append({"setor": s, "n": len(g), "b_vida_pct": b[1] / ref * 100,
                    "b_volta_s": b[2], "tempo_tipico": ref,
                    "corr_vida_volta": np.corrcoef(g["TyreLife"], g["LapNumber"])[0, 1]})
    out = pd.DataFrame(out)
    out["frac_volta"] = out["b_volta_s"] / out["b_volta_s"].sum()
    out["frac_tempo"] = out["setor"].map(rep["tempo"])
    out["frac_acel"] = out["setor"].map(rep["acel"])
    return out


def reparticao_sustentada(mod):
    """Qual repartição fica mais perto do que o modelo cruzado estimou."""
    e_t = (mod["frac_volta"] - mod["frac_tempo"]).abs().sum()
    e_a = (mod["frac_volta"] - mod["frac_acel"]).abs().sum()
    return "tempo" if e_t <= e_a else "acel"


# --------------------------------------------------------------------------
# Parte 4 - Projeção dos setores nos centroides da Etapa 1
# --------------------------------------------------------------------------
FEATS_PROJ = ["vel_media", "vel_minima", "zonas_freio_km", "pico_desac", "densidade_curvas"]


def achar(caminho):
    """Usa o caminho configurado; se não existir, procura o arquivo pelo nome no repositório."""
    p = Path(caminho)
    if p.exists():
        return p
    achados = sorted(x for x in Path(".").rglob(p.name)
                     if ".venv" not in x.parts and "cache" not in x.parts)
    if not achados:
        raise FileNotFoundError(
            f"{p.name} não encontrado em {caminho} nem em nenhuma pasta do repositório; "
            f"ajuste a constante no topo do script")
    if len(achados) > 1:
        print(f"    aviso: {p.name} existe em {len(achados)} lugares; usando {achados[0]}")
    else:
        print(f"    {p.name} encontrado em {achados[0]}")
    return achados[0]


def espaco_etapa1():
    feats = pd.read_csv(achar(ARQ_FEATURES))
    comp = pd.read_csv(achar(ARQ_COMPRIMENTO), sep=";", decimal=",", encoding="utf-8-sig")
    comp.columns = ["circuito", "comprimento_m"]
    cl = pd.read_csv(achar(ARQ_CLUSTERS))[["circuito", COL_CLUSTER]]
    f = feats.merge(comp, on="circuito").merge(cl, on="circuito")
    if len(f) != 26:
        print(f"    aviso: {len(f)} circuitos casaram entre os três arquivos (esperado 26)")
    # contagem por volta -> por km, para comparar com setores de comprimentos diferentes
    f["zonas_freio_km"] = f["zonas_freio"] / (f["comprimento_m"] / 1000)
    mu, sd = f[FEATS_PROJ].mean(), f[FEATS_PROJ].std()
    z = (f[FEATS_PROJ] - mu) / sd
    centroides = z.groupby(f[COL_CLUSTER]).mean()
    return mu, sd, centroides, cl


def projetar(setores, mu, sd, centroides):
    z = (setores[FEATS_PROJ] - mu) / sd
    d = np.sqrt(((z.to_numpy()[:, None, :] - centroides.to_numpy()[None]) ** 2).sum(axis=2))
    ordem = np.argsort(d, axis=1)
    out = setores[["circuito", "setor"]].copy()
    out["cluster_mais_proximo"] = centroides.index[ordem[:, 0]]
    out["perfil_mais_proximo"] = out["cluster_mais_proximo"].map(NOMES_CLUSTER)
    out["distancia"] = d[np.arange(len(d)), ordem[:, 0]]
    out["margem_2o"] = d[np.arange(len(d)), ordem[:, 1]] - out["distancia"]
    return out


# --------------------------------------------------------------------------
# Execução
# --------------------------------------------------------------------------
def carregar(ano, rnd, tipo, **kw):
    import fastf1
    ses = fastf1.get_session(ano, rnd, tipo)
    ses.load(messages=False, **kw)
    return ses


def grafico(ics, caminho):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    circs = ics["circuito"].unique()
    fig, axs = plt.subplots(1, len(circs), figsize=(3.2 * len(circs), 3.4), sharey=True)
    for ax, c in zip(np.atleast_1d(axs), circs):
        g = ics[ics["circuito"] == c]
        ax.errorbar(g["setor"], g["media"],
                    yerr=[g["media"] - g["ic_lo"], g["ic_hi"] - g["media"]],
                    fmt="o", color="#1f5f8b", capsize=4)
        ax.axhline(0, color="#999", lw=0.8)
        ax.set_xticks(SETORES, [f"S{s}" for s in SETORES])
        ax.set_title(f"{c} (n={int(g['n_stints'].iloc[0])})", fontsize=10)
    np.atleast_1d(axs)[0].set_ylabel("Degradação (% do tempo do setor por volta)")
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)


def main():
    import fastf1
    Path(CACHE).mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(CACHE)
    SAIDA.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEMENTE)

    mu, sd, centroides, cl = espaco_etapa1()
    todas_setores, todas_incl, todos_modelos, ics, criterios = [], [], [], [], []

    for ano, rnd, circ in CORRIDAS:
        print(f"\n=== {circ} {ano} (r{rnd}) ===")
        q = carregar(ano, rnd, "Q", laps=True, telemetry=True, weather=False)
        setores, c2, comp_volta = setores_da_classificacao(q)
        setores.insert(0, "circuito", circ)
        print(f"    volta de referência {comp_volta:.0f} m; limites de setor "
              f"variam até {c2:.2%} da volta entre pilotos")

        r = carregar(ano, rnd, "R", laps=True, telemetry=False, weather=True)
        chuva = float(r.weather_data["Rainfall"].mean()) if r.weather_data is not None else 0.0
        if chuva > 0:
            print(f"    AVISO: chuva em {chuva:.0%} das amostras de clima; trocar a corrida")
        laps = pd.DataFrame(r.laps)
        n_voltas = int(laps["LapNumber"].max())
        longo, c1, n_base = voltas_limpas(laps)
        print(f"    {n_base} voltas limpas; {c1:.1%} com os três setores")

        rep = reparticao_combustivel(setores, longo)
        incl = inclinacoes(longo, rep, n_voltas, ano)
        incl.insert(0, "circuito", circ)
        stat, p, n_st = teste_sinal_setores(incl, VERSAO_PRINCIPAL, rng)
        print(f"    {len(incl)} stint-setores; {n_st} stints com os três setores; "
              f"setor explica {stat:.1%} da variação dentro do stint (p = {p:.4f})")

        mod = modelo_cruzado(longo, rep)
        rep_dados = reparticao_sustentada(mod)
        mod.insert(0, "circuito", circ)
        ic = ic_bootstrap(incl, VERSAO_PRINCIPAL, rng)
        ic.insert(0, "circuito", circ)

        setores["frac_comb_tempo"] = setores["setor"].map(rep["tempo"])
        setores["frac_comb_acel"] = setores["setor"].map(rep["acel"])
        todas_setores.append(setores)
        todas_incl.append(incl)
        todos_modelos.append(mod)
        ics.append(ic)
        criterios.append({"circuito": circ, "ano": ano, "chuva": chuva,
                          "c1_cobertura": c1, "c2_fronteira": c2,
                          "c3_estat": stat, "c3_p": p, "c3_n_stints": n_st,
                          "n_stint_setores": len(incl), "reparticao_dados": rep_dados})

    setores = pd.concat(todas_setores, ignore_index=True)
    incl = pd.concat(todas_incl, ignore_index=True)
    modelos = pd.concat(todos_modelos, ignore_index=True)
    ics = pd.concat(ics, ignore_index=True)
    crit = pd.DataFrame(criterios)
    c4, c5_rho, rhos = estabilidade_combustivel(incl)
    proj = projetar(setores, mu, sd, centroides)

    setores.merge(proj, on=["circuito", "setor"]).to_csv(SAIDA / "setores_features.csv", index=False)
    incl.to_csv(SAIDA / "stints_setores.csv", index=False)
    modelos.to_csv(SAIDA / "modelo_cruzado.csv", index=False)
    ics.to_csv(SAIDA / "degradacao_por_setor_ic.csv", index=False)
    crit.to_csv(SAIDA / "criterios_por_corrida.csv", index=False)
    grafico(ics, SAIDA / "degradacao_por_setor.png")

    # ---------------- Resumo ----------------
    ok1 = (crit["c1_cobertura"] >= LIM_C1).all()
    ok2 = (crit["c2_fronteira"] <= LIM_C2).all()
    n3 = int((crit["c3_p"] < LIM_C3_P).sum())
    ok3 = n3 >= LIM_C3_N
    ok4 = c4 >= LIM_C4
    rep_princ = VERSAO_PRINCIPAL.split("_")[1]
    n5 = int((crit["reparticao_dados"] == rep_princ).sum())
    ok5 = c5_rho >= LIM_C4 or n5 >= LIM_C3_N
    rot = lambda ok: "PASSA" if ok else "FALHA"

    print("\n" + "=" * 70 + "\nRESUMO DO PILOTO\n" + "=" * 70)
    print(f"C1 cobertura   mín {crit['c1_cobertura'].min():.1%} (>= {LIM_C1:.0%})  {rot(ok1)}")
    print(f"C2 fronteiras  máx {crit['c2_fronteira'].max():.2%} (<= {LIM_C2:.0%})  {rot(ok2)}")
    print(f"C3 sinal       {n3} de {len(crit)} corridas com p < {LIM_C3_P} (>= {LIM_C3_N})  {rot(ok3)}")
    print(f"C4 combustível Spearman mín ao variar K {c4:.2f} (>= {LIM_C4})  {rot(ok4)}")
    print(f"C5 repartição  Spearman tempo x pé embaixo {c5_rho:.2f}; modelo cruzado "
          f"aponta '{rep_princ}' em {n5} de {len(crit)} corridas  {rot(ok5)}")
    for v, rho in rhos.items():
        print(f"      {v}: {rho:.2f}")

    print("\nDegradação média por setor (% por volta, IC 95% por bootstrap de stints):")
    print(ics.round(4).to_string(index=False))

    print("\nC5 - modelo cruzado (vida do pneu separada do número da volta):")
    print(modelos[["circuito", "setor", "n", "b_vida_pct", "frac_volta", "frac_tempo",
                   "frac_acel", "corr_vida_volta"]].round(3).to_string(index=False))
    print("  |corr_vida_volta| > 0,9 -> pouca variação de estratégia, C5 pouco confiável")

    print("\nSetores projetados nos centroides da Etapa 1:")
    print(proj.round(2).to_string(index=False))

    decisao = ok1 and ok2 and ok3 and ok4 and ok5
    print("\nDECISÃO:", "seguir para a base completa" if decisao else
          "não seguir sem rever o desenho (ver critério que falhou)")
    print(f"Arquivos em {SAIDA}/")


if __name__ == "__main__":
    main()
