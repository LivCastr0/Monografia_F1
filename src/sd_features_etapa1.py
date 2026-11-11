"""
Etapa 2 - Descoberta de subgrupos com as features da Etapa 1 como descritores.

A SD continua no centro. O cluster deixa de ser descritor; entram as cinco
features que definiram os perfis, em tercis entre os 26 circuitos (baixa, média,
alta): vel_media, vel_minima, zonas_freio, pico_desac, densidade_curvas. Cada
hipótese define um alvo; as condições de corrida das bases existentes entram como
os outros descritores.

Entra: data/processed/base_corrida.csv, base_duelos.csv, base_stint.csv
       data/processed/features_completas_por_circuito.csv
       data/interim/voltas/*_voltas.csv (só para montar a base de H5)
Sai:   data/processed/sd_features/<H>_subgrupos.csv, <H>_acrescenta.csv, resumo.txt
       data/processed/base_consistencia.csv (H5)
Uso:   python src/sd_features_etapa1.py              (as cinco hipóteses)
       python src/sd_features_etapa1.py --so H3      (uma)
       python src/sd_features_etapa1.py --sintetico  (teste do procedimento)

Hipóteses (fixadas antes de rodar; ver claude/hipoteses-features-continuas-msi2.md)
-----------------------------------------------------------------------------
  H1 valor da largada   alvo: Spearman largada x chegada da corrida (base_corrida,
                        uma linha por corrida). Previsto: vel_minima baixa -> maior.
  H2 de onde se passa   alvo: passou (base_duelos). Previsto: vel_media alta E
                        distância 0,3 a 0,6 s ou 0,6 a 1,0 s -> maior que a mesma
                        distância nas outras pistas.
  H3 trens de DRS       alvo: defensor em trem de DRS (base_duelos, sem d_trem como
                        descritor). Previsto: vel_media alta -> maior.
  H4 degradação         alvo: deg_pct por stint (base_stint). Previsto: zonas_freio
                        alta -> maior.
  H5 consistência       alvo: variação volta a volta do stint em % (base nova, abaixo).
                        Previsto: densidade_curvas alta -> maior.

Busca (igual para todas)
------------------------
  - Descrições de profundidade 1 e 2; suporte mínimo por hipótese (SUPORTE); 4+
    circuitos e nenhum circuito acima de 40% das linhas.
  - z = (média no subgrupo - média geral) * sqrt(n) / desvio-padrão.
  - Limiar: 95% do máximo |z| em 500 permutações. Em cada uma, o alvo é embaralhado
    dentro da corrida (entre corridas, em H1) e as features são trocadas entre os
    circuitos em bloco (o vetor de 5 features de um circuito vai inteiro para outro).
    A elegibilidade é refeita em cada permutação.
  - Firme: passa do limiar, mesmo sentido nas duas eras e sem o circuito mais frequente.

A pista acrescenta? (a referência certa)
----------------------------------------
Para cada subgrupo firme com uma condição de feature, e para os subgrupos previstos
nas hipóteses (firmes ou não), compara o subgrupo com as linhas que têm as MESMAS
condições de corrida mas NÃO a condição de feature (referência = a mesma situação nas
outras pistas). z_acr = diferença / (dp * sqrt(1/n_a + 1/n_b)). Limiar: 95% do
máximo |z_acr| em 2.000 trocas das features entre circuitos (o alvo fica parado).
Hipótese confirmada: o subgrupo previsto acrescenta, no sentido previsto, e é firme.

Verificação contínua (informativa): Spearman, nos 26 circuitos, entre a média do alvo
por circuito e a feature principal, p unilateral por permutação.

H5 - medida de consistência (fixada antes)
------------------------------------------
Voltas válidas e em ar limpo do piloto de ar sujo (volta 3+, verde, sem box, precisa,
até 107% da mediana, 3 s ou mais do carro da frente). Por stint, diferenças entre
voltas limpas consecutivas, em % da mediana do stint; alvo = 1,4826 * MAD / sqrt(2)
(ruído por volta). Stint entra com 6+ diferenças. Descritores: d_composto,
d_ordem_stint e d_temp_pista da base_stint. Qualidade (decide se H5 é interpretada):
Spearman, entre circuitos, da mediana do alvo em stints ímpares e pares >= 0,50.

Associação, não causa.
"""
import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))

PROC = Path("data/processed")
SAIDA = PROC / "sd_features"
FEATURES = ["vel_media", "vel_minima", "zonas_freio", "pico_desac", "densidade_curvas"]
PERMUTACOES, PERM_ACR, SEMENTE = 500, 2000, 2026
MIN_CIRCUITOS, MAX_CIRCUITO = 4, 0.40

HIPOTESES = {
    "H1": dict(base="corrida", alvo="rho_largada", suporte=15,
               condicoes=["d_chuva", "d_safety_car", "d_paradas"],
               previstos=[{"f_vel_minima": "baixa"}], sentido=+1, feature="vel_minima",
               sentido_cont=-1),
    "H2": dict(base="duelos", alvo="passou", suporte=100,
               condicoes=["d_dif_idade", "d_dif_composto", "d_distancia", "d_trem", "d_largada",
                          "d_fase", "d_relargada", "d_posicao", "d_mesma_equipe"],
               previstos=[{"f_vel_media": "alta", "d_distancia": "0,3 a 0,6 s"},
                          {"f_vel_media": "alta", "d_distancia": "0,6 a 1,0 s"}],
               sentido=+1, feature="vel_media", sentido_cont=+1),
    "H3": dict(base="duelos", alvo="trem", suporte=100,
               condicoes=["d_dif_idade", "d_dif_composto", "d_distancia", "d_largada",
                          "d_fase", "d_relargada", "d_posicao", "d_mesma_equipe"],
               previstos=[{"f_vel_media": "alta"}], sentido=+1, feature="vel_media",
               sentido_cont=+1),
    "H4": dict(base="stint", alvo="deg_pct", suporte=100,
               condicoes=["d_composto", "d_pneu_novo", "d_ordem_stint", "d_temp_pista"],
               previstos=[{"f_zonas_freio": "alta"}], sentido=+1, feature="zonas_freio",
               sentido_cont=+1),
    "H5": dict(base="consistencia", alvo="ruido_pct", suporte=100,
               condicoes=["d_composto", "d_ordem_stint", "d_temp_pista"],
               previstos=[{"f_densidade_curvas": "alta"}], sentido=+1, feature="densidade_curvas",
               sentido_cont=+1),
}


def titulo(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


# ---------------------------------------------------------------------------
def tercis():
    f = pd.read_csv(PROC / "features_completas_por_circuito.csv").set_index("circuito")[FEATURES]
    t = f.rank(method="first").apply(lambda s: pd.cut(s, 3, labels=["baixa", "média", "alta"]))
    return f, t.astype(str)


def base_consistencia():
    arq = PROC / "base_consistencia.csv"
    if arq.exists():
        return pd.read_csv(arq)
    import avalia_piloto_ar_sujo as A
    st = pd.read_csv(PROC / "base_stint.csv")
    linhas = []
    for (ano, rnd), _ in st.groupby(["ano", "round"]):
        v = A.prepara(pd.read_csv(Path("data/interim/voltas") / f"{ano}_{rnd:02d}_voltas.csv"))
        v = v[v["valida"] & (v["faixa"] == "limpo")].sort_values(["piloto", "stint", "volta"])
        for (p, s), g in v.groupby(["piloto", "stint"]):
            g = g[g["volta"].diff() == 1]
            ant = v[(v["piloto"] == p) & (v["stint"] == s)].set_index("volta")["tempo_volta_s"]
            d = [(t - ant.get(L - 1, np.nan)) for L, t in zip(g["volta"], g["tempo_volta_s"])]
            d = np.array(d, dtype=float)
            d = d[~np.isnan(d)]
            if len(d) < 6:
                continue
            med = ant.median()
            mad = np.median(np.abs(d - np.median(d)))
            linhas.append({"ano": ano, "round": rnd, "piloto": p, "stint": float(s),
                           "n_dif": len(d), "ruido_pct": 100 * 1.4826 * mad / np.sqrt(2) / med})
    c = pd.DataFrame(linhas)
    c = c.merge(st[["ano", "round", "piloto", "stint", "circuito", "d_composto", "d_ordem_stint",
                    "d_temp_pista", "d_era"]], on=["ano", "round", "piloto", "stint"], how="inner")
    c.to_csv(arq, index=False)
    return c


def carrega(nome):
    if nome == "corrida":
        return pd.read_csv(PROC / "base_corrida.csv")
    if nome == "duelos":
        d = pd.read_csv(PROC / "base_duelos.csv")
        d["trem"] = (d["d_trem"] == "defensor em trem de DRS").astype(int)
        return d
    if nome == "stint":
        return pd.read_csv(PROC / "base_stint.csv")
    return base_consistencia()


# ---------------------------------------------------------------------------
class Busca:
    """Descrições, cobertura e estatísticas com as features trocáveis entre circuitos."""

    def __init__(self, df, alvo, condicoes, suporte, t_feat):
        df = df.dropna(subset=[alvo]).sort_values(["circuito", "ano", "round"], kind="stable") \
            .reset_index(drop=True)
        self.df, self.alvo, self.suporte = df, alvo, suporte
        self.circs = sorted(df["circuito"].unique())
        self.ci = df["circuito"].map({c: i for i, c in enumerate(self.circs)}).to_numpy()
        self.inicios = np.flatnonzero(np.r_[True, self.ci[1:] != self.ci[:-1]])
        self.T = t_feat.reindex(self.circs)
        if self.T.isna().any().any():
            raise ValueError(f"circuito sem feature: {self.T[self.T.isna().any(axis=1)].index.tolist()}")
        self.cond = {c: df[c].astype(str).to_numpy() for c in condicoes}
        sel = [(c, v) for c in condicoes for v in sorted(df[c].dropna().astype(str).unique())]
        sel += [(f"f_{f}", v) for f in FEATURES for v in ("baixa", "média", "alta")]
        self.descs = [(s,) for s in sel] + [(a, b) for a, b in combinations(sel, 2) if a[0] != b[0]]
        self.y = df[alvo].to_numpy(dtype=float)
        self.corrida = pd.factorize(df["ano"].astype(str) + "_" + df["round"].astype(str))[0]
        self.era = df["d_era"].astype(str).to_numpy()

    def colunas(self, mapa=None):
        """mapa: permutação dos circuitos (índice -> circuito cujas features recebe)."""
        T = self.T.to_numpy() if mapa is None else self.T.to_numpy()[mapa]
        col = dict(self.cond)
        for j, f in enumerate(FEATURES):
            col[f"f_{f}"] = T[self.ci, j]
        return col

    def cobertura(self, col, descs=None):
        descs = self.descs if descs is None else descs
        cache = {}
        m = np.empty((len(descs), len(self.y)), dtype=bool)
        for k, d in enumerate(descs):
            r = np.ones(len(self.y), dtype=bool)
            for s in d:
                if s not in cache:
                    cache[s] = col[s[0]] == s[1]
                r &= cache[s]
            m[k] = r
        return m

    def z(self, cob, y):
        cont = np.add.reduceat(cob, self.inicios, axis=1, dtype=np.int32)
        n = cont.sum(1)
        ok = ((n >= self.suporte) & ((cont > 0).sum(1) >= MIN_CIRCUITOS)
              & (cont.max(1) <= MAX_CIRCUITO * np.maximum(n, 1)))
        yc = (y - y.mean()).astype(np.float64)
        z = np.zeros(len(n))
        z[ok] = (cob[ok].astype(np.float32) @ yc.astype(np.float32)) / np.sqrt(n[ok]) / y.std(ddof=1)
        return z, ok, n

    def embaralha_alvo(self, rng):
        if len(np.unique(self.corrida)) == len(self.y):          # uma linha por corrida (H1)
            return self.y[rng.permutation(len(self.y))]
        v = self.y.copy()                                          # dentro de cada corrida
        for idx in self._grupos():
            v[idx] = self.y[rng.permutation(idx)]
        return v

    def _grupos(self):
        if not hasattr(self, "_g"):
            self._g = list(pd.Series(np.arange(len(self.y))).groupby(self.corrida).indices.values())
        return self._g

    def roda(self, rng):
        col = self.colunas()
        cob = self.cobertura(col)
        zs, ok, n = self.z(cob, self.y)
        maximos = np.empty(PERMUTACOES)
        for i in range(PERMUTACOES):
            colp = self.colunas(rng.permutation(len(self.circs)))
            cobp = self.cobertura(colp)
            zp, okp, _ = self.z(cobp, self.embaralha_alvo(rng))
            maximos[i] = np.abs(zp[okp]).max() if okp.any() else 0.0
        limiar = float(np.quantile(maximos, 0.95))
        media = self.y.mean()
        linhas = []
        for k in np.flatnonzero(ok):
            m = cob[k]
            s = np.sign(self.y[m].mean() - media)
            conta = pd.Series(self.ci[m]).value_counts()
            sem = m & (self.ci != conta.index[0])
            eras = [self.y[m & (self.era == e)].mean() for e in np.unique(self.era) if (m & (self.era == e)).any()]
            eras_ok = len(eras) == 2 and all(np.sign(e - media) == s for e in eras)
            sem_ok = sem.any() and np.sign(self.y[sem].mean() - media) == s
            d = self.descs[k]
            linhas.append({"chave": d, "descricao": texto(d), "linhas": int(n[k]),
                           "circuitos": len(conta), "media": self.y[m].mean(),
                           "diferenca": self.y[m].mean() - media, "z": zs[k],
                           "acima_do_limiar": abs(zs[k]) > limiar,
                           "firme": bool(abs(zs[k]) > limiar and eras_ok and sem_ok),
                           "tem_feature": any(s_[0].startswith("f_") for s_ in d)})
        res = pd.DataFrame(linhas).sort_values("z", key=abs, ascending=False)
        res.attrs.update(limiar=limiar, media=media)
        return res

    def acrescenta(self, res, previstos, rng):
        """A pista acrescenta? Subgrupo contra as mesmas condições nas outras pistas."""
        alvo_d = [r["chave"] for _, r in res[res["firme"] & res["tem_feature"]].iterrows()]
        for p in previstos:
            d = tuple(sorted(p.items(), key=lambda kv: list(p).index(kv[0])))
            alvo_d.append(tuple((c, v) for c, v in p.items()))
        vistos, casos = set(), []
        for d in alvo_d:
            chave = tuple(sorted(d))
            if chave in vistos:
                continue
            vistos.add(chave)
            casos.append(d)
        sd = self.y.std(ddof=1)

        def est(col):
            z = np.full(len(casos), np.nan)
            info = []
            for k, d in enumerate(casos):
                feat = [s for s in d if s[0].startswith("f_")]
                cond = [s for s in d if not s[0].startswith("f_")]
                mc = np.ones(len(self.y), dtype=bool)
                for c, v in cond:
                    mc &= col[c] == v
                mf = np.ones(len(self.y), dtype=bool)
                for c, v in feat:
                    mf &= col[c] == v
                a, b = mc & mf, mc & ~mf
                if a.sum() >= 2 and b.sum() >= 2:
                    z[k] = (self.y[a].mean() - self.y[b].mean()) / (sd * np.sqrt(1 / a.sum() + 1 / b.sum()))
                info.append((int(a.sum()), self.y[a].mean() if a.any() else np.nan,
                             int(b.sum()), self.y[b].mean() if b.any() else np.nan,
                             len(np.unique(self.ci[a]))))
            return z, info

        obs, info = est(self.colunas())
        maximos = np.array([np.nanmax(np.abs(est(self.colunas(rng.permutation(len(self.circs))))[0]))
                            for _ in range(PERM_ACR)])
        limiar = float(np.quantile(maximos, 0.95))
        firmes = {tuple(sorted(r["chave"])): r["firme"] for _, r in res.iterrows()}
        linhas = []
        for k, d in enumerate(casos):
            na, ma, nb, mb, nc = info[k]
            linhas.append({"descricao": texto(d), "linhas": na, "circuitos": nc, "media": ma,
                           "referencia_outras_pistas": mb, "linhas_referencia": nb,
                           "diferenca_da_pista": ma - mb, "z_acr": obs[k],
                           "acrescenta": bool(abs(obs[k]) > limiar),
                           "firme_na_busca": bool(firmes.get(tuple(sorted(d)), False)),
                           "previsto": any(tuple(sorted(p.items())) == tuple(sorted(d)) for p in previstos)})
        t = pd.DataFrame(linhas).sort_values("z_acr", key=abs, ascending=False)
        t.attrs["limiar"] = limiar
        return t


def texto(d):
    nomes = lambda c: c[2:] if c.startswith("f_") else c[2:]
    return " E ".join(f"{nomes(c)}={v}" for c, v in d)


# ---------------------------------------------------------------------------
def continua(df, alvo, f, feature, sentido, rng, n=10000):
    m = df.groupby("circuito")[alvo].mean()
    x = f.loc[m.index, feature].to_numpy()
    y = m.to_numpy()
    obs = spearmanr(x, y).correlation
    nul = np.array([spearmanr(x, rng.permutation(y)).correlation for _ in range(n)])
    p = (np.sum(sentido * nul >= sentido * obs) + 1) / (n + 1)
    return obs, p


def roda_hipotese(nome, cfg, f, t, rng, df=None, saida=SAIDA):
    df = carrega(cfg["base"]) if df is None else df
    titulo(f"{nome}: alvo {cfg['alvo']} ({len(df)} linhas, {df['circuito'].nunique()} circuitos)")
    linhas_resumo = []
    if nome == "H5":
        st = df.sort_values(["circuito", "ano", "round", "piloto", "stint"]).copy()
        st["metade"] = np.arange(len(st)) % 2
        mm = st.groupby(["circuito", "metade"])[cfg["alvo"]].median().unstack()
        r_conf = spearmanr(mm[0], mm[1]).correlation
        ok_conf = r_conf >= 0.50
        s = f"Qualidade de H5 (metades de stints, entre circuitos): Spearman {r_conf:.2f} " \
            f"(>= 0,50) {'PASSA' if ok_conf else 'FALHA: H5 não é interpretada'}"
        print(s)
        linhas_resumo.append(s)
    b = Busca(df, cfg["alvo"], cfg["condicoes"], cfg["suporte"], t)
    res = b.roda(rng)
    print(f"Média geral {res.attrs['media']:.4f}; elegíveis {len(res)}; limiar |z| "
          f"{res.attrs['limiar']:.2f}; firmes {int(res['firme'].sum())}; firmes com feature "
          f"{int((res['firme'] & res['tem_feature']).sum())}")
    cols = ["descricao", "linhas", "circuitos", "media", "diferenca", "z", "firme"]
    print("\nMaiores |z| com feature:")
    print(res[res["tem_feature"]].head(10)[cols].to_string(index=False, float_format=lambda x: f"{x:+.4f}",
                                                           max_colwidth=60))
    acr = b.acrescenta(res, cfg["previstos"], rng)
    print(f"\nA pista acrescenta? (limiar |z_acr| {acr.attrs['limiar']:.2f}, {len(acr)} testados)")
    print(acr.head(12)[["descricao", "linhas", "circuitos", "media", "referencia_outras_pistas",
                        "diferenca_da_pista", "z_acr", "acrescenta", "firme_na_busca", "previsto"]]
          .to_string(index=False, float_format=lambda x: f"{x:+.4f}", max_colwidth=50))
    prev = acr[acr["previsto"]]
    conf = prev[prev["acrescenta"] & prev["firme_na_busca"]
                & (np.sign(prev["diferenca_da_pista"]) == cfg["sentido"])]
    rho, p = continua(b.df, cfg["alvo"], f, cfg["feature"], cfg["sentido_cont"], rng)
    veredito = "CONFIRMADA" if len(conf) else "NÃO CONFIRMADA"
    if nome == "H5" and not ok_conf:
        veredito += " (medida não passou na qualidade; não interpretar)"
    s1 = f"{nome}: {veredito}. Previstos: " + "; ".join(
        f"{r['descricao']} {r['diferenca_da_pista']:+.4f} (z_acr {r['z_acr']:+.2f}, "
        f"{'acrescenta' if r['acrescenta'] else 'não acrescenta'}, "
        f"{'firme' if r['firme_na_busca'] else 'não firme'})" for _, r in prev.iterrows())
    s2 = f"   contínuo (informativo): Spearman alvo médio x {cfg['feature']} = {rho:+.2f}, p = {p:.4f}"
    print("\n" + s1 + "\n" + s2)
    saida.mkdir(parents=True, exist_ok=True)
    res.drop(columns="chave").to_csv(saida / f"{nome}_subgrupos.csv", index=False)
    acr.to_csv(saida / f"{nome}_acrescenta.csv", index=False)
    return linhas_resumo + [s1, s2]


def sintetico(f, t):
    """Base de duelos real com alvo sintético 'trem': sem efeito de pista e com efeito
    plantado nos circuitos de vel_media alta (+12 p.p.)."""
    d = pd.read_csv(PROC / "base_duelos.csv")
    rng = np.random.default_rng(5)
    alta = d["circuito"].map(t["vel_media"]).eq("alta").to_numpy()
    base_p = np.where(d["d_distancia"] == "0,6 a 1,0 s", 0.45, 0.38)
    for efeito in (0.0, 0.12):
        x = d.copy()
        x["trem"] = (rng.random(len(x)) < base_p + efeito * alta).astype(int)
        cfg = dict(HIPOTESES["H3"])
        roda_hipotese(f"sintetico_{int(efeito * 100)}", cfg, f, t, np.random.default_rng(SEMENTE),
                      df=x, saida=SAIDA / "sintetico")


def main():
    global PERMUTACOES, PERM_ACR
    ap = argparse.ArgumentParser()
    ap.add_argument("--so")
    ap.add_argument("--sintetico", action="store_true")
    args = ap.parse_args()
    f, t = tercis()
    titulo("TERCIS DAS FEATURES (26 circuitos)")
    print(t.to_string())
    if args.sintetico:
        PERMUTACOES, PERM_ACR = 200, 500
        sintetico(f, t)
        return
    resumo = []
    for nome, cfg in HIPOTESES.items():
        if args.so and nome != args.so:
            continue
        resumo += roda_hipotese(nome, cfg, f, t, np.random.default_rng(SEMENTE))
    titulo("RESUMO")
    print("\n".join(resumo))
    if not args.so:
        (SAIDA / "resumo.txt").write_text("\n".join(resumo) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
