"""
Etapa 2 - Hipótese (ii) original: afinidade piloto/equipe x perfil, sem exigir suporte.

A versão com o piloto como descritor foi descartada em setembro por falta de
suporte (diagnostico_h2.py), antes de qualquer busca. Esta é a busca dessa
versão, rodada como registro com suporte mínimo reduzido para 10 linhas.
Critérios fixados antes de rodar: claude/h2-piloto-perfil-msi2.md (projeto no
claude.ai). Exploratória declarada.

Alvo: `alvo` de base_analise_sd.csv (referência móvel do piloto - posição
normalizada), sem desconto da largada.
Descritores: piloto, equipe, perfil; profundidade 1 e 2.

Uso:
  python src/sd_h2_piloto_perfil.py               base real
  python src/sd_h2_piloto_perfil.py --sintetico   testes sintéticos
"""
import sys
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pandas as pd

ENTRADA = Path("data/processed/base_analise_sd.csv")
SAIDA = Path("data/processed/sd_h2_piloto_perfil")

DESCRITORES = ["d_piloto", "d_equipe", "d_perfil"]
ALVO = "alvo"
SUPORTE = 10
PROFUNDIDADE = 2
MIN_ERA = 5
PERMUTACOES = 1000
CORRIDA = ["ano", "round"]
PLANTADO = {"d_piloto": "HAM", "d_perfil": "stop-and-go"}
EFEITO_PLANTADO = 3.0


def monta_base():
    df = pd.read_csv(ENTRADA)
    df["d_piloto"] = df["piloto"]
    df["d_equipe"] = df["equipe"]
    df["era"] = np.where(df["ano"] <= 2021, "2018-21", "2022+")
    return df.reset_index(drop=True)


def descricoes(df):
    sel = [(d, v) for d in DESCRITORES for v in sorted(df[d].dropna().unique())]
    masc = {s: (df[s[0]] == s[1]).to_numpy() for s in sel}
    lista, mascs = [], []
    for k in range(1, PROFUNDIDADE + 1):
        for attrs in combinations(DESCRITORES, k):
            for comb in product(*[[s for s in sel if s[0] == a] for a in attrs]):
                m = np.logical_and.reduce([masc[s] for s in comb])
                if m.sum() >= SUPORTE:
                    lista.append(comb)
                    mascs.append(m)
    return lista, np.array(mascs)


def texto(comb):
    return " E ".join(f"{a[2:]}={v}" for a, v in comb)


def zs(cobre, y):
    return cobre @ (y - y.mean()) / np.sqrt(cobre.sum(axis=1)) / y.std(ddof=1)


def limiar_busca(df, cobre, y, rng):
    corridas = list(df.groupby(CORRIDA).indices.values())
    maxs = np.empty(PERMUTACOES)
    for i in range(PERMUTACOES):
        e = y.copy()
        for linhas in corridas:
            e[linhas] = y[rng.permutation(linhas)]
        maxs[i] = np.abs(zs(cobre, e)).max()
    return np.quantile(maxs, 0.95)


def prepara_acrescenta(df, lista):
    """Descrições quem E perfil: índice, perfil P e máscara de 'quem'."""
    itens = []
    for i, comb in enumerate(lista):
        d = dict(comb)
        if len(comb) == 2 and "d_perfil" in d:
            quem = [s for s in comb if s[0] != "d_perfil"][0]
            itens.append((i, d["d_perfil"], (df[quem[0]] == quem[1]).to_numpy()))
    return itens


def z_acrescenta(itens, perfil, y):
    sd = y.std(ddof=1)
    out = {}
    for i, p, quem in itens:
        dentro, fora = quem & (perfil == p), quem & (perfil != p)
        n1, n2 = dentro.sum(), fora.sum()
        if n1 < 2 or n2 < 2:
            continue
        out[i] = (y[dentro].mean() - y[fora].mean()) / (sd * np.sqrt(1 / n1 + 1 / n2))
    return out


def limiar_acrescenta(df, itens, y, rng):
    mapa = df.groupby("circuito")["d_perfil"].first()
    circ_idx = pd.Categorical(df["circuito"], categories=mapa.index).codes
    rot = mapa.to_numpy()
    maxs = np.empty(PERMUTACOES)
    for k in range(PERMUTACOES):
        z = z_acrescenta(itens, rng.permutation(rot)[circ_idx], y)
        maxs[k] = np.abs(list(z.values())).max()
    return np.quantile(maxs, 0.95)


def rodada(df, rotulo, semente=2026, salvar=True):
    rng = np.random.default_rng(semente)
    y = df[ALVO].to_numpy().astype(float)
    lista, cobre = descricoes(df)
    cobre_f = cobre.astype(float)
    z = zs(cobre_f, y)
    lim = limiar_busca(df, cobre_f, y, rng)
    itens = prepara_acrescenta(df, lista)
    za = z_acrescenta(itens, df["d_perfil"].to_numpy(), y)
    lim_acr = limiar_acrescenta(df, itens, y, rng)

    linhas = []
    for i, comb in enumerate(lista):
        sub = df[cobre[i]]
        media = sub[ALVO].mean()
        sinal = np.sign(media)
        maior = sub["circuito"].value_counts().index[0]
        resto = sub[sub["circuito"] != maior]
        sem_maior = resto[ALVO].mean() if len(resto) else np.nan
        eras = sub.groupby("era")[ALVO].agg(["size", "mean"])
        checa_era = len(eras) == 2 and (eras["size"] >= MIN_ERA).all()
        ok_era = (np.sign(eras["mean"]) == sinal).all() if checa_era else True
        firme = abs(z[i]) > lim and np.sign(sem_maior) == sinal and ok_era
        linhas.append({
            "descricao": texto(comb), "condicoes": len(comb), "linhas": len(sub),
            "media": media, "z": z[i], "acima_limiar": bool(abs(z[i]) > lim),
            "media_2018_21": eras["mean"].get("2018-21", np.nan),
            "media_2022": eras["mean"].get("2022+", np.nan),
            "era_checada": checa_era,
            "media_sem_maior_circuito": sem_maior,
            "circuitos": sub["circuito"].nunique(), "corridas": sub.groupby(CORRIDA).ngroups,
            "firme": bool(firme), "z_acrescenta": za.get(i, np.nan),
        })
    res = pd.DataFrame(linhas)
    res["perfil_acrescenta"] = res["firme"] & (res["z_acrescenta"].abs() > lim_acr)
    res = res.sort_values("z", key=abs, ascending=False)

    com_perfil = res[res["z_acrescenta"].notna()]
    print(f"\n{'=' * 78}\n{rotulo}\n{'=' * 78}")
    print(f"Linhas {len(df)}; desvio do alvo {y.std(ddof=1):.2f}; descrições {len(lista)} "
          f"(piloto/equipe E perfil: {len(com_perfil)})")
    print(f"Limiar |z| {lim:.2f}; maior |z| {res.z.abs().max():.2f}; "
          f"acima {int(res.acima_limiar.sum())}; firmes {int(res.firme.sum())}")
    print(f"Limiar |z_acr| {lim_acr:.2f}; maior |z_acr| {com_perfil.z_acrescenta.abs().max():.2f}; "
          f"afinidades confirmadas (firme e perfil acrescenta): {int(res.perfil_acrescenta.sum())}")
    cols = ["descricao", "linhas", "media", "z", "firme", "z_acrescenta",
            "media_sem_maior_circuito", "circuitos"]
    print("\nMaiores |z| entre piloto/equipe E perfil:")
    print(com_perfil[cols].head(12).to_string(index=False, float_format=lambda x: f"{x:+.2f}"))
    print("\nMaiores |z_acr|:")
    print(com_perfil.reindex(com_perfil.z_acrescenta.abs().sort_values(ascending=False).index)
          [cols].head(8).to_string(index=False, float_format=lambda x: f"{x:+.2f}"))
    if res.firme.any():
        print("\nFirmes:")
        print(res[res.firme][cols].to_string(index=False, float_format=lambda x: f"{x:+.2f}"))
    if salvar:
        SAIDA.mkdir(parents=True, exist_ok=True)
        nome = rotulo.lower().replace(" ", "_")
        res.to_csv(SAIDA / f"subgrupos_{nome}.csv", index=False)
        pd.DataFrame([{"versao": rotulo, "linhas": len(df), "descricoes": len(lista),
                       "descricoes_com_perfil": len(com_perfil), "limiar_z": lim,
                       "acima": int(res.acima_limiar.sum()), "firmes": int(res.firme.sum()),
                       "limiar_z_acrescenta": lim_acr,
                       "maior_z_acrescenta": com_perfil.z_acrescenta.abs().max(),
                       "afinidades_confirmadas": int(res.perfil_acrescenta.sum())}]
                     ).to_csv(SAIDA / f"resumo_{nome}.csv", index=False)
    return res


def sintetico():
    df = monta_base()
    rng = np.random.default_rng(7)
    y = df[ALVO].to_numpy().copy()
    for linhas in df.groupby(CORRIDA).indices.values():
        y[linhas] = y[rng.permutation(linhas)]
    a = df.copy()
    a[ALVO] = y
    rodada(a, "SINTETICO sem efeito", semente=11)
    b = a.copy()
    m = np.logical_and.reduce([(b[k] == v).to_numpy() for k, v in PLANTADO.items()])
    b.loc[m, ALVO] += EFEITO_PLANTADO
    print(f"\nPlantado +{EFEITO_PLANTADO} em {m.sum()} linhas ({texto(tuple(PLANTADO.items()))})")
    rodada(b, "SINTETICO com efeito", semente=12)


if __name__ == "__main__":
    sintetico() if "--sintetico" in sys.argv else rodada(monta_base(), "BASE REAL")
