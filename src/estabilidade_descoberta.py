"""
Etapa 2 - A descoberta se repete? Estabilidade das regras de duelos e de parada
quando a busca inteira é refeita em amostras diferentes.

Até aqui, a estabilidade relatada era só das regras já escolhidas, medida nos mesmos
dados da busca (mesmo sentido nas eras, sem cada circuito, intervalos por circuitos).
Aqui a DESCOBERTA é repetida: a busca roda numa parte dos dados e as regras são
avaliadas na outra parte.

Entra: data/processed/base_duelos.csv, base_decisao_parada.csv,
       regras_globais_duelos.csv, regras_globais_parada.csv (as regras de referência)
Sai:   data/processed/estabilidade/<analise>_por_regra.csv, <analise>_por_divisao.csv,
       <analise>_eras.csv, resumo.txt
Uso:   python src/estabilidade_descoberta.py [--divisoes 100] [--so duelos]

Desenho (fixado antes de rodar)
-------------------------------
Divisões:
  E  entre eras: descobre em 2018-21 e avalia em 2022+; e o inverso.
  C  entre circuitos: 100 divisões aleatórias dos 26 circuitos em 13 + 13
     (default_rng(2026)); descobre numa metade, avalia na outra.

Busca em cada amostra de descoberta (versão reduzida da original, declarada):
  - descritores das análises originais sem perfil e sem era; profundidade 2 nas duas
    análises (a original de duelos tinha profundidade 3, mas todas as regras finais
    têm 1 ou 2 condições);
  - suporte absoluto igual ao original (100 voltas de duelo; 200 voltas de parada),
    4+ circuitos, nenhum acima de 40%;
  - limiar: 95% do máximo |z| em 200 permutações do alvo dentro da corrida;
  - firme: passa do limiar, mesmo sentido sem o circuito mais frequente e, quando a
    amostra tem as duas eras, mesmo sentido nas duas;
  - redundância como na original (3 p.p. nos duelos, 1 p.p. na parada; 2 erros-padrão);
  - seleção por cobertura ponderada como em consolida_regras_globais.py (limiar da
    busca; até 8 regras acima e 3 abaixo; candidatas aninhadas saem).

Medidas
  Por regra de referência (as 19 do documento):
    redescoberta  % das divisões em que a regra é firme na amostra de descoberta
    selecionada   % das divisões em que ela entra no conjunto final
    fora          efeito na amostra de avaliação (taxa na regra − taxa da amostra),
                  mediana e % das divisões com o mesmo sentido
    encolhimento  mediana de efeito fora / efeito dentro, nas divisões em que a regra
                  foi selecionada (mede o viés de seleção)
  Por divisão:
    Jaccard entre o conjunto selecionado e o de referência
    % das regras selecionadas que mantêm o sentido fora e com pelo menos metade do
    efeito de dentro
Uma regra com condições iguais e na mesma ordem lógica é a mesma regra (comparação
por conjunto de condições).

Associação, não causa.
"""
import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import consolida_regras_globais as CR   # noqa: E402
import sd_permutacao as S               # noqa: E402

P = Path("data/processed")
SAIDA = P / "estabilidade"
PERMUTACOES, SEMENTE = 200, 2026
MIN_CIRCUITOS, MAX_CIRCUITO = 4, 0.40

ANALISES = {
    "duelos": dict(base="base_duelos.csv", regras="regras_globais_duelos.csv", alvo="passou",
                   suporte=100, min_ganho=0.03,
                   descritores=["d_dif_idade", "d_dif_composto", "d_distancia", "d_trem",
                                "d_largada", "d_fase", "d_relargada", "d_posicao",
                                "d_mesma_equipe"]),
    "parada": dict(base="base_decisao_parada.csv", regras="regras_globais_parada.csv", alvo="parou",
                   suporte=200, min_ganho=0.01,
                   descritores=["d_idade_pneu", "d_composto", "d_neutralizacao", "d_rival_frente",
                                "d_rival_tras", "d_janela", "d_dist_frente", "d_paradas_feitas",
                                "d_fase", "d_posicao"]),
}


def chave(desc):
    return frozenset(desc.split(" E "))


def descricoes(df, descritores):
    sel = [(c, v) for c in descritores for v in sorted(df[c].dropna().astype(str).unique())]
    return [(s,) for s in sel] + [(a, b) for a, b in combinations(sel, 2) if a[0] != b[0]]


def texto(d):
    return " E ".join(f"{c[2:]}={v}" for c, v in d)


def busca(df, descs, alvo, suporte, rng):
    """Busca com limiar por permutação; devolve a tabela de elegíveis com 'firme'."""
    df = df.sort_values(["circuito", "ano", "round"], kind="stable").reset_index(drop=True)
    y = df[alvo].to_numpy(dtype=float)
    circ = df["circuito"].to_numpy()
    inicios = np.flatnonzero(np.r_[True, circ[1:] != circ[:-1]])
    cols = {c: df[c].astype(str).to_numpy() for c in {c for d in descs for c, _ in d}}
    base = {}
    cob = np.empty((len(descs), len(df)), dtype=bool)
    for k, d in enumerate(descs):
        r = np.ones(len(df), dtype=bool)
        for s in d:
            if s not in base:
                base[s] = cols[s[0]] == s[1]
            r &= base[s]
        cob[k] = r
    cont = np.add.reduceat(cob, inicios, axis=1, dtype=np.int32)
    n = cont.sum(1)
    ok = ((n >= suporte) & ((cont > 0).sum(1) >= MIN_CIRCUITOS)
          & (cont.max(1) <= MAX_CIRCUITO * np.maximum(n, 1)))
    idx = np.flatnonzero(ok)
    if len(idx) == 0:
        return df, pd.DataFrame(), np.inf
    c = cob[idx].astype(np.float32)
    media, dp = y.mean(), y.std(ddof=1)
    z = (c @ (y - media).astype(np.float32)) / np.sqrt(n[idx]) / dp
    corrida = pd.factorize(df["ano"].astype(str) + "_" + df["round"].astype(str))[0]
    grupos = list(pd.Series(np.arange(len(df))).groupby(corrida).indices.values())
    maximos = np.empty(PERMUTACOES)
    for i in range(PERMUTACOES):
        v = y.copy()
        for g in grupos:
            v[g] = y[rng.permutation(g)]
        zp = (c @ (v - v.mean()).astype(np.float32)) / np.sqrt(n[idx]) / v.std(ddof=1)
        maximos[i] = np.abs(zp).max()
    limiar = float(np.quantile(maximos, 0.95))
    era = df["d_era"].to_numpy()
    eras = np.unique(era)
    linhas = []
    for j, k in enumerate(idx):
        m = cob[k]
        s = np.sign(y[m].mean() - media)
        conta = pd.Series(circ[m]).value_counts()
        sem = m & (circ != conta.index[0])
        sem_ok = sem.any() and np.sign(y[sem].mean() - media) == s
        if len(eras) == 2:
            eras_ok = all((m & (era == e)).any() and np.sign(y[m & (era == e)].mean() - media) == s
                          for e in eras)
        else:
            eras_ok = True
        linhas.append({"descricao": texto(descs[k]), "linhas": int(n[k]), "media": y[m].mean(),
                       "z": float(z[j]), "firme": bool(abs(z[j]) > limiar and sem_ok and eras_ok),
                       "tem_perfil": False})
    return df, pd.DataFrame(linhas), limiar


def descobre(df, cfg, descs, rng):
    """Busca + redundância + seleção. Devolve (firmes, selecionadas com z)."""
    df, res, limiar = busca(df, descs, cfg["alvo"], cfg["suporte"], rng)
    if res.empty or not res["firme"].any():
        return set(), []
    fin = S.sem_redundancia(df, res, cfg["alvo"], cfg["min_ganho"], 2.0)
    firmes = {chave(d) for d in res.loc[res["firme"], "descricao"]}
    if fin is None or len(fin) == 0:
        return firmes, []
    y = df[cfg["alvo"]].to_numpy(dtype=float)
    base, dp = y.mean(), y.std()
    mascaras = {d: CR.mascara(df, d) for d in fin["descricao"]}
    sel = (CR.seleciona(y, mascaras, +1, limiar, CR.MAX_ACIMA, base, dp)
           + CR.seleciona(y, mascaras, -1, limiar, CR.MAX_ABAIXO, base, dp))
    return firmes, [d for d, _ in sel]


def efeito(df, alvo, desc):
    y = df[alvo].to_numpy(dtype=float)
    m = CR.mascara(df, desc)
    return (y[m].mean() - y.mean()) if m.sum() >= 1 else np.nan, int(m.sum())


def avalia_divisao(cfg, ref, disc, aval, descs, rng):
    firmes, sel = descobre(disc, cfg, descs, rng)
    a = cfg["alvo"]
    por_regra = []
    for r in ref:
        e_in, n_in = efeito(disc, a, r)
        e_out, n_out = efeito(aval, a, r)
        por_regra.append({"regra": r, "firme": chave(r) in firmes,
                          "selecionada": chave(r) in {chave(s) for s in sel},
                          "efeito_dentro": e_in, "efeito_fora": e_out,
                          "linhas_fora": n_out})
    ksel, kref = {chave(s) for s in sel}, {chave(r) for r in ref}
    jac = len(ksel & kref) / len(ksel | kref) if (ksel | kref) else np.nan
    mantem = []
    for s in sel:
        e_in, _ = efeito(disc, a, s)
        e_out, n_out = efeito(aval, a, s)
        mantem.append(n_out > 0 and np.sign(e_out) == np.sign(e_in) and abs(e_out) >= 0.5 * abs(e_in))
    resumo = {"selecionadas": len(sel), "em_comum_com_referencia": len(ksel & kref),
              "jaccard": jac, "pct_selecionadas_que_se_mantem_fora": 100 * np.mean(mantem) if mantem else np.nan,
              "regras_selecionadas": " | ".join(sel)}
    return pd.DataFrame(por_regra), resumo


def roda(nome, cfg, n_div, rng):
    df = pd.read_csv(P / cfg["base"])
    ref = pd.read_csv(P / cfg["regras"])["regra"].tolist()
    sentido_ref = {r: np.sign(d) for r, d in pd.read_csv(P / cfg["regras"])[["regra", "dif_pp"]].itertuples(index=False)}
    descs = descricoes(df, cfg["descritores"])
    print(f"\n{'=' * 78}\n{nome.upper()}: {len(df)} linhas, {len(ref)} regras de referência, "
          f"{len(descs)} descrições\n{'=' * 78}")

    # E: entre eras
    linhas_e, resumo_e = [], []
    for d_era, a_era in (("2018-21", "2022+"), ("2022+", "2018-21")):
        pr, rs = avalia_divisao(cfg, ref, df[df["d_era"] == d_era], df[df["d_era"] == a_era], descs, rng)
        pr.insert(0, "descobre_em", d_era)
        linhas_e.append(pr)
        resumo_e.append({"descobre_em": d_era, **rs})
        print(f"\nDescobre em {d_era}, avalia em {a_era}: {rs['selecionadas']} regras selecionadas, "
              f"{rs['em_comum_com_referencia']} em comum com a referência (Jaccard {rs['jaccard']:.2f}); "
              f"{rs['pct_selecionadas_que_se_mantem_fora']:.0f}% das selecionadas mantêm sentido e metade do efeito fora")
        print(pr.assign(efeito_dentro=100 * pr["efeito_dentro"], efeito_fora=100 * pr["efeito_fora"])
              [["regra", "firme", "selecionada", "efeito_dentro", "efeito_fora", "linhas_fora"]]
              .to_string(index=False, float_format=lambda x: f"{x:+.1f}", max_colwidth=62))
    eras = pd.concat(linhas_e, ignore_index=True)

    # C: entre circuitos
    circs = np.array(sorted(df["circuito"].unique()))
    linhas_c, resumo_c = [], []
    for i in range(n_div):
        metade = set(rng.permutation(circs)[: len(circs) // 2])
        disc, aval = df[df["circuito"].isin(metade)], df[~df["circuito"].isin(metade)]
        pr, rs = avalia_divisao(cfg, ref, disc, aval, descs, rng)
        pr.insert(0, "divisao", i)
        linhas_c.append(pr)
        resumo_c.append({"divisao": i, **rs})
        if (i + 1) % 10 == 0:
            print(f"  divisões de circuitos: {i + 1}/{n_div}", flush=True)
    c = pd.concat(linhas_c, ignore_index=True)
    rc = pd.DataFrame(resumo_c)

    def por_regra(g):
        s = sentido_ref[g.name]
        sel = g[g["selecionada"]]
        enc = (sel["efeito_fora"] / sel["efeito_dentro"]).median() if len(sel) else np.nan
        return pd.Series({
            "redescoberta_pct": 100 * g["firme"].mean(),
            "selecionada_pct": 100 * g["selecionada"].mean(),
            "efeito_fora_mediana_pp": 100 * g["efeito_fora"].median(),
            "mesmo_sentido_fora_pct": 100 * (np.sign(g["efeito_fora"]) == s).mean(),
            "encolhimento_mediano": enc,
            "linhas_fora_mediana": g["linhas_fora"].median(),
        })
    pr = c.groupby("regra", sort=False).apply(por_regra).reset_index()
    print(f"\nEntre circuitos ({n_div} divisões 13 + 13), por regra de referência:")
    print(pr.to_string(index=False, float_format=lambda x: f"{x:.2f}", max_colwidth=62))
    print(f"\nPor divisão: selecionadas mediana {rc['selecionadas'].median():.0f}; em comum com a referência "
          f"mediana {rc['em_comum_com_referencia'].median():.0f}; Jaccard mediano {rc['jaccard'].median():.2f}; "
          f"selecionadas que se mantêm fora, mediana {rc['pct_selecionadas_que_se_mantem_fora'].median():.0f}%")

    SAIDA.mkdir(parents=True, exist_ok=True)
    eras.to_csv(SAIDA / f"{nome}_eras.csv", index=False)
    pd.DataFrame(resumo_e).to_csv(SAIDA / f"{nome}_eras_resumo.csv", index=False)
    pr.to_csv(SAIDA / f"{nome}_por_regra.csv", index=False)
    rc.to_csv(SAIDA / f"{nome}_por_divisao.csv", index=False)
    c.to_csv(SAIDA / f"{nome}_circuitos_detalhe.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--divisoes", type=int, default=100)
    ap.add_argument("--so")
    args = ap.parse_args()
    rng = np.random.default_rng(SEMENTE)
    for nome, cfg in ANALISES.items():
        if args.so and nome != args.so:
            continue
        roda(nome, cfg, args.divisoes, rng)


if __name__ == "__main__":
    main()
