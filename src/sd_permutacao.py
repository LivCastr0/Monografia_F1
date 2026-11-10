"""
Etapa 2 - Funções comuns da busca de subgrupos com limiar por permutação.

Usado por busca_subgrupos_decisao_parada.py. É a mesma lógica de
busca_subgrupos_duelos.py e reduz_redundancia_duelos.py, escrita de forma genérica
para qualquer alvo 0/1 ou numérico. Os scripts da opção 4 não foram alterados.

Todas as funções esperam o DataFrame ORDENADO POR CIRCUITO, com as colunas
`circuito`, `ano`, `round`, `d_perfil` e `d_era`.
"""
from itertools import combinations

import numpy as np
import pandas as pd
import pysubgroup as ps

BLOCO = 400   # subgrupos calculados de cada vez, para não estourar a memória


def taxa_por_perfil(df, alvo, rng, permutacoes):
    t = df.groupby("d_perfil").agg(linhas=(alvo, "size"), circuitos=("circuito", "nunique"),
                                   media=(alvo, "mean"))
    print(t.to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"\nGeral: {df[alvo].mean():.4f}")
    g = df.groupby("circuito")[alvo].agg(["sum", "size"])
    perfil = df.drop_duplicates("circuito").set_index("circuito")["d_perfil"].reindex(g.index)
    media = g["sum"].sum() / g["size"].sum()

    def sep(rotulos):
        s = g.groupby(np.asarray(rotulos)).sum()
        return ((s["sum"] / s["size"] - media) ** 2 * s["size"]).sum()

    obs = sep(perfil)
    maiores = sum(sep(rng.permutation(perfil.to_numpy())) >= obs for _ in range(permutacoes))
    print(f"O perfil muda o alvo? p = {(maiores + 1) / (permutacoes + 1):.4f} "
          f"(permutação entre {len(g)} circuitos)")
    tc = g.assign(media=g["sum"] / g["size"], perfil=perfil).sort_values("media", ascending=False)
    print("\nPor circuito:")
    print(tc[["perfil", "size", "media"]].rename(columns={"size": "linhas"})
          .to_string(float_format=lambda x: f"{x:.4f}"))


def busca(df, descritores, alvo, rng, profundidade, suporte, permutacoes,
          min_circuitos, max_circuito):
    """Busca, limiar por permutação e verificações. Devolve a tabela de subgrupos."""
    dados = df[descritores + [alvo]].astype({alvo: float})
    tarefa = ps.SubgroupDiscoveryTask(
        dados, ps.NumericTarget(alvo), ps.create_selectors(dados, ignore=[alvo]),
        result_set_size=100_000, depth=profundidade, qf=ps.StandardQFNumeric(a=0.5),
        constraints=[ps.MinSupportConstraint(suporte)])
    grupos = [{s.attribute_name: s.attribute_value for s in sg.selectors}
              for _, sg in ps.Apriori().execute(tarefa).to_descriptions()]

    condicoes = sorted({cv for g in grupos for cv in g.items()})
    pos = {cv: i for i, cv in enumerate(condicoes)}
    idx = np.full((len(grupos), profundidade), len(condicoes))
    for k, g in enumerate(grupos):
        for j, cv in enumerate(g.items()):
            idx[k, j] = pos[cv]
    circ = df["circuito"].to_numpy()
    inicios = np.flatnonzero(np.r_[True, circ[1:] != circ[:-1]])

    def avalia(colunas, y):
        m = np.ones((len(condicoes) + 1, len(df)), dtype=bool)
        for i, (c, v) in enumerate(condicoes):
            m[i] = colunas[c] == v
        yc = (y - y.mean()).astype(np.float32)
        dp = y.std(ddof=1)
        z = np.zeros(len(grupos))
        ok = np.zeros(len(grupos), dtype=bool)
        for a in range(0, len(grupos), BLOCO):
            b = min(a + BLOCO, len(grupos))
            cobre = m[idx[a:b, 0]]
            for j in range(1, profundidade):
                cobre = cobre & m[idx[a:b, j]]
            cont = np.add.reduceat(cobre, inicios, axis=1, dtype=np.int32)
            n = cont.sum(axis=1)
            o = ((n >= suporte) & ((cont > 0).sum(axis=1) >= min_circuitos)
                 & (cont.max(axis=1) <= max_circuito * np.maximum(n, 1)))
            soma = cobre.astype(np.float32) @ yc
            z[a:b][o] = soma[o] / np.sqrt(n[o]) / dp
            ok[a:b] = o
        return z, ok

    originais = {c: df[c].to_numpy() for c in descritores}
    y = df[alvo].to_numpy(dtype=float)
    zs, ok = avalia(originais, y)

    corrida = (df["ano"].astype(str) + "_" + df["round"].astype(str)).to_numpy()
    por_corrida = list(pd.Series(np.arange(len(df))).groupby(corrida).indices.values())
    mapa = df.drop_duplicates("circuito").set_index("circuito")["d_perfil"]
    eras = pd.DataFrame({"corrida": corrida, "circuito": circ, "d_era": df["d_era"].to_numpy()}) \
        .drop_duplicates("corrida").set_index("corrida")
    maximos = np.empty(permutacoes)
    for i in range(permutacoes):
        v = y.copy()
        for linhas in por_corrida:
            v[linhas] = y[rng.permutation(linhas)]
        col = dict(originais)
        if "d_perfil" in col:
            novo = pd.Series(rng.permutation(mapa.to_numpy()), index=mapa.index)
            col["d_perfil"] = pd.Series(circ).map(novo).to_numpy()
        if "d_era" in col:
            nova = eras.groupby("circuito")["d_era"].transform(lambda s: rng.permutation(s.to_numpy()))
            col["d_era"] = pd.Series(corrida).map(nova).to_numpy()
        maximos[i] = np.abs(avalia(col, v)[0]).max()
    limiar = np.quantile(maximos, 0.95)

    media = y.mean()
    linhas = []
    for k in np.where(ok)[0]:
        g = grupos[k]
        m = np.ones(len(df), dtype=bool)
        for c, v in g.items():
            m &= originais[c] == v
        sub = df[m]
        s = np.sign(y[m].mean() - media)
        n = sub["circuito"].value_counts()
        sem = sub[sub["circuito"] != n.index[0]]
        por_era = sub.groupby("d_era")[alvo].mean()
        eras_ok = "d_era" in g or (len(por_era) == 2 and all(np.sign(por_era - media) == s))
        sem_ok = len(sem) > 0 and np.sign(sem[alvo].mean() - media) == s
        passou = abs(zs[k]) > limiar
        linhas.append({"descricao": " E ".join(f"{c[2:]}={v}" for c, v in g.items()),
                       "linhas": int(m.sum()), "circuitos": len(n), "media": y[m].mean(),
                       "z": zs[k], "acima_do_limiar": passou, "firme": passou and eras_ok and sem_ok,
                       "tem_perfil": "d_perfil" in g,
                       "media_2018_21": por_era.get("2018-21", np.nan),
                       "media_2022_mais": por_era.get("2022+", np.nan)})
    res = pd.DataFrame(linhas).sort_values("z", key=abs, ascending=False)
    print(f"Subgrupos: {len(grupos)}, elegíveis ({min_circuitos}+ circuitos, nenhum acima de "
          f"{max_circuito:.0%}): {len(res)}")
    print(f"Limiar |z| (5%, {permutacoes} permutações): {limiar:.2f}")
    print(f"Acima do limiar: {int(res['acima_do_limiar'].sum())}   firmes: {int(res['firme'].sum())}"
          f"   firmes com perfil: {int((res['firme'] & res['tem_perfil']).sum())}")
    return res


def mascara(df, descricao, sem_perfil=False):
    m = np.ones(len(df), dtype=bool)
    for cond in descricao.split(" E "):
        c, v = cond.split("=", 1)
        if not (sem_perfil and c == "perfil"):
            m &= df["d_" + c].to_numpy() == v
    return m


def perfil_acrescenta(df, res, alvo, rng, permutacoes):
    """Para cada subgrupo firme com perfil: o perfil acrescenta algo além do resto?"""
    res = res.copy()
    res["perfil_acrescenta"] = pd.NA
    alvo_res = res[res["firme"] & res["tem_perfil"]]
    if alvo_res.empty:
        print("Nenhum subgrupo firme com condição de perfil.")
        return res
    y = df[alvo].to_numpy(dtype=float)
    mapa = df.drop_duplicates("circuito").set_index("circuito")["d_perfil"]
    circ = df["circuito"]
    casos = []
    for i, r in alvo_res.iterrows():
        valor = r["descricao"].split("perfil=")[1].split(" E ")[0]
        resto = mascara(df, r["descricao"], sem_perfil=True)
        casos.append((i, valor, resto, y[resto].mean(), y[resto].std(ddof=1)))

    def est(perfis):
        z = np.zeros(len(casos))
        for k, (_, valor, resto, base, dp) in enumerate(casos):
            m = resto & (perfis == valor)
            if m.any() and dp > 0:
                z[k] = (y[m].mean() - base) * np.sqrt(m.sum()) / dp
        return z

    obs = est(df["d_perfil"].to_numpy())
    maximos = np.array([np.abs(est(circ.map(pd.Series(rng.permutation(mapa.to_numpy()),
                                                       index=mapa.index)).to_numpy())).max()
                        for _ in range(permutacoes)])
    limiar = np.quantile(maximos, 0.95)
    linhas = []
    for k, (i, _, _, base, _) in enumerate(casos):
        res.at[i, "perfil_acrescenta"] = abs(obs[k]) > limiar
        linhas.append({"descricao": res.at[i, "descricao"], "com_perfil": res.at[i, "media"],
                       "sem_perfil": base, "z_perfil": obs[k], "acrescenta": abs(obs[k]) > limiar})
    t = pd.DataFrame(linhas).sort_values("z_perfil", key=abs, ascending=False)
    print(f"Testados: {len(t)}. Limiar |z| corrigido ({permutacoes} permutações): {limiar:.2f}\n")
    print(t.head(15).to_string(index=False, float_format=lambda x: f"{x:.4f}", max_colwidth=60))
    print(f"\nO perfil acrescenta em {int(t['acrescenta'].sum())} de {len(t)}.")
    return res


def sem_redundancia(df, res, alvo, min_ganho, min_z):
    """Fica o firme que muda a média em min_ganho e min_z erros-padrão contra cada
    versão mais simples dele, incluindo a base inteira."""
    y = df[alvo].to_numpy(dtype=float)
    media = y.mean()
    linhas = []
    for _, r in res[res["firme"]].iterrows():
        conds = r["descricao"].split(" E ")
        m = mascara(df, r["descricao"])
        n, valor = m.sum(), y[m].mean()
        sinal = np.sign(valor - media)
        pior, pior_z, com = np.inf, np.inf, None
        for k in range(len(conds)):
            for sub in combinations(conds, k):
                base = y[mascara(df, " E ".join(sub))].mean() if sub else media
                ganho = sinal * (valor - base)
                z = ganho / np.sqrt(max(base * (1 - base), 1e-12) / n)
                if ganho < pior:
                    pior, pior_z, com = ganho, z, " E ".join(sub) or "base inteira"
        linhas.append({**r.to_dict(), "menor_ganho": pior, "z_do_menor_ganho": pior_z,
                       "comparado_com": com, "fica": pior >= min_ganho and pior_z >= min_z})
    t = pd.DataFrame(linhas)
    return t[t["fica"]].sort_values("z", key=abs, ascending=False) if len(t) else t
