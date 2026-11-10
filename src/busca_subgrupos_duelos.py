"""
Etapa 2 - Busca de subgrupos nos duelos.

Pergunta: em que condições o carro de trás consegue passar?

Entra: data/processed/base_duelos.csv
Sai:   data/processed/subgrupos_duelos.csv           (versão principal)
       data/processed/subgrupos_duelos_reduzida.csv  (com --reduzida)

  python src/busca_subgrupos_duelos.py              11 descritores
  python src/busca_subgrupos_duelos.py --reduzida    7 descritores

As duas versões foram fixadas antes de ver qualquer resultado. A reduzida é
verificação: um subgrupo firme na principal que também aparece na reduzida é mais
forte; um que aparece só na reduzida é pista, não achado.

Alvo: `passou` (0 ou 1). Linha: volta de duelo.

PARTE 1. A taxa de ultrapassagem por perfil, com p por permutação dos perfis entre
os 26 circuitos. É o único teste em que o urbano-técnico pode aparecer: com 2
circuitos, ele nunca cumpre a exigência de 4 circuitos da busca.

PARTE 2. pysubgroup, StandardQFNumeric (a=0.5), profundidade 3, suporte mínimo 100
voltas de duelo, os dois sentidos. Só concorrem subgrupos com 4+ circuitos e nenhum
circuito acima de 40% das linhas, e a mesma exigência vale dentro das permutações.
Cada repetição do teste embaralha ao mesmo tempo:
  - o alvo entre as voltas de duelo da mesma corrida (testa as condições do duelo);
  - os perfis entre os circuitos (testa o perfil);
  - a era entre as corridas do mesmo circuito (testa a era além do circuito).
O maior valor de cada repetição define o limiar.

Firme, fixado antes de rodar: passa do limiar; mesmo sinal nas duas eras (quando a
era não é condição); mesmo sinal sem o circuito mais frequente.

PARTE 3. O perfil acrescenta? Um subgrupo "perfil=potência E atacante 10+ voltas mais
novo" pode ser firme só por herdar o efeito do pneu. Para cada subgrupo firme com
condição de perfil, compara-se a taxa dele com a do mesmo subgrupo sem a condição de
perfil. O limiar vem de embaralhar os perfis entre os circuitos e guardar a maior
diferença entre todos os subgrupos testados, o que já desconta o número de testes.

Descoberta de subgrupos descreve associação, não causa. Carro mais rápido não está
controlado, exceto em parte por "quem largou à frente".
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pysubgroup as ps

ENTRADA = Path("data/processed/base_duelos.csv")
PRINCIPAL = ["d_perfil", "d_era", "d_dif_idade", "d_dif_composto", "d_distancia", "d_trem",
             "d_largada", "d_fase", "d_relargada", "d_posicao", "d_mesma_equipe"]
REDUZIDA = ["d_perfil", "d_era", "d_dif_idade", "d_dif_composto", "d_distancia", "d_trem",
            "d_largada"]
ALVO = "passou"
SUPORTE, PROFUNDIDADE = 100, 3
PERMUTACOES, PERMUTACOES_PERFIL, SEMENTE = 500, 5000, 14
MIN_CIRCUITOS, MAX_CIRCUITO = 4, 0.40
FMT = lambda x: f"{x:+.3f}"


def parte1(df, rng):
    print("=" * 78 + "\n1. TAXA DE ULTRAPASSAGEM POR PERFIL\n" + "=" * 78)
    t = df.groupby("d_perfil").agg(voltas_de_duelo=(ALVO, "size"),
                                   circuitos=("circuito", "nunique"), taxa=(ALVO, "mean"))
    print(t.to_string(float_format=lambda x: f"{x:.3f}"))
    print(f"\nTaxa geral: {df[ALVO].mean():.3f}")
    g = df.groupby("circuito")[ALVO].agg(["sum", "size"])
    perfil = df.drop_duplicates("circuito").set_index("circuito")["d_perfil"].reindex(g.index)
    media = g["sum"].sum() / g["size"].sum()

    def sep(rotulos):
        s = g.groupby(rotulos.to_numpy() if hasattr(rotulos, "to_numpy") else rotulos).sum()
        return ((s["sum"] / s["size"] - media) ** 2 * s["size"]).sum()

    obs = sep(perfil)
    maiores = sum(sep(rng.permutation(perfil.to_numpy())) >= obs for _ in range(PERMUTACOES_PERFIL))
    print(f"O perfil muda a taxa? p = {(maiores + 1) / (PERMUTACOES_PERFIL + 1):.4f} "
          f"(permutação entre {len(g)} circuitos)")
    print("\nTaxa por circuito:")
    tc = g.assign(taxa=g["sum"] / g["size"], perfil=perfil).sort_values("taxa", ascending=False)
    print(tc[["perfil", "size", "taxa"]].rename(columns={"size": "voltas_de_duelo"})
          .to_string(float_format=lambda x: f"{x:.3f}"))
    print()


def parte2(df, descritores, rng, saida):
    print("=" * 78 + f"\n2. A BUSCA ({len(descritores)} descritores)\n" + "=" * 78)
    busca = df[descritores + [ALVO]].astype({ALVO: float})
    tarefa = ps.SubgroupDiscoveryTask(
        busca, ps.NumericTarget(ALVO), ps.create_selectors(busca, ignore=[ALVO]),
        result_set_size=100_000, depth=PROFUNDIDADE, qf=ps.StandardQFNumeric(a=0.5),
        constraints=[ps.MinSupportConstraint(SUPORTE)])
    grupos = [{s.attribute_name: s.attribute_value for s in sg.selectors}
              for _, sg in ps.Apriori().execute(tarefa).to_descriptions()]

    condicoes = sorted({cv for g in grupos for cv in g.items()})
    pos = {cv: i for i, cv in enumerate(condicoes)}
    idx = np.full((len(grupos), PROFUNDIDADE), len(condicoes))
    for k, g in enumerate(grupos):
        for j, cv in enumerate(g.items()):
            idx[k, j] = pos[cv]
    # As linhas estão ordenadas por circuito: a contagem por circuito é a soma de
    # cada bloco de linhas seguidas.
    inicios = np.flatnonzero(np.r_[True, df["circuito"].to_numpy()[1:] != df["circuito"].to_numpy()[:-1]])

    def avalia(colunas, y):
        m = np.ones((len(condicoes) + 1, len(df)), dtype=bool)
        for i, (c, v) in enumerate(condicoes):
            m[i] = colunas[c] == v
        cobre = m[idx[:, 0]] & m[idx[:, 1]] & m[idx[:, 2]]
        cf = cobre.astype(np.float32)
        cont = np.add.reduceat(cobre, inicios, axis=1, dtype=np.int32)
        n = cont.sum(axis=1)
        ok = ((n >= SUPORTE) & ((cont > 0).sum(axis=1) >= MIN_CIRCUITOS)
              & (cont.max(axis=1) <= MAX_CIRCUITO * np.maximum(n, 1)))
        z = np.zeros(len(grupos))
        soma = cf @ (y - y.mean()).astype(np.float32)
        z[ok] = soma[ok] / np.sqrt(n[ok]) / y.std(ddof=1)
        return cobre, z, ok

    originais = {c: df[c].to_numpy() for c in descritores}
    y = df[ALVO].to_numpy(dtype=float)
    cobre, zs, ok = avalia(originais, y)

    corrida = df["ano"].astype(str) + "_" + df["round"].astype(str)
    por_corrida = list(df.groupby(corrida).indices.values())
    mapa = df.drop_duplicates("circuito").set_index("circuito")["d_perfil"]
    eras = df.assign(corrida=corrida).drop_duplicates("corrida").set_index("corrida")[["circuito", "d_era"]]
    maximos = np.empty(PERMUTACOES)
    for i in range(PERMUTACOES):
        v = y.copy()
        for linhas in por_corrida:
            v[linhas] = y[rng.permutation(linhas)]
        col = dict(originais)
        if "d_perfil" in col:
            novo = pd.Series(rng.permutation(mapa.to_numpy()), index=mapa.index)
            col["d_perfil"] = df["circuito"].map(novo).to_numpy()
        if "d_era" in col:
            nova = eras.groupby("circuito")["d_era"].transform(lambda s: rng.permutation(s.to_numpy()))
            col["d_era"] = corrida.map(nova).to_numpy()
        maximos[i] = np.abs(avalia(col, v)[1]).max()
    limiar = np.quantile(maximos, 0.95)

    media = y.mean()
    linhas = []
    for k in np.where(ok)[0]:
        g, sub = grupos[k], df[cobre[k]]
        s = np.sign(sub[ALVO].mean() - media)
        n = sub["circuito"].value_counts()
        sem = sub[sub["circuito"] != n.index[0]]
        por_era = sub.groupby("d_era")[ALVO].mean()
        eras_ok = "d_era" in g or (len(por_era) == 2 and all(np.sign(por_era - media) == s))
        sem_ok = len(sem) > 0 and np.sign(sem[ALVO].mean() - media) == s
        passou = abs(zs[k]) > limiar
        linhas.append({"descricao": " E ".join(f"{c[2:]}={v}" for c, v in g.items()),
                       "voltas_de_duelo": len(sub), "circuitos": len(n), "taxa": sub[ALVO].mean(),
                       "diferenca": sub[ALVO].mean() - media, "z": zs[k],
                       "acima_do_limiar": passou, "firme": passou and eras_ok and sem_ok,
                       "ok_eras": eras_ok, "ok_sem_maior": sem_ok,
                       "tem_perfil": "d_perfil" in g,
                       "taxa_2018_21": por_era.get("2018-21", np.nan),
                       "taxa_2022_mais": por_era.get("2022+", np.nan)})
    res = pd.DataFrame(linhas).sort_values("z", key=abs, ascending=False)
    firmes = res[res["firme"]]
    print(f"Subgrupos: {len(grupos)}, elegíveis (4+ circuitos, nenhum acima de 40%): {len(res)}")
    print(f"Limiar |z| (5%, {PERMUTACOES} permutações): {limiar:.2f}")
    print(f"Acima do limiar: {int(res['acima_do_limiar'].sum())}   firmes: {len(firmes)}")
    print(f"  firmes com condição de perfil: {int(firmes['tem_perfil'].sum())}\n")
    cols = ["descricao", "voltas_de_duelo", "circuitos", "taxa", "z", "firme"]
    print("Maiores |z|:")
    print(res.head(20)[cols].to_string(index=False, float_format=FMT, max_colwidth=70))
    com_perfil = firmes[firmes["tem_perfil"]]
    if len(com_perfil):
        print("\nFirmes com condição de perfil:")
        print(com_perfil.head(15)[cols + ["taxa_2018_21", "taxa_2022_mais"]].to_string(
            index=False, float_format=FMT, max_colwidth=60))
    res = parte3(df, res, rng)
    res.to_csv(saida, index=False)
    print(f"\nSalvo em {saida}")


def filtro(df, descricao, sem_perfil=False):
    m = np.ones(len(df), dtype=bool)
    for cond in descricao.split(" E "):
        c, v = cond.split("=", 1)
        if sem_perfil and c == "perfil":
            continue
        m &= df["d_" + c].to_numpy() == v
    return m


def parte3(df, res, rng):
    print("\n" + "=" * 78 + "\n3. O PERFIL ACRESCENTA?\n" + "=" * 78)
    res["perfil_acrescenta"] = pd.NA
    alvo = res[res["firme"] & res["tem_perfil"]]
    if alvo.empty:
        print("Nenhum subgrupo firme com condição de perfil.")
        return res
    y = df[ALVO].to_numpy(dtype=float)
    mapa = df.drop_duplicates("circuito").set_index("circuito")["d_perfil"]
    circ = df["circuito"]
    casos = []
    for i, r in alvo.iterrows():
        valor = r["descricao"].split("perfil=")[1].split(" E ")[0]
        resto = filtro(df, r["descricao"], sem_perfil=True)
        casos.append((i, valor, resto, y[resto].mean(), y[resto].std(ddof=1)))

    def estatisticas(perfis):
        z = np.zeros(len(casos))
        for k, (_, valor, resto, base, dp) in enumerate(casos):
            m = resto & (perfis == valor)
            n = m.sum()
            if n and dp > 0:
                z[k] = (y[m].mean() - base) * np.sqrt(n) / dp
        return z

    obs = estatisticas(df["d_perfil"].to_numpy())
    maximos = np.empty(PERMUTACOES)
    for j in range(PERMUTACOES):
        novo = pd.Series(rng.permutation(mapa.to_numpy()), index=mapa.index)
        maximos[j] = np.abs(estatisticas(circ.map(novo).to_numpy())).max()
    limiar = np.quantile(maximos, 0.95)

    linhas = []
    for k, (i, valor, resto, base, _) in enumerate(casos):
        acrescenta = abs(obs[k]) > limiar
        res.at[i, "perfil_acrescenta"] = acrescenta
        linhas.append({"descricao": res.at[i, "descricao"], "taxa_com_perfil": res.at[i, "taxa"],
                       "taxa_sem_perfil": base, "z_perfil": obs[k], "acrescenta": acrescenta})
    t = pd.DataFrame(linhas).sort_values("z_perfil", key=abs, ascending=False)
    print(f"Testados: {len(t)}. Limiar |z| corrigido para todos de uma vez "
          f"(5%, {PERMUTACOES} permutações dos perfis entre circuitos): {limiar:.2f}\n")
    print(t.head(20).to_string(index=False, float_format=lambda x: f"{x:.3f}", max_colwidth=60))
    print(f"\nO perfil acrescenta em {int(t['acrescenta'].sum())} de {len(t)}.")
    return res


def main():
    p = argparse.ArgumentParser(description="Busca de subgrupos nos duelos")
    p.add_argument("--reduzida", action="store_true", help="usa os 7 descritores da versão reduzida")
    args = p.parse_args()
    df = pd.read_csv(ENTRADA).sort_values(["circuito", "ano", "round"], kind="stable").reset_index(drop=True)
    descritores = REDUZIDA if args.reduzida else PRINCIPAL
    saida = Path("data/processed/subgrupos_duelos" + ("_reduzida" if args.reduzida else "") + ".csv")
    rng = np.random.default_rng(SEMENTE)
    print(f"Base: {len(df)} voltas de duelo, {df.groupby(['ano', 'round']).ngroups} corridas, "
          f"{df['circuito'].nunique()} circuitos\n")
    parte1(df, rng)
    parte2(df, descritores, rng, saida)


if __name__ == "__main__":
    main()
