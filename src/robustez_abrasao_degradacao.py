"""
Etapa 2 - Robustez do efeito da abrasão sobre a degradação, e busca corrigida.

Entra: data/processed/base_stint.csv
       data/processed/pirelli_agregado_por_circuito.csv
Sai:   data/processed/subgrupos_degradacao_corrigida.csv

Não usa internet.

Em busca_subgrupos_degradacao.py, a abrasão explicou 5,2% da degradação acima do
esperado para o pneu (p = 0,0008), e o perfil da Etapa 1 não (p = 0,241). A busca de
subgrupos não aprovou nada porque o limiar foi definido por subgrupos de 1 ou 2
circuitos, que a regra nunca deixaria ser firmes.

PARTE A. Robustez da abrasão. Regra fixada antes de rodar: o efeito é FIRME se todas
valerem, e FRÁGIL caso contrário.
  1. Abrasão alta acima de baixa nas duas eras.
  2. p < 0,05 com as suposições de combustível de 0,02 e 0,04 s/kg.
  3. p < 0,05 retirando cada um dos 26 circuitos, um de cada vez (pior caso).
  4. Abrasão alta acima de baixa em cada composto (macio, médio, duro).
  5. p < 0,05 com a nota de abrasão da Pirelli sem faixas (Spearman entre a nota e a
     degradação média de cada circuito), para o resultado não depender dos cortes.
  Todos os p vêm de embaralhar as características entre os 26 circuitos.

PARTE B. Busca corrigida, DEFINIDA DEPOIS DE VER A BUSCA ORIGINAL; relatar assim.
  Mesma busca, mas só concorrem subgrupos com 4+ circuitos e nenhum circuito acima de
  40% dos stints, e a mesma exigência vale dentro de cada permutação. Assim o limiar
  passa a vir só de subgrupos que poderiam ser firmes.
  Firme: passa do limiar, mesmo sinal nas duas eras, mesmo sinal com 0,02 e 0,04 s/kg,
  mesmo sinal sem o circuito mais frequente.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pysubgroup as ps

BASE = Path("data/processed/base_stint.csv")
PIRELLI = Path("data/processed/pirelli_agregado_por_circuito.csv")
SAIDA = Path("data/processed/subgrupos_degradacao_corrigida.csv")

NOTA = "#7_asphalt_abrasion_1_5"
DO_STINT = ["d_composto", "d_pneu_novo", "d_ordem_stint"]
DO_CIRCUITO = ["d_perfil", "d_abrasao", "d_estresse_pneu", "d_lateral"]
DESCRITORES = DO_STINT + ["d_temp_pista"] + DO_CIRCUITO
SUPORTE, PROFUNDIDADE = 50, 3
PERMUTACOES, SEMENTE, LIMITE = 1000, 13, 0.05
MIN_CIRCUITOS, MAX_CIRCUITO = 4, 0.40
FMT = lambda x: f"{x:+.3f}"


def ajusta(df, coluna):
    """Degradação acima do esperado para aquele pneu."""
    return df[coluna] - df.groupby(DO_STINT)[coluna].transform("mean")


class TesteCircuito:
    """p de uma característica do circuito, embaralhando-a entre os circuitos.
    Usa só somas por circuito, então cada repetição é rápida."""

    def __init__(self, df, coluna_y, rng):
        r = ajusta(df, coluna_y)
        g = r.groupby(df["circuito"]).agg(["sum", "size"])
        self.circuitos = g.index.to_numpy()
        self.soma, self.n = g["sum"].to_numpy(), g["size"].to_numpy()
        self.media_circuito = self.soma / self.n
        self.total = ((r - r.mean()) ** 2).sum()
        self.rng = rng

    def parcela(self, rotulos):
        s = pd.DataFrame({"r": rotulos, "s": self.soma, "n": self.n}).groupby("r").sum()
        geral = self.soma.sum() / self.n.sum()
        return ((s["s"] / s["n"] - geral) ** 2 * s["n"]).sum() / self.total

    def p(self, rotulos, repeticoes=PERMUTACOES):
        obs = self.parcela(rotulos)
        maiores = sum(self.parcela(self.rng.permutation(rotulos)) >= obs
                      for _ in range(repeticoes))
        return obs, (maiores + 1) / (repeticoes + 1)

    def p_spearman(self, nota, repeticoes=PERMUTACOES):
        x = pd.Series(nota).rank().to_numpy()
        y = pd.Series(self.media_circuito).rank().to_numpy()
        obs = np.corrcoef(x, y)[0, 1]
        maiores = sum(abs(np.corrcoef(self.rng.permutation(x), y)[0, 1]) >= abs(obs)
                      for _ in range(repeticoes))
        return obs, (maiores + 1) / (repeticoes + 1)


def rotulos_abrasao(df, circuitos):
    mapa = df.drop_duplicates("circuito").set_index("circuito")["d_abrasao"]
    return mapa.reindex(circuitos).to_numpy()


def alta_menos_baixa(df, coluna="deg_pct"):
    r = ajusta(df, coluna)
    m = r.groupby(df["d_abrasao"]).mean()
    return m.get("alta", np.nan) - m.get("baixa", np.nan)


def parte_a(df, rng):
    print("=" * 78 + "\nPARTE A. ROBUSTEZ DA ABRASÃO\n" + "=" * 78)
    t = TesteCircuito(df, "deg_pct", rng)
    obs, p = t.p(rotulos_abrasao(df, t.circuitos))
    print(f"Referência: abrasão explica {obs:.1%}, p = {p:.4f}, "
          f"alta − baixa = {alta_menos_baixa(df):+.3f}\n")

    eras = {e: alta_menos_baixa(g) for e, g in df.groupby("d_era")}
    print("1. Alta − baixa por era: " + "  ".join(f"{e} {v:+.3f}" for e, v in eras.items()))

    comb = {}
    for c in ["deg_pct_baixo", "deg_pct_alto"]:
        tc = TesteCircuito(df, c, rng)
        comb[c] = tc.p(rotulos_abrasao(df, tc.circuitos))[1]
    print("2. Combustível: " + "  ".join(f"{c} p = {v:.4f}" for c, v in comb.items()))

    sem = {}
    for c in sorted(df["circuito"].unique()):
        d = df[df["circuito"] != c]
        tc = TesteCircuito(d, "deg_pct", rng)
        sem[c] = tc.p(rotulos_abrasao(d, tc.circuitos), 500)[1]
    sem = pd.Series(sem).sort_values(ascending=False)
    print("3. Retirando cada circuito, os 5 piores:")
    print("   " + "  ".join(f"{c} {v:.4f}" for c, v in sem.head(5).items()))

    comp = {c: alta_menos_baixa(g) for c, g in df.groupby("d_composto")}
    print("4. Alta − baixa por composto: " + "  ".join(f"{c} {v:+.3f}" for c, v in comp.items()))

    nota = pd.read_csv(PIRELLI).set_index("circuito")[NOTA].reindex(t.circuitos)
    rho, p_nota = t.p_spearman(nota.to_numpy())
    print(f"5. Nota sem faixas: Spearman {rho:+.2f} entre {len(nota)} circuitos, p = {p_nota:.4f}")

    print("\nAbrasão por circuito (degradação acima do esperado para o pneu):")
    tab = pd.DataFrame({"nota_abrasao": nota, "faixa": rotulos_abrasao(df, t.circuitos),
                        "stints": t.n, "acima_do_esperado": t.media_circuito}, index=t.circuitos)
    print(tab.sort_values("acima_do_esperado", ascending=False).to_string(float_format=FMT))

    itens = [("Mesmo sentido nas duas eras", all(v > 0 for v in eras.values())),
             ("Combustível 0,02 e 0,04 s/kg", all(v < LIMITE for v in comb.values())),
             ("Retirando cada circuito", sem.iloc[0] < LIMITE),
             ("Mesmo sentido em cada composto", all(v > 0 for v in comp.values())),
             ("Nota sem faixas", p_nota < LIMITE and rho > 0)]
    print("\nREGRA")
    for i, (nome, ok) in enumerate(itens, 1):
        print(f"{i}. {'PASSOU' if ok else 'NÃO PASSOU'}  {nome}")
    firme = all(ok for _, ok in itens)
    print(f"\nConclusão da parte A: {'FIRME' if firme else 'FRÁGIL'}.\n")


def parte_b(df, rng):
    print("=" * 78 + "\nPARTE B. BUSCA CORRIGIDA (definida depois de ver a original)\n" + "=" * 78)
    for c, aj in [("deg_pct", "aj"), ("deg_pct_baixo", "aj_baixo"), ("deg_pct_alto", "aj_alto")]:
        df[aj] = ajusta(df, c)
    busca = df[DESCRITORES + ["aj"]].copy()
    tarefa = ps.SubgroupDiscoveryTask(
        busca, ps.NumericTarget("aj"), ps.create_selectors(busca, ignore=["aj"]),
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

    circ_codigo, circ_nomes = pd.factorize(df["circuito"])
    um_circuito = np.zeros((len(df), len(circ_nomes)), dtype=np.float32)
    um_circuito[np.arange(len(df)), circ_codigo] = 1

    def avalia(colunas, y):
        m = np.ones((len(condicoes) + 1, len(df)), dtype=bool)
        for i, (c, v) in enumerate(condicoes):
            m[i] = colunas[c] == v
        cobre = m[idx[:, 0]] & m[idx[:, 1]] & m[idx[:, 2]]
        contagem = cobre.astype(np.float32) @ um_circuito
        n = contagem.sum(axis=1)
        elegivel = ((n >= SUPORTE) & ((contagem > 0).sum(axis=1) >= MIN_CIRCUITOS)
                    & (contagem.max(axis=1) <= MAX_CIRCUITO * np.maximum(n, 1)))
        z = np.zeros(len(grupos))
        soma = cobre.astype(float) @ (y - y.mean())
        z[elegivel] = soma[elegivel] / np.sqrt(n[elegivel]) / y.std(ddof=1)
        return cobre, z, elegivel

    originais = {c: df[c].to_numpy() for c in DESCRITORES}
    y = df["aj"].to_numpy(dtype=float)
    cobre, zs, elegivel = avalia(originais, y)

    df["corrida"] = df["ano"].astype(str) + "_" + df["round"].astype(str)
    tabela = df.drop_duplicates("circuito").set_index("circuito")[DO_CIRCUITO]
    circuitos = tabela.index.to_numpy()
    por_corrida = list(df.groupby("corrida").indices.values())
    temp = df.drop_duplicates("corrida").set_index("corrida")[["circuito", "d_temp_pista"]]
    maximos = np.empty(PERMUTACOES)
    for i in range(PERMUTACOES):
        v = y.copy()
        for linhas in por_corrida:
            v[linhas] = y[rng.permutation(linhas)]
        novo = pd.DataFrame(tabela.to_numpy(), columns=DO_CIRCUITO,
                            index=rng.permutation(circuitos)).reindex(df["circuito"])
        nova_temp = temp.groupby("circuito")["d_temp_pista"].transform(
            lambda s: rng.permutation(s.to_numpy()))
        colunas = dict(originais)
        for c in DO_CIRCUITO:
            colunas[c] = novo[c].to_numpy()
        colunas["d_temp_pista"] = df["corrida"].map(nova_temp).to_numpy()
        maximos[i] = np.abs(avalia(colunas, v)[1]).max()
    limiar = np.quantile(maximos, 0.95)

    media = y.mean()
    linhas = []
    for k in np.where(elegivel)[0]:
        g, sub = grupos[k], df[cobre[k]]
        s = np.sign(sub["aj"].mean() - media)
        n = sub["circuito"].value_counts()
        sem = sub[sub["circuito"] != n.index[0]]
        eras = sub.groupby("d_era")["aj"].mean()
        ok = (len(eras) == 2 and all(np.sign(eras - media) == s)
              and all(np.sign(sub[c].mean() - df[c].mean()) == s for c in ["aj_baixo", "aj_alto"])
              and len(sem) > 0 and np.sign(sem["aj"].mean() - media) == s)
        linhas.append({"descricao": " E ".join(f"{c[2:]}={v}" for c, v in g.items()),
                       "stints": len(sub), "circuitos": len(n),
                       "maior_circuito_pct": n.iloc[0] / len(sub),
                       "acima_do_esperado": sub["aj"].mean() - media, "z": zs[k],
                       "acima_do_limiar": abs(zs[k]) > limiar,
                       "firme": abs(zs[k]) > limiar and ok,
                       "tem_circuito": any(c in DO_CIRCUITO for c in g),
                       "tem_perfil": "d_perfil" in g,
                       "media_2018_21": eras.get("2018-21", np.nan),
                       "media_2022_mais": eras.get("2022+", np.nan)})
    res = pd.DataFrame(linhas).sort_values("z", key=abs, ascending=False)
    firmes = res[res["firme"]]
    print(f"Subgrupos: {len(grupos)}, elegíveis (4+ circuitos, nenhum acima de 40%): {len(res)}")
    print(f"Limiar |z| (5%, {PERMUTACOES} permutações): {limiar:.2f}")
    print(f"Acima do limiar: {int(res['acima_do_limiar'].sum())}   firmes: {len(firmes)}")
    print(f"  firmes com condição do circuito: {int(firmes['tem_circuito'].sum())}")
    print(f"  firmes com condição de perfil:   {int(firmes['tem_perfil'].sum())}\n")
    cols = ["descricao", "stints", "circuitos", "acima_do_esperado", "z", "firme"]
    print(res.head(15)[cols].to_string(index=False, float_format=FMT, max_colwidth=62))
    if len(firmes):
        print("\nFirmes, com as duas eras:")
        print(firmes.head(15)[["descricao", "stints", "acima_do_esperado",
                               "media_2018_21", "media_2022_mais"]].to_string(
            index=False, float_format=FMT, max_colwidth=62))
    res.to_csv(SAIDA, index=False)
    print(f"\nSalvo em {SAIDA}")


def main():
    df = pd.read_csv(BASE).dropna(subset=DESCRITORES + ["deg_pct"]).reset_index(drop=True)
    rng = np.random.default_rng(SEMENTE)
    print(f"Base: {len(df)} stints, {df['circuito'].nunique()} circuitos\n")
    parte_a(df, rng)
    parte_b(df, rng)


if __name__ == "__main__":
    main()
