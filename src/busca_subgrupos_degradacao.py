"""
Etapa 2 - Busca de subgrupos sobre a degradação de pneu.

Pergunta: em que combinações de pneu, condição de corrida e característica do
circuito o pneu se degrada mais rápido ou mais devagar?

Entra: data/processed/base_stint.csv   (monta_base_stint.py)
Sai:   data/processed/subgrupos_degradacao.csv

Alvo: deg_pct, a piora do tempo de volta por volta de vida do pneu, em % do tempo
típico do stint.

A busca usa a degradação ACIMA DO ESPERADO PARA AQUELE PNEU: deg_pct menos a média
dos stints com o mesmo composto, pneu novo ou usado e ordem do stint. Sem isso, o
efeito óbvio do composto (macio degrada mais) contamina tudo: num teste com dados sem
efeito nenhum, 189 subgrupos "composto E condição do circuito" passavam como firmes só
por herdarem o composto. O efeito do próprio pneu aparece numa tabela à parte.

Descritores:
  do stint     d_composto, d_pneu_novo, d_ordem_stint
  da corrida   d_temp_pista
  do circuito  d_perfil, d_abrasao, d_estresse_pneu, d_lateral
A era fica fora dos descritores, como no relatório (temporada só como diagnóstico),
e entra na verificação de robustez.

São duas partes.

1. PERFIL OU PIRELLI? Qual característica do circuito descreve melhor a degradação:
   a tipologia da Etapa 1 ou as notas da Pirelli? Primeiro desconta-se o efeito do
   composto, do pneu novo e da ordem do stint. Depois mede-se quanto cada
   característica explica, com p por permutação entre os 26 circuitos.

2. O PRÓPRIO PNEU. Médias por composto, pneu novo e ordem do stint (esperado).

3. A BUSCA. pysubgroup, StandardQFNumeric (a=0.5), profundidade 3, suporte mínimo 50
   stints, os dois sentidos. Cada repetição do teste embaralha ao mesmo tempo:
     - o alvo entre os stints da mesma corrida (testa as condições do stint);
     - as características inteiras entre os circuitos (testa o circuito);
     - a temperatura entre as corridas do mesmo circuito (testa a temperatura além
       do circuito).
   O maior valor de cada repetição define o limiar, descontando os subgrupos testados.

Firme, fixado antes de rodar. Um subgrupo é FIRME se:
  1. passa do limiar;
  2. tem o mesmo sinal nas duas eras;
  3. tem o mesmo sinal com as duas outras suposições de combustível (0,02 e 0,04 s/kg);
  4. tem 4 ou mais circuitos e nenhum circuito com mais de 40% dos stints;
  5. tem o mesmo sinal sem o circuito mais frequente.

Subgrupos só com condições do stint ficam perto de zero por construção; o que
interessa são as combinações com o circuito ou a temperatura.

Descoberta de subgrupos descreve associação, não causa.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pysubgroup as ps

ENTRADA = Path("data/processed/base_stint.csv")
SAIDA = Path("data/processed/subgrupos_degradacao.csv")

ALVO = "deg_pct"
AJUSTADO = "deg_ajustada"
DO_STINT = ["d_composto", "d_pneu_novo", "d_ordem_stint"]
DA_CORRIDA = ["d_temp_pista"]
DO_CIRCUITO = ["d_perfil", "d_abrasao", "d_estresse_pneu", "d_lateral"]
DESCRITORES = DO_STINT + DA_CORRIDA + DO_CIRCUITO
SUPORTE, PROFUNDIDADE, PESO = 50, 3, 0.5
PERMUTACOES, PERMUTACOES_PARTE1, SEMENTE = 1000, 5000, 12
MIN_CIRCUITOS, MAX_CIRCUITO = 4, 0.40
QUANTOS_MOSTRAR = 15


# ---------------------------------------------------------------------------
# 1. Perfil ou Pirelli
# ---------------------------------------------------------------------------
def parte1(df, rng):
    print("=" * 78 + "\n1. PERFIL OU PIRELLI?\n" + "=" * 78)
    resto = df[ALVO] - df.groupby(DO_STINT)[ALVO].transform("mean")
    total = ((resto - resto.mean()) ** 2).sum()
    tabela = df.drop_duplicates("circuito").set_index("circuito")[DO_CIRCUITO]
    circuitos = tabela.index.to_numpy()
    circ = df["circuito"].to_numpy()

    def parcela(rotulos):
        g = resto.groupby(rotulos).agg(["size", "mean"])
        return (g["size"] * (g["mean"] - resto.mean()) ** 2).sum() / total

    print("Depois de descontar composto, pneu novo e ordem do stint.\n")
    print(f"{'característica':<18}{'explica':>9}{'p':>9}   média por faixa")
    for c in DO_CIRCUITO:
        observado = parcela(df[c].to_numpy())
        valores = tabela[c].to_numpy()
        maiores = 0
        for _ in range(PERMUTACOES_PARTE1):
            novo = dict(zip(circuitos, rng.permutation(valores)))
            maiores += parcela(pd.Series(circ).map(novo).to_numpy()) >= observado
        p = (maiores + 1) / (PERMUTACOES_PARTE1 + 1)
        medias = resto.groupby(df[c]).mean().sort_index()
        texto = "  ".join(f"{k} {v:+.3f}" for k, v in medias.items())
        print(f"{c[2:]:<18}{observado:>8.1%}{p:>9.4f}   {texto}")
    so_circuito = parcela(circ)
    print(f"\nO circuito em si (26 categorias) explica {so_circuito:.1%}: é o teto do que "
          "qualquer característica do circuito pode explicar.\n")


# ---------------------------------------------------------------------------
# 2. A busca
# ---------------------------------------------------------------------------
def main():
    df = pd.read_csv(ENTRADA).dropna(subset=DESCRITORES + [ALVO]).reset_index(drop=True)
    df["corrida"] = df["ano"].astype(str) + "_" + df["round"].astype(str)
    rng = np.random.default_rng(SEMENTE)
    print(f"Base: {len(df)} stints, {df['corrida'].nunique()} corridas, "
          f"{df['circuito'].nunique()} circuitos")
    print(f"Degradação média {df[ALVO].mean():+.3f}% por volta, desvio {df[ALVO].std():.3f}\n")
    for origem, ajustada in [(ALVO, AJUSTADO), ("deg_pct_baixo", "aj_baixo"),
                             ("deg_pct_alto", "aj_alto")]:
        df[ajustada] = df[origem] - df.groupby(DO_STINT)[origem].transform("mean")

    parte1(df, rng)

    print("=" * 78 + "\n2. O PRÓPRIO PNEU (esperado, não é descoberta)\n" + "=" * 78)
    for c in DO_STINT:
        t = df.groupby(c)[ALVO].agg(stints="size", media="mean")
        print(t.to_string(float_format=lambda x: f"{x:+.3f}") + "\n")

    print("=" * 78 + "\n3. A BUSCA (degradação acima do esperado para o pneu)\n" + "=" * 78)
    busca = df[DESCRITORES + [AJUSTADO]].copy()
    tarefa = ps.SubgroupDiscoveryTask(
        busca, ps.NumericTarget(AJUSTADO), ps.create_selectors(busca, ignore=[AJUSTADO]),
        result_set_size=100_000, depth=PROFUNDIDADE, qf=ps.StandardQFNumeric(a=PESO),
        constraints=[ps.MinSupportConstraint(SUPORTE)])
    achados = ps.Apriori().execute(tarefa).to_descriptions()
    grupos = [{s.attribute_name: s.attribute_value for s in sg.selectors} for _, sg in achados]

    # Cada condição vira uma linha de uma matriz; cada subgrupo, até 3 linhas dela.
    condicoes = sorted({(c, v) for g in grupos for c, v in g.items()})
    pos = {cv: i for i, cv in enumerate(condicoes)}
    sempre = len(condicoes)
    indices = np.full((len(grupos), PROFUNDIDADE), sempre)
    for k, g in enumerate(grupos):
        for j, cv in enumerate(g.items()):
            indices[k, j] = pos[cv]

    def cobertura(colunas):
        m = np.ones((len(condicoes) + 1, len(df)), dtype=bool)
        for i, (c, v) in enumerate(condicoes):
            m[i] = colunas[c] == v
        cobre = m[indices[:, 0]] & m[indices[:, 1]] & m[indices[:, 2]]
        return cobre

    def z(cobre, y):
        n = cobre.sum(axis=1)
        ok = n >= SUPORTE
        soma = cobre.astype(float) @ (y - y.mean())
        resultado = np.zeros(len(n))
        resultado[ok] = soma[ok] / np.sqrt(n[ok]) / y.std(ddof=1)
        return resultado

    originais = {c: df[c].to_numpy() for c in DESCRITORES}
    cobre = cobertura(originais)
    y = df[AJUSTADO].to_numpy(dtype=float)
    zs = z(cobre, y)

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
        maximos[i] = np.abs(z(cobertura(colunas), v)).max()
    limiar = np.quantile(maximos, 0.95)

    media = y.mean()
    linhas = []
    for k, g in enumerate(grupos):
        sub = df[cobre[k]]
        s = np.sign(sub[AJUSTADO].mean() - media)
        n = sub["circuito"].value_counts()
        sem = sub[sub["circuito"] != n.index[0]]
        eras = sub.groupby("d_era")[AJUSTADO].mean()
        verif = {
            "eras": len(eras) == 2 and all(np.sign(eras - media) == s),
            "combustivel": all(np.sign(sub[c].mean() - df[c].mean()) == s
                               for c in ["aj_baixo", "aj_alto"]),
            "circuitos": len(n) >= MIN_CIRCUITOS and n.iloc[0] / len(sub) <= MAX_CIRCUITO,
            "sem_maior": len(sem) > 0 and np.sign(sem[AJUSTADO].mean() - media) == s,
        }
        passou = abs(zs[k]) > limiar
        linhas.append({
            "descricao": " E ".join(f"{c[2:]}={v}" for c, v in g.items()),
            "stints": len(sub), "corridas": sub["corrida"].nunique(), "circuitos": len(n),
            "deg_media": sub[ALVO].mean(), "acima_do_esperado": sub[AJUSTADO].mean(), "z": zs[k],
            "acima_do_limiar": passou, **{f"ok_{c}": v for c, v in verif.items()},
            "firme": passou and all(verif.values()),
            "tem_circuito": any(c in DO_CIRCUITO for c in g),
            "tem_perfil": "d_perfil" in g,
            "media_2018_21": eras.get("2018-21", np.nan), "media_2022_mais": eras.get("2022+", np.nan),
        })
    res = pd.DataFrame(linhas).sort_values("z", key=abs, ascending=False)
    firmes = res[res["firme"]]

    print(f"Subgrupos: {len(res)}   limiar |z| (5%, {PERMUTACOES} permutações): {limiar:.2f}")
    print(f"Acima do limiar: {int(res['acima_do_limiar'].sum())}   firmes: {len(firmes)}")
    print(f"  firmes com condição do circuito: {int(firmes['tem_circuito'].sum())}")
    print(f"  firmes com condição de perfil:   {int(firmes['tem_perfil'].sum())}")
    print(f"  firmes só com condições do stint: {int((~firmes['tem_circuito'] & ~firmes['descricao'].str.contains('temp_pista')).sum())}\n")
    cols = ["descricao", "stints", "circuitos", "acima_do_esperado", "z", "firme"]
    fmt = lambda x: f"{x:+.3f}"
    print("Maiores |z|:")
    print(res.head(QUANTOS_MOSTRAR)[cols].to_string(index=False, float_format=fmt, max_colwidth=60))

    com_circ = firmes[firmes["tem_circuito"]]
    if len(com_circ):
        print("\nFirmes com condição do circuito:")
        print(com_circ.head(QUANTOS_MOSTRAR)[cols + ["media_2018_21", "media_2022_mais"]]
              .to_string(index=False, float_format=fmt, max_colwidth=55))
    reprovados = res[res["acima_do_limiar"] & ~res["firme"]]
    if len(reprovados):
        print("\nAcima do limiar mas não firmes, por verificação que falhou:")
        for c in ["eras", "combustivel", "circuitos", "sem_maior"]:
            print(f"  {c}: {int((~reprovados[f'ok_{c}']).sum())}")

    res.to_csv(SAIDA, index=False)
    print(f"\nSalvo em {SAIDA}")


if __name__ == "__main__":
    main()
