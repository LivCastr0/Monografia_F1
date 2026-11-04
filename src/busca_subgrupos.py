"""
Etapa 2 - Primeira rodada da busca de subgrupos.

Pergunta: quais escolhas de estratégia, em que perfil de circuito e sob quais
condições, acompanham desempenho acima do padrão do próprio piloto?

Entra: data/processed/base_analise_sd.csv
Sai:   data/processed/subgrupos_rodada1.csv

Configuração, toda fixada antes de olhar resultado:
  alvo              `alvo` (posições acima/abaixo do padrão do próprio piloto)
  descritores       as 7 colunas com prefixo d_
  medida            StandardQFNumeric, baseada em média (a=0.5)
  profundidade      até 3 condições
  suporte mínimo    50 linhas (número único para a base inteira)

O que este script NÃO faz: julgar se um subgrupo é interessante. Ele mede e
reporta. A leitura dos achados vem depois, e com as ressalvas de sempre -
descoberta de subgrupos descreve associação, não causa, e estratégia é
escolhida durante a corrida em função de como ela está indo.
"""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pysubgroup as ps

ENTRADA = Path("data/processed/base_analise_sd.csv")
SAIDA = Path("data/processed/subgrupos_rodada1.csv")

SUPORTE_MINIMO = 50
PROFUNDIDADE = 3
QUANTOS_RESULTADOS = 30
PESO_TAMANHO = 0.5     # o `a` da medida: 0 ignora tamanho, 1 pesa linearmente

df = pd.read_csv(ENTRADA)
descritores = [c for c in df.columns if c.startswith("d_")]

print(f"Base: {len(df)} linhas")
print(f"Descritores ({len(descritores)}): {', '.join(descritores)}")

desvio = df["alvo"].std()
print(f"Alvo: média {df['alvo'].mean():+.3f}, desvio {desvio:.2f}")


def efeito_detectavel(n):
    """Quanto um subgrupo de n linhas precisa se afastar de zero para não ser
    ruído. Dois erros-padrão, na unidade do alvo (posições)."""
    return 2 * desvio / np.sqrt(n)


print(f"\nEfeito detectável por tamanho de subgrupo:")
for n in [50, 100, 200, 400, 800]:
    print(f"  {n:>4} linhas -> {efeito_detectavel(n):.2f} posição")


# ---------------------------------------------------------------------------
# A busca
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("BUSCA")
print("=" * 72)

busca_df = df[descritores + ["alvo"]].copy()

alvo = ps.NumericTarget("alvo")
espaco = ps.create_selectors(busca_df, ignore=["alvo"])
print(f"Espaço de busca: {len(espaco)} seletores")

tarefa = ps.SubgroupDiscoveryTask(
    busca_df,
    alvo,
    espaco,
    result_set_size=QUANTOS_RESULTADOS,
    depth=PROFUNDIDADE,
    qf=ps.StandardQFNumeric(a=PESO_TAMANHO),
    constraints=[ps.MinSupportConstraint(SUPORTE_MINIMO)],
)

resultado = ps.Apriori().execute(tarefa)
achados = resultado.to_descriptions()
print(f"Subgrupos retornados: {len(achados)}\n")


# ---------------------------------------------------------------------------
# As estatísticas de cada subgrupo, calculadas aqui
# ---------------------------------------------------------------------------
# Calculo em vez de ler do to_dataframe() do pysubgroup porque quero a mediana
# (verificação de robustez contra a cauda do alvo) e o número médio de
# finalistas (verificação do resíduo do viés de abandono).
linhas = []
coberturas = {}

for i, (qualidade, sg) in enumerate(achados):
    cobre = sg.covers(busca_df)
    coberturas[i] = cobre
    sub = df[cobre]
    n = len(sub)
    media = sub["alvo"].mean()

    linhas.append({
        "i": i,
        "descricao": str(sg),
        "condicoes": str(sg).count("AND") + 1,
        "linhas": n,
        "media": media,
        "mediana": sub["alvo"].median(),
        # robustez: o achado sobrevive às versões alternativas do alvo?
        "media_ref_temporada": sub["alvo_temporada"].mean(),
        "media_pos_bruta": sub["alvo_pos_bruta"].mean(),
        # o efeito supera o ruído do próprio tamanho?
        "limiar": efeito_detectavel(n),
        "acima_do_ruido": abs(media) > efeito_detectavel(n),
        # resíduo do viés de abandono: o subgrupo é feito de corridas de
        # muito abandono? (média geral da base para comparar)
        "finalistas_medio": sub["n_terminaram"].mean(),
        # o padrão é do perfil ou de um circuito só com fantasia de perfil?
        "circuitos": sub["circuito"].nunique(),
        "maior_circuito_pct": sub["circuito"].value_counts(normalize=True).iloc[0],
        "qualidade": qualidade,
    })

res = pd.DataFrame(linhas)

print("Os subgrupos, em ordem de qualidade:")
print(res[["descricao", "linhas", "media", "mediana", "limiar",
           "acima_do_ruido"]].to_string(
    index=False, float_format=lambda x: f"{x:+.2f}", max_colwidth=58))


# ---------------------------------------------------------------------------
# Redundância: quantos destes são o mesmo subgrupo com outro nome?
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("REDUNDÂNCIA")
print("=" * 72)
print("Pares que cobrem quase as mesmas linhas (Jaccard > 0,8).")
print("Esperado numa primeira rodada: muitos. É configuração, não achado.\n")

pares_redundantes = []
for a, b in combinations(range(len(achados)), 2):
    ca, cb = coberturas[a], coberturas[b]
    uniao = (ca | cb).sum()
    if uniao and (ca & cb).sum() / uniao > 0.8:
        pares_redundantes.append((a, b))

print(f"Pares redundantes: {len(pares_redundantes)} "
      f"de {len(achados) * (len(achados) - 1) // 2} possíveis")
for a, b in pares_redundantes[:10]:
    print(f"  [{a}] {res.loc[a, 'descricao'][:45]}")
    print(f"  [{b}] {res.loc[b, 'descricao'][:45]}\n")


# ---------------------------------------------------------------------------
# As três verificações
# ---------------------------------------------------------------------------
print("=" * 72)
print("VERIFICAÇÕES")
print("=" * 72)

print("\n1. O efeito supera o ruído do próprio tamanho?")
print(f"   {int(res['acima_do_ruido'].sum())} de {len(res)} subgrupos")

print("\n2. O achado sobrevive às versões alternativas do alvo?")
print("   (as três médias deveriam ter o mesmo sinal e ordem de grandeza)")
disc = res[
    (np.sign(res["media"]) != np.sign(res["media_ref_temporada"]))
    | (np.sign(res["media"]) != np.sign(res["media_pos_bruta"]))
]
print(f"   subgrupos que mudam de sinal em alguma versão: {len(disc)}")
if len(disc):
    print(disc[["descricao", "media", "media_ref_temporada",
                "media_pos_bruta"]].to_string(
        index=False, float_format=lambda x: f"{x:+.2f}", max_colwidth=45))

print("\n3. É padrão de perfil ou de um circuito só?")
print(f"   média de finalistas na base: {df['n_terminaram'].mean():.1f}")
concentrados = res[res["maior_circuito_pct"] > 0.5]
print(f"   subgrupos em que um único circuito responde por mais de metade"
      f" das linhas: {len(concentrados)}")
if len(concentrados):
    print(concentrados[["descricao", "linhas", "circuitos",
                        "maior_circuito_pct"]].to_string(
        index=False, float_format=lambda x: f"{x:.2f}", max_colwidth=45))

res.to_csv(SAIDA, index=False)
print(f"\n\nSalvo em {SAIDA}")