"""
Etapa 2 - Monta a tabela de análise que a busca de subgrupos vai consumir.

Entra:  data/processed/base_piloto_corrida.csv   (uma linha por piloto por corrida)
Sai:    data/processed/base_analise_sd.csv       (o alvo + os descritores, pronta)

Não usa internet, roda em segundos, pode rodar quantas vezes quiser.

O que ele faz, em ordem:

1. Filtra a amostra da H2: só quem terminou, fora as corridas sem disputa efetiva.
2. Calcula o alvo = referência do piloto - posição final, com a referência móvel
   (janela de 11 corridas vizinhas da mesma temporada, a própria corrida fora).
   Calcula também a versão com referência de temporada, para verificação de
   robustez - nunca como alvo principal.
3. Monta os descritores em faixas, todos com prefixo `d_`.
4. Confere e salva.

Convenção do prefixo: na configuração da busca, os descritores são
`[c for c in df.columns if c.startswith("d_")]`. As colunas de identificação
(ano, round, circuito, piloto, equipe) ficam na tabela para conferência e NÃO
entram no espaço de busca - circuito tornaria a tipologia inútil, piloto trava
em 13 corridas no perfil urbano-técnico.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ENTRADA = Path("data/processed/base_piloto_corrida.csv")
SAIDA = Path("data/processed/base_analise_sd.csv")

JANELA_MOVEL = 11      # corridas vizinhas, centradas na corrida atual
MIN_VIZINHAS = 4       # abaixo disso a referência não é confiável

NOMES_PERFIL = {
    0: "alto downforce",
    1: "stop-and-go",
    2: "potência",
    3: "urbano-técnico",
}

base = pd.read_csv(ENTRADA)
print(f"Base completa: {len(base)} linhas")


# ---------------------------------------------------------------------------
# 1. A amostra da H2
# ---------------------------------------------------------------------------
# Abandono não tem posição: tratá-lo como último lugar faria quebra de motor
# virar antipatia pela pista.
# Spa 2021 (e qualquer corrida parecida que a regra pegue) sai porque a chegada
# foi a ordem do grid - as linhas são ruído e ainda distorceriam a referência
# das corridas vizinhas.
amostra = base[base["terminou"] & ~base["corrida_sem_disputa"]].copy()
amostra = amostra.sort_values(["piloto", "ano", "round"])

print(f"  - não terminaram:        {int((~base['terminou']).sum())}")
print(f"  - corrida sem disputa:   {int(base['corrida_sem_disputa'].sum())}")
print(f"Amostra da H2: {len(amostra)} linhas\n")


# ---------------------------------------------------------------------------
# 1b. A posição numa escala fixa
# ---------------------------------------------------------------------------
# Excluir quem abandonou cria um viés: numa corrida em que 14 carros terminam
# as posições vão de 1 a 14 (média 7,5), numa em que terminam 20 vão de 1 a 20
# (média 10,5). Como o piloto é comparado com uma referência construída sobre
# corridas de todo tipo, sobreviver a uma hecatombe parecia desempenho.
#
# Medido: o alvo médio da corrida contra o número de finalistas dava
# rho = -0,827 (162 corridas). Em faixas, +1,43 nas corridas de até 14
# sobreviventes contra -0,90 nas de 19 ou mais. Como a taxa de abandono varia
# com o perfil do circuito, o viés contaminava justamente a comparação que é o
# objeto do trabalho.
#
# A correção traduz a posição para um grid de tamanho fixo: o vencedor continua
# em 1, o último finalista vira 20, independente de quantos sobreviveram.
# Resultado: rho vai a +0,224 e a amplitude entre perfis cai pela metade.
# Custo: o desvio do alvo sobe ~12%, porque esticar um pelotão de 14 para a
# escala de 20 multiplica as distâncias por 1,46.
GRID_PADRAO = 20

amostra["n_terminaram"] = amostra.groupby(["ano", "round"])["piloto"].transform("size")
# rank em vez de usar posicao_final direto: não depende de os finalistas
# ocuparem posições 1..N contíguas
amostra["pos"] = amostra.groupby(["ano", "round"])["posicao_final"].rank(method="min")
amostra["pos_norm"] = (
    1 + (amostra["pos"] - 1) * (GRID_PADRAO - 1) / (amostra["n_terminaram"] - 1)
)


# ---------------------------------------------------------------------------
# 2. O alvo
# ---------------------------------------------------------------------------
def referencias(g):
    """As referências do piloto naquela temporada.

    A própria corrida fica fora de todas: se entrasse, cada corrida puxaria a
    própria referência na direção dela mesma, encolhendo o alvo - e mais para
    quem correu menos, bem onde os dados já são frágeis.
    """
    y = g["pos_norm"].astype(float)
    n = len(y)

    # móvel: a vizinhança no calendário, que absorve o desenvolvimento do carro
    soma = y.rolling(JANELA_MOVEL, center=True, min_periods=MIN_VIZINHAS).sum()
    cont = y.rolling(JANELA_MOVEL, center=True, min_periods=MIN_VIZINHAS).count()
    g["ref_movel"] = (soma - y) / (cont - 1)

    # temporada inteira: verificação de robustez da janela
    g["ref_temporada"] = (y.sum() - y) / (n - 1) if n > 1 else np.nan

    # móvel sobre a posição bruta: verificação de robustez da normalização
    yb = g["pos"].astype(float)
    sb = yb.rolling(JANELA_MOVEL, center=True, min_periods=MIN_VIZINHAS).sum()
    cb = yb.rolling(JANELA_MOVEL, center=True, min_periods=MIN_VIZINHAS).count()
    g["ref_pos_bruta"] = (sb - yb) / (cb - 1)
    return g


amostra = (
    amostra.groupby(["piloto", "ano"], group_keys=False)
    .apply(referencias, include_groups=False)
    .join(amostra[["piloto", "ano"]])
)

# Positivo = foi melhor do que costuma ir. A unidade é posição num grid de 20.
amostra["alvo"] = amostra["ref_movel"] - amostra["pos_norm"]
# As duas verificações de robustez. Nunca alvo principal.
amostra["alvo_temporada"] = amostra["ref_temporada"] - amostra["pos_norm"]
amostra["alvo_pos_bruta"] = amostra["ref_pos_bruta"] - amostra["pos"]


# ---------------------------------------------------------------------------
# 3. Os descritores
# ---------------------------------------------------------------------------
amostra["d_perfil"] = amostra["cluster"].map(NOMES_PERFIL)


def faixa_paradas(n):
    if pd.isna(n):
        return np.nan
    if n <= 1:
        return "até 1 parada"
    if n == 2:
        return "2 paradas"
    return "3+ paradas"


amostra["d_paradas"] = amostra["n_paradas"].map(faixa_paradas)

# Momento da primeira parada: cortado pelos terços da distribuição OBSERVADA,
# não por terços da distância da corrida.
# Com terços da distância, "parou tarde" tinha 31/65/16/15 linhas nos quatro
# perfis - quase nada, porque a primeira parada praticamente nunca cai no terço
# final. Cortando pela distribuição, as três faixas ficam com peso parecido.
# Os cortes são globais, não por perfil: se cada perfil tivesse os seus, "parou
# cedo" significaria coisas diferentes em cada um e a comparação entre perfis
# deixaria de fazer sentido.
q1, q2 = amostra["momento_primeira_parada"].quantile([1 / 3, 2 / 3])


def faixa_momento(m):
    if pd.isna(m):
        return np.nan
    if m < q1:
        return "parada antecipada"
    if m < q2:
        return "parada intermediária"
    return "parada tardia"


amostra["d_momento"] = amostra["momento_primeira_parada"].map(faixa_momento)

# Dureza relativa à escolha daquele fim de semana, não o nome do composto:
# a Pirelli leva três de uma escala de cinco por evento, então o "hard" de uma
# corrida pode ser mais macio que o "medium" de outra.
amostra["d_composto"] = amostra["composto_largada_rel"].map(
    lambda c: f"largou {c}" if pd.notna(c) else np.nan
)


def faixa_grid(g):
    # grid 0 é largada do pit lane, não posição zero
    if pd.isna(g) or g == 0:
        return np.nan
    if g <= 5:
        return "largou top 5"
    if g <= 10:
        return "largou 6-10"
    return "largou 11+"


amostra["d_grid"] = amostra["grid"].map(faixa_grid)

amostra["d_chuva"] = amostra["faixa_chuva"]
amostra["d_safety_car"] = np.where(
    amostra["n_periodos_sc"] > 0, "com safety car", "sem safety car"
)

DESCRITORES = [c for c in amostra.columns if c.startswith("d_")]


# ---------------------------------------------------------------------------
# 4. Linhas incompletas
# ---------------------------------------------------------------------------
print("Vazios na amostra:")
faltando = amostra[["alvo"] + DESCRITORES].isna().sum()
faltando = faltando[faltando > 0]
print(faltando.to_string() if len(faltando) else "  nenhum")

antes = len(amostra)
final = amostra.dropna(subset=["alvo"] + DESCRITORES).copy()
print(f"\nLinhas descartadas por estarem incompletas: {antes - len(final)}")
print(f"Tabela final: {len(final)} linhas")

COLUNAS = (
    # identificação - para conferência, NÃO entram no espaço de busca
    ["ano", "round", "data", "circuito", "piloto", "equipe", "cluster"]
    # o alvo e suas peças
    + ["alvo", "alvo_temporada", "alvo_pos_bruta",
       "posicao_final", "pos_norm", "n_terminaram", "ref_movel", "ref_temporada"]
    # os descritores
    + DESCRITORES
    # cru, para quem quiser refazer uma faixa sem remontar tudo
    + ["grid", "n_paradas", "momento_primeira_parada", "composto_largada_rel",
       "pct_tempo_chuva", "n_periodos_sc", "voltas_corrida"]
)
final = final[COLUNAS].sort_values(["ano", "round", "posicao_final"])


# ---------------------------------------------------------------------------
# 5. Conferência
# ---------------------------------------------------------------------------
print("\n" + "=" * 64)
print("CONFERÊNCIA")
print("=" * 64)

print(f"Linhas: {len(final)}   Corridas: {final.groupby(['ano','round']).ngroups}"
      f"   Pilotos: {final['piloto'].nunique()}")

print(f"\nO alvo (posições acima/abaixo do próprio padrão):")
a = final["alvo"]
print(f"  média {a.mean():+.3f}   mediana {a.median():+.3f}   desvio {a.std():.2f}")
print(f"  faixa {a.min():+.1f} a {a.max():+.1f}")
print(f"  assimetria {a.skew():+.3f}  (perto de 0 = simétrico)")
print(f"  correlação com a versão de temporada: "
      f"{final['alvo'].corr(final['alvo_temporada']):.3f}")
print(f"  correlação com a versão de posição bruta: "
      f"{final['alvo'].corr(final['alvo_pos_bruta']):.3f}")

# O viés de abandono tinha que ter sumido: o alvo médio da corrida não pode
# andar com quantos carros terminaram.
por_corrida = final.groupby(["ano", "round"]).agg(
    n_terminaram=("n_terminaram", "first"), alvo=("alvo", "mean")
)
print(f"\nViés de abandono (alvo médio da corrida x nº de finalistas):")
print(f"  antes da correção o rho era -0,827")
print(f"  agora: {por_corrida['n_terminaram'].corr(por_corrida['alvo'], method='spearman'):+.3f}")

print("\nO alvo por perfil de circuito (média tem que ficar perto de zero -")
print("o alvo é relativo ao próprio piloto, não ao perfil):")
print(final.groupby("d_perfil")["alvo"].agg(
    linhas="size", media="mean", desvio="std"
).to_string(float_format=lambda x: f"{x:+.3f}"))

print(f"\nCortes do momento da parada: {q1:.3f} e {q2:.3f} da distância")
print("(fração da corrida em que a primeira parada aconteceu)")

print("\nOs descritores:")
for d in DESCRITORES:
    print(f"\n  {d}")
    print(final[d].value_counts().to_string().replace("\n", "\n    ").rjust(4))

final.to_csv(SAIDA, index=False)
print(f"\n\nSalvo em {SAIDA}")
print(f"Descritores ({len(DESCRITORES)}): {', '.join(DESCRITORES)}")
print("\nNenhum valor do alvo dentro de nenhum subgrupo foi examinado aqui.")