"""
Etapa 2 - Medição dos dois descritores que dominaram a primeira busca.

A primeira rodada devolveu 23 dos 30 subgrupos contendo `largou top 5`, e
`até 1 parada` logo atrás. Nenhum dos dois é estratégia:

  largada    a referência do piloto é a posição média de chegada dele - ela
             não sabe de onde ele largou. Largar melhor que o próprio normal
             implica terminar melhor que o próprio normal, por construção.
             Isso mede o sábado.

  paradas    parada extra muitas vezes não é escolha: é furo, asa quebrada,
             consequência de um problema. E o número absoluto de paradas não
             é comparável entre corridas - duas paradas em Monza e duas em
             Barcelona são decisões diferentes, do mesmo jeito que "medium"
             significava coisas diferentes antes da correção de composto.

Este script mede os dois antes de mexer em qualquer coisa, e simula as
correções para ver se funcionam. Não altera o pipeline nem salva base nova.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ENTRADA = Path("data/processed/base_analise_sd.csv")

df = pd.read_csv(ENTRADA)
desvio = df["alvo"].std()
print(f"Base: {len(df)} linhas, alvo com desvio {desvio:.2f}\n")


# ===========================================================================
# A. A LARGADA
# ===========================================================================
print("=" * 70)
print("A. O EFEITO DA POSIÇÃO DE LARGADA")
print("=" * 70)

por_grid = df.groupby("grid").agg(
    linhas=("alvo", "size"),
    alvo_medio=("alvo", "mean"),
    alvo_mediano=("alvo", "median"),
).round(3)
por_grid["erro_padrao"] = (
    df.groupby("grid")["alvo"].std() / np.sqrt(por_grid["linhas"])
).round(3)

print("Alvo médio por posição de largada:\n")
print(por_grid.to_string())

rho, p = spearmanr(df["grid"], df["alvo"])
print(f"\nCorrelação largada x alvo: rho = {rho:+.3f} (p = {p:.2g})")

# Quanto da variação do alvo a largada sozinha explica
media_por_grid = df.groupby("grid")["alvo"].transform("mean")
var_explicada = 1 - ((df["alvo"] - media_por_grid).var() / df["alvo"].var())
print(f"Variação do alvo explicada só pela largada: {var_explicada:.1%}")

amplitude = por_grid["alvo_medio"].max() - por_grid["alvo_medio"].min()
print(f"Amplitude entre a melhor e a pior largada: {amplitude:.2f} posições")
print(f"(o desvio do alvo inteiro é {desvio:.2f} - para comparar)")


# --- a correção: subtrair o que normalmente acontece a partir dali ----------
print("\n--- Simulando a correção ---")
print("alvo_ajustado = alvo - (alvo médio de quem larga daquela posição)")
print("Isso responde: dado de onde ele largou, foi melhor do que costuma")
print("ser de lá? O teto da H1 original fica embutido, porque quem larga em")
print("1º já tem expectativa baixa e ela é descontada.\n")

df["alvo_ajustado"] = df["alvo"] - media_por_grid

rho2, p2 = spearmanr(df["grid"], df["alvo_ajustado"])
print(f"Correlação largada x alvo ajustado: rho = {rho2:+.3f} (p = {p2:.2g})")
print(f"Desvio do alvo ajustado: {df['alvo_ajustado'].std():.2f} "
      f"(era {desvio:.2f})")
print(f"Correlação entre as duas versões do alvo: "
      f"{df['alvo'].corr(df['alvo_ajustado']):.3f}")

print("\nO que a correção faz com as faixas de largada:")
print(df.groupby("d_grid").agg(
    linhas=("alvo", "size"),
    antes=("alvo", "mean"),
    depois=("alvo_ajustado", "mean"),
).to_string(float_format=lambda x: f"{x:+.3f}"))

print("\nE com os perfis de circuito (não pode estragar o que já estava ok):")
print(df.groupby("d_perfil").agg(
    linhas=("alvo", "size"),
    antes=("alvo", "mean"),
    depois=("alvo_ajustado", "mean"),
).to_string(float_format=lambda x: f"{x:+.3f}"))


# ===========================================================================
# B. AS PARADAS
# ===========================================================================
print("\n" + "=" * 70)
print("B. O EFEITO DO NÚMERO DE PARADAS")
print("=" * 70)

print("Alvo médio por número absoluto de paradas:\n")
print(df.groupby("n_paradas").agg(
    linhas=("alvo", "size"),
    alvo_medio=("alvo", "mean"),
    alvo_ajustado=("alvo_ajustado", "mean"),
).to_string(float_format=lambda x: f"{x:+.3f}"))

# Quantas paradas a corrida pediu, e quanto o piloto fugiu disso.
# Mesma lógica da correção de composto: o número absoluto não é comparável
# entre corridas, a posição relativa à prática daquela corrida é.
df["paradas_da_corrida"] = df.groupby(["ano", "round"])["n_paradas"].transform("median")
df["paradas_extra"] = df["n_paradas"] - df["paradas_da_corrida"]

print(f"\nQuantas paradas cada corrida pediu (mediana do pelotão):")
print(df.groupby(["ano", "round"])["paradas_da_corrida"].first()
      .value_counts().sort_index().to_string())


def faixa_relativa(d):
    if d <= -1:
        return "menos que a corrida"
    if d < 1:
        return "no padrão da corrida"
    return "mais que a corrida"


df["paradas_rel"] = df["paradas_extra"].map(faixa_relativa)

print("\nAlvo médio por paradas RELATIVAS ao que a corrida pediu:\n")
print(df.groupby("paradas_rel").agg(
    linhas=("alvo", "size"),
    alvo_medio=("alvo", "mean"),
    alvo_ajustado=("alvo_ajustado", "mean"),
).to_string(float_format=lambda x: f"{x:+.3f}"))

print("\n>> Se 'mais que a corrida' for fortemente negativo, parada extra está")
print("   marcando problema (furo, dano), não escolha estratégica.")

# Os dois descritores lado a lado: o absoluto separa melhor ou pior que o
# relativo? Uso a variação do alvo explicada por cada um.
for nome, col in [("absoluto (até 1 / 2 / 3+)", "paradas_abs"),
                  ("relativo à corrida", "paradas_rel")]:
    m = df.groupby(col)["alvo_ajustado"].transform("mean")
    ve = 1 - ((df["alvo_ajustado"] - m).var() / df["alvo_ajustado"].var())
    print(f"\nVariação do alvo ajustado explicada por paradas {nome}: {ve:.2%}")


# ===========================================================================
# C. O que sobraria para a busca
# ===========================================================================
print("\n" + "=" * 70)
print("C. O QUE SOBRA PARA OS DESCRITORES DE ESTRATÉGIA")
print("=" * 70)
print("Variação do alvo ajustado explicada por cada descritor sozinho.")
print("Com o alvo atual, largada e paradas abafavam o resto.\n")

for d in [c for c in df.columns if c.startswith("d_")] + ["paradas_rel"]:
    m = df.groupby(d)["alvo_ajustado"].transform("mean")
    ve = 1 - ((df["alvo_ajustado"] - m).var() / df["alvo_ajustado"].var())
    m0 = df.groupby(d)["alvo"].transform("mean")
    ve0 = 1 - ((df["alvo"] - m0).var() / df["alvo"].var())
    print(f"  {d:<16} alvo atual {ve0:>6.2%}   alvo ajustado {ve:>6.2%}")

print("\nNada foi salvo.")