"""
Etapa 2 - Teste do viés de abandono no alvo, e da correção proposta.

A suspeita: como a análise exclui quem abandonou, terminar uma corrida de muito
abandono dá alvo positivo de graça. Numa corrida em que 14 carros terminam, as
posições vão de 1 a 14 (média 7,5); numa em que terminam 20, de 1 a 20 (média
10,5). O piloto é comparado com uma referência construída sobre corridas de
todo tipo, então sobreviver a uma corrida de hecatombe parece desempenho.

Isso importa porque a taxa de abandono varia com o perfil do circuito. Se o
viés for real, um achado "circuitos de potência rendem acima do padrão" seria
carro quebrando, não estratégia.

O indício que levantou a suspeita foram as médias do alvo por perfil, ordenadas
junto com a taxa de abandono nos quatro grupos. Quatro pontos em ordem podem
ser sorte (~8%). Este script testa com 162 pontos - uma corrida cada.

NOTA sobre a ordem das decisões: olhar a média do alvo por perfil é, tecnicamente,
olhar um subgrupo de uma condição. Vale como verificação de validade do alvo,
não como busca de achado: o critério ("a média por perfil tem que ficar perto de
zero, porque o alvo é relativo ao próprio piloto") foi declarado antes de rodar,
e o que se faz com um desvio é corrigir o alvo, nunca reportá-lo como resultado.

Diagnóstico puro: não altera o pipeline nem salva base nova.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

BASE = Path("data/processed/base_piloto_corrida.csv")
JANELA_MOVEL = 11
MIN_VIZINHAS = 4
GRID_PADRAO = 20   # escala de referência para a posição normalizada

NOMES_PERFIL = {0: "alto downforce", 1: "stop-and-go",
                2: "potência", 3: "urbano-técnico"}

base = pd.read_csv(BASE)
am = base[base["terminou"] & ~base["corrida_sem_disputa"]].copy()
am = am.sort_values(["piloto", "ano", "round"])
am["perfil"] = am["cluster"].map(NOMES_PERFIL)

# Quantos carros terminaram cada corrida, e a posição DENTRO desse pelotão.
# Uso rank em vez de confiar que as posições dos finalistas sejam 1..N
# contíguas - não depender da suposição custa uma linha.
am["n_terminaram"] = am.groupby(["ano", "round"])["piloto"].transform("size")
am["pos"] = am.groupby(["ano", "round"])["posicao_final"].rank(method="min")

# A posição traduzida para um grid de 20: o vencedor continua 1, o último
# finalista vira 20, independente de quantos sobreviveram.
am["pos_norm"] = 1 + (am["pos"] - 1) * (GRID_PADRAO - 1) / (am["n_terminaram"] - 1)


def alvo_a_partir_de(coluna):
    """O alvo da H2 calculado sobre a coluna de posição que for passada."""
    def ref(g):
        y = g[coluna].astype(float)
        soma = y.rolling(JANELA_MOVEL, center=True, min_periods=MIN_VIZINHAS).sum()
        cont = y.rolling(JANELA_MOVEL, center=True, min_periods=MIN_VIZINHAS).count()
        return (soma - y) / (cont - 1)

    referencia = am.groupby(["piloto", "ano"], group_keys=False).apply(
        ref, include_groups=False
    )
    return referencia - am[coluna]


am["alvo_atual"] = alvo_a_partir_de("pos")
am["alvo_corrigido"] = alvo_a_partir_de("pos_norm")

print(f"Amostra: {len(am)} linhas, "
      f"{am.groupby(['ano','round']).ngroups} corridas")
print(f"Carros que terminaram por corrida: "
      f"{int(am['n_terminaram'].min())} a {int(am['n_terminaram'].max())}, "
      f"mediana {int(am['n_terminaram'].median())}")


# ---------------------------------------------------------------------------
# 1. O teste: o alvo médio da corrida anda com quantos carros terminaram?
# ---------------------------------------------------------------------------
print("\n" + "=" * 66)
print("1. O VIÉS EXISTE?")
print("=" * 66)

por_corrida = am.groupby(["ano", "round"]).agg(
    circuito=("circuito", "first"),
    perfil=("perfil", "first"),
    n_terminaram=("n_terminaram", "first"),
    alvo_atual=("alvo_atual", "mean"),
    alvo_corrigido=("alvo_corrigido", "mean"),
).dropna()

for nome in ["alvo_atual", "alvo_corrigido"]:
    rho, p_rho = spearmanr(por_corrida["n_terminaram"], por_corrida[nome])
    r, p_r = pearsonr(por_corrida["n_terminaram"], por_corrida[nome])
    inclinacao = np.polyfit(por_corrida["n_terminaram"], por_corrida[nome], 1)[0]
    print(f"\n{nome}  ({len(por_corrida)} corridas)")
    print(f"  Spearman rho = {rho:+.3f}  (p = {p_rho:.2g})")
    print(f"  Pearson   r  = {r:+.3f}  (p = {p_r:.2g})")
    print(f"  inclinação: {inclinacao:+.3f} posição no alvo médio por carro"
          f" a mais que termina")

print("\n>> Correlação negativa e forte no alvo_atual confirma o viés:")
print("   menos carros terminando, alvo médio maior.")

# A mesma coisa em faixas, que é mais fácil de ler que um coeficiente
faixas = pd.cut(por_corrida["n_terminaram"], [0, 14, 16, 18, 30],
                labels=["até 14", "15-16", "17-18", "19+"])
print("\nAlvo médio da corrida, por quantos carros terminaram:")
print(por_corrida.groupby(faixas, observed=True).agg(
    corridas=("n_terminaram", "size"),
    atual=("alvo_atual", "mean"),
    corrigido=("alvo_corrigido", "mean"),
).to_string(float_format=lambda x: f"{x:+.3f}"))


# ---------------------------------------------------------------------------
# 2. A correção limpa as médias por perfil?
# ---------------------------------------------------------------------------
print("\n" + "=" * 66)
print("2. AS MÉDIAS POR PERFIL")
print("=" * 66)
print("Critério: têm que ficar perto de zero. O alvo é relativo ao próprio")
print("piloto, e todo piloto corre em todos os perfis.\n")

abandono = (
    base.groupby(base["cluster"].map(NOMES_PERFIL))["terminou"]
    .apply(lambda s: 1 - s.mean())
    .rename("abandono")
)

comp = am.groupby("perfil").agg(
    linhas=("alvo_atual", "size"),
    atual=("alvo_atual", "mean"),
    corrigido=("alvo_corrigido", "mean"),
).join(abandono).sort_values("abandono", ascending=False)

comp["erro_padrao"] = (
    am.groupby("perfil")["alvo_corrigido"].std() / np.sqrt(comp["linhas"])
)
print(comp.to_string(float_format=lambda x: f"{x:+.3f}"))

amp_atual = comp["atual"].max() - comp["atual"].min()
amp_corr = comp["corrigido"].max() - comp["corrigido"].min()
print(f"\nAmplitude entre perfis: {amp_atual:.3f} agora, "
      f"{amp_corr:.3f} depois da correção")
print("(quanto menor, menos o perfil do circuito desloca o alvo por construção)")


# ---------------------------------------------------------------------------
# 3. O que a correção faz com o resto
# ---------------------------------------------------------------------------
print("\n" + "=" * 66)
print("3. EFEITOS COLATERAIS DA CORREÇÃO")
print("=" * 66)

for nome in ["alvo_atual", "alvo_corrigido"]:
    s = am[nome].dropna()
    print(f"{nome}: média {s.mean():+.3f}  desvio {s.std():.2f}  "
          f"assimetria {s.skew():+.3f}  faixa {s.min():+.1f} a {s.max():+.1f}")

r = am[["alvo_atual", "alvo_corrigido"]].corr().iloc[0, 1]
print(f"\nCorrelação entre as duas versões: {r:.3f}")
print("Alta correlação é o esperado: a correção mexe na escala da posição,")
print("não na ideia do alvo. Se fosse baixa, uma das duas estaria errada.")

# O quanto cada corrida é deslocada pela correção, para dimensionar
desloc = (am["alvo_corrigido"] - am["alvo_atual"]).abs()
print(f"\nDeslocamento por linha: mediana {desloc.median():.2f}, "
      f"90º percentil {desloc.quantile(0.9):.2f} posição")

print("\n" + "=" * 66)
print("Nada foi salvo. Se a correção se confirmar, ela entra no")
print("monta_base_analise.py e a tabela de análise é remontada.")