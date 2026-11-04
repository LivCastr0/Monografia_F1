"""
Etapa 2 - Quanto do ruído do alvo é da corrida, e não do piloto.

O alvo tem desvio de 3,4 posições, e a medição de interação não achou nada
acima desse ruído. Antes de aceitar o resultado negativo, vale separar duas
fontes de variação que estão somadas ali:

  entre corridas   uma corrida com três safety cars embaralha os vinte
                   pilotos; uma corrida limpa não embaralha ninguém. Esse
                   deslocamento é comum a todo o grid e não tem relação com
                   a escolha de cada um.

  dentro da corrida   o que distingue os pilotos que correram a MESMA prova.
                      É aqui que mora a escolha estratégica.

Se a parcela entre corridas for grande, dá para removê-la subtraindo de cada
linha a média da própria corrida. O alvo passa a perguntar: dentro desta
corrida, ele se saiu melhor que os outros?

Uma consequência importante dessa centragem: descritores que são constantes
dentro de uma corrida - perfil do circuito, chuva, safety car - perdem o
efeito principal por construção, já que todas as linhas daquela corrida
recebem o mesmo desconto. Mas a INTERAÇÃO deles com estratégia sobrevive
intacta, porque ela compara escolhas diferentes dentro da mesma prova. E a
interação é exatamente o que a hipótese prevê.

Diagnóstico. Não altera o pipeline nem salva base nova.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ENTRADA = Path("data/processed/base_analise_sd.csv")

df = pd.read_csv(ENTRADA)
CORRIDA = ["ano", "round"]

# Largada fora, como já estabelecido
df["alvo_aj"] = df["alvo"] - df.groupby("grid")["alvo"].transform("mean")


# ---------------------------------------------------------------------------
# A. A decomposição
# ---------------------------------------------------------------------------
print("=" * 72)
print("A. DE ONDE VEM A VARIAÇÃO DO ALVO")
print("=" * 72)

for nome, col in [("alvo original", "alvo"), ("alvo sem a largada", "alvo_aj")]:
    media_corrida = df.groupby(CORRIDA)[col].transform("mean")
    var_total = df[col].var()
    var_entre = media_corrida.var()
    var_dentro = (df[col] - media_corrida).var()

    print(f"\n{nome}:")
    print(f"  desvio total           {np.sqrt(var_total):.2f} posições")
    print(f"  entre corridas         {var_entre / var_total:>6.1%} "
          f"(desvio {np.sqrt(var_entre):.2f})")
    print(f"  dentro das corridas    {var_dentro / var_total:>6.1%} "
          f"(desvio {np.sqrt(var_dentro):.2f})")

print("\n>> A parcela 'entre corridas' é ruído comum ao grid inteiro.")
print("   Se for grande, removê-la aumenta a sensibilidade sem mudar o desenho.")


# ---------------------------------------------------------------------------
# B. O alvo centrado na corrida
# ---------------------------------------------------------------------------
df["alvo_dentro"] = df["alvo_aj"] - df.groupby(CORRIDA)["alvo_aj"].transform("mean")

desvio_antes = df["alvo_aj"].std()
desvio_depois = df["alvo_dentro"].std()

print("\n" + "=" * 72)
print("B. O ALVO CENTRADO NA CORRIDA")
print("=" * 72)
print(f"Desvio: {desvio_antes:.2f} -> {desvio_depois:.2f} "
      f"({1 - desvio_depois / desvio_antes:.1%} menor)")
print(f"Correlação com a versão anterior: "
      f"{df['alvo_aj'].corr(df['alvo_dentro']):.3f}")

print("\nEfeito detectável, antes e depois:")
print(f"{'linhas':>8} {'antes':>8} {'depois':>8}")
for n in [50, 100, 200, 300, 500, 800]:
    print(f"{n:>8} {2 * desvio_antes / np.sqrt(n):>8.2f} "
          f"{2 * desvio_depois / np.sqrt(n):>8.2f}")


# ---------------------------------------------------------------------------
# C. A interação, refeita com o alvo mais limpo
# ---------------------------------------------------------------------------
df["paradas_extra"] = (
    df["n_paradas"] - df.groupby(CORRIDA)["n_paradas"].transform("median")
)
df["s_paradas"] = pd.cut(
    df["paradas_extra"], [-np.inf, -0.5, 0.5, np.inf],
    labels=["parou menos", "parou igual", "parou mais"],
)
df["momento_extra"] = (
    df["momento_primeira_parada"]
    - df.groupby(CORRIDA)["momento_primeira_parada"].transform("median")
)
q1, q2 = df["momento_extra"].quantile([1 / 3, 2 / 3])
df["s_momento"] = pd.cut(
    df["momento_extra"], [-np.inf, q1, q2, np.inf],
    labels=["undercut", "junto", "overcut"],
)

print("\n" + "=" * 72)
print("C. A INTERAÇÃO COM O ALVO CENTRADO")
print("=" * 72)
print("Lembrete: o efeito principal do perfil agora é zero por construção")
print("(o perfil é constante dentro de uma corrida). O que importa aqui é o")
print("ganho da combinação - a interação, que a centragem não afeta.\n")

perfis = ["alto downforce", "stop-and-go", "potência", "urbano-técnico"]


def ve(col, alvo):
    m = df.groupby(col, observed=True)[alvo].transform("mean")
    return 1 - ((df[alvo] - m).var() / df[alvo].var())


for col, rotulo in [("s_paradas", "paradas relativas ao pelotão"),
                    ("s_momento", "undercut / overcut"),
                    ("d_composto", "composto de largada")]:
    medias = df.pivot_table(index=col, columns="d_perfil",
                            values="alvo_dentro", aggfunc="mean", observed=True)
    tam = df.pivot_table(index=col, columns="d_perfil",
                         values="alvo_dentro", aggfunc="size", observed=True)
    cols = [p for p in perfis if p in medias.columns]
    medias, tam = medias.reindex(columns=cols), tam.reindex(columns=cols)

    print(f"--- {rotulo} ---")
    print(medias.to_string(float_format=lambda x: f"{x:+.3f}"))

    acima = []
    for i in medias.index:
        for c in cols:
            m, n = medias.loc[i, c], tam.loc[i, c]
            if pd.notna(m) and n and abs(m) > 2 * desvio_depois / np.sqrt(n):
                acima.append((i, c, m, int(n)))
    print(f"\ncélulas acima do próprio ruído: {len(acima)} de {medias.size}")
    for i, c, m, n in sorted(acima, key=lambda t: -abs(t[2])):
        print(f"   {str(i):<14} {c:<16} {m:+.2f}  ({n} linhas, "
              f"limiar {2 * desvio_depois / np.sqrt(n):.2f})")

    ganho = ve([col, "d_perfil"], "alvo_dentro") - ve(col, "alvo_dentro")
    ganho_antes = ve([col, "d_perfil"], "alvo_aj") - ve(col, "alvo_aj")
    print(f"\ninteração: {ganho:.2%}  (era {ganho_antes:.2%} "
          f"com o alvo anterior)\n")


# ---------------------------------------------------------------------------
# D. O teto: qual o maior efeito que poderia estar escondido
# ---------------------------------------------------------------------------
print("=" * 72)
print("D. O QUE DÁ PARA AFIRMAR SE NADA APARECER")
print("=" * 72)
print("Um resultado negativo só vale com o tamanho do efeito que você")
print("CONSEGUIRIA ter detectado. Por tamanho de célula, com o alvo centrado:\n")

tamanhos_tipicos = (
    df.groupby(["s_momento", "d_perfil"], observed=True).size()
)
print(f"células de interação estratégia x perfil: "
      f"{tamanhos_tipicos.min()} a {tamanhos_tipicos.max()} linhas, "
      f"mediana {int(tamanhos_tipicos.median())}")
n_med = int(tamanhos_tipicos.median())
print(f"\n=> com {n_med} linhas por célula, o estudo detecta efeitos a partir")
print(f"   de {2 * desvio_depois / np.sqrt(n_med):.2f} posição.")
print(f"   Nada abaixo disso pode ser afirmado nem negado com esta base.")

print("\nNada foi salvo.")