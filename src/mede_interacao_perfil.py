"""
Etapa 2 - A pergunta central do trabalho, reduzida a uma tabela.

A hipótese não prevê que um perfil de circuito renda mais que outro - isso
seria quase contraditório, já que o alvo compara cada piloto com a média dele
próprio, que inclui as corridas de todos os perfis. O que ela prevê é
INTERAÇÃO: que uma mesma escolha de estratégia funcione num perfil e não em
outro.

Interação não aparece em nenhuma medida global. Se "parada antecipada" render
+0,5 em stop-and-go e -0,5 em alto downforce, a média do momento da parada é
zero, a média dos perfis é zero, e mesmo assim existe o padrão que a
monografia procura.

Este script mede isso diretamente: para cada descritor, a média do alvo dentro
de cada perfil. Linhas paralelas = sem interação. Linhas que cruzam = há onde
procurar.

Usa o alvo corrigido pela largada e os descritores na forma relativa à corrida
(que a medição anterior mostrou ser a forma certa). Não altera o pipeline nem
salva base nova - é diagnóstico.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ENTRADA = Path("data/processed/base_analise_sd.csv")

df = pd.read_csv(ENTRADA)


# ---------------------------------------------------------------------------
# Preparo: o alvo corrigido e os descritores na forma relativa
# ---------------------------------------------------------------------------
# Largada fora do alvo: a medição anterior mostrou amplitude de 2,88 posições
# entre a melhor e a pior largada, e a correção zera isso (rho -0,303 -> -0,023)
df["alvo_aj"] = df["alvo"] - df.groupby("grid")["alvo"].transform("mean")
desvio = df["alvo_aj"].std()

# Paradas relativas ao que a corrida pediu (explica 7,21% contra 2,51% da
# versão absoluta)
df["paradas_extra"] = (
    df["n_paradas"] - df.groupby(["ano", "round"])["n_paradas"].transform("median")
)
df["s_paradas"] = pd.cut(
    df["paradas_extra"], [-np.inf, -0.5, 0.5, np.inf],
    labels=["parou menos que o pelotão", "parou igual ao pelotão",
            "parou mais que o pelotão"],
)

# Momento da parada relativo ao pelotão daquela corrida: é isso que distingue
# undercut de overcut. O descritor atual mede terços da distância da prova, o
# que mistura as duas coisas.
df["momento_extra"] = (
    df["momento_primeira_parada"]
    - df.groupby(["ano", "round"])["momento_primeira_parada"].transform("median")
)
q1, q2 = df["momento_extra"].quantile([1 / 3, 2 / 3])
df["s_momento"] = pd.cut(
    df["momento_extra"], [-np.inf, q1, q2, np.inf],
    labels=["undercut (antes do pelotão)", "junto com o pelotão",
            "overcut (depois do pelotão)"],
)

DESCRITORES = {
    "s_paradas": "paradas, relativas ao pelotão",
    "s_momento": "momento da parada, relativo ao pelotão",
    "d_composto": "composto de largada (dureza relativa ao fim de semana)",
    "d_chuva": "chuva",
    "d_safety_car": "safety car",
}

print(f"Base: {len(df)} linhas")
print(f"Alvo corrigido pela largada: desvio {desvio:.2f}")
print(f"Cortes do momento relativo: {q1:+.3f} e {q2:+.3f} da distância da prova")
print(f"(negativo = parou antes do pelotão daquela corrida)\n")


def limiar(n):
    """Quanto uma célula precisa se afastar de zero para não ser ruído."""
    return 2 * desvio / np.sqrt(n) if n else np.inf


# ---------------------------------------------------------------------------
# As tabelas de interação
# ---------------------------------------------------------------------------
perfis = ["alto downforce", "stop-and-go", "potência", "urbano-técnico"]

for col, rotulo in DESCRITORES.items():
    print("=" * 76)
    print(rotulo.upper())
    print("=" * 76)

    medias = df.pivot_table(index=col, columns="d_perfil", values="alvo_aj",
                            aggfunc="mean", observed=True)
    tamanhos = df.pivot_table(index=col, columns="d_perfil", values="alvo_aj",
                              aggfunc="size", observed=True)
    medias = medias.reindex(columns=[p for p in perfis if p in medias.columns])
    tamanhos = tamanhos.reindex(columns=medias.columns)

    print("\nMédia do alvo (posições acima/abaixo do padrão do piloto):")
    print(medias.to_string(float_format=lambda x: f"{x:+.3f}"))

    print("\nLinhas em cada célula:")
    print(tamanhos.to_string())

    # Uma célula só conta se o efeito dela supera o próprio ruído
    destaque = []
    for i in medias.index:
        for c in medias.columns:
            m, n = medias.loc[i, c], tamanhos.loc[i, c]
            if pd.notna(m) and n and abs(m) > limiar(n):
                destaque.append((i, c, m, int(n), limiar(n)))

    print(f"\nCélulas acima do próprio ruído: {len(destaque)} "
          f"de {medias.size}")
    for i, c, m, n, lim in sorted(destaque, key=lambda t: -abs(t[2])):
        print(f"  {str(i):<30} {c:<16} {m:+.2f}  "
              f"({n} linhas, limiar {lim:.2f})")

    # O sinal da interação: a amplitude de cada linha entre os perfis.
    # Linha paralela = amplitude pequena. Linha que cruza = amplitude grande.
    print("\nAmplitude entre perfis, por valor do descritor:")
    for i in medias.index:
        linha = medias.loc[i].dropna()
        if len(linha) > 1:
            amp = linha.max() - linha.min()
            muda_sinal = (linha.min() < 0 < linha.max())
            print(f"  {str(i):<30} {amp:.2f}"
                  f"{'   <- muda de sinal entre perfis' if muda_sinal else ''}")

    # Quanto a combinação explica além do que cada um explica sozinho
    def ve(grupos):
        m = df.groupby(grupos, observed=True)["alvo_aj"].transform("mean")
        return 1 - ((df["alvo_aj"] - m).var() / df["alvo_aj"].var())

    ve_desc, ve_perfil = ve(col), ve("d_perfil")
    ve_junto = ve([col, "d_perfil"])
    print(f"\nVariação do alvo explicada:")
    print(f"  só pelo descritor      {ve_desc:>6.2%}")
    print(f"  só pelo perfil         {ve_perfil:>6.2%}")
    print(f"  pelos dois combinados  {ve_junto:>6.2%}")
    print(f"  ganho da combinação    {ve_junto - ve_desc - ve_perfil:>6.2%}"
          f"   <- isto é a interação")
    print()


# ---------------------------------------------------------------------------
# E se as corridas com problema saírem?
# ---------------------------------------------------------------------------
print("=" * 76)
print("SEM AS LINHAS DE 'PAROU MAIS QUE O PELOTÃO'")
print("=" * 76)
print("Essas 451 linhas deram -2,26 na medição anterior: parada extra é em")
print("boa parte furo e dano, não escolha. Vale ver se a interação muda sem")
print("elas - a decisão de excluir ou não ainda está em aberto.\n")

limpo = df[df["s_paradas"] != "parou mais que o pelotão"].copy()
limpo["alvo_aj"] = limpo["alvo"] - limpo.groupby("grid")["alvo"].transform("mean")
print(f"Linhas: {len(limpo)} (eram {len(df)})")
print(f"Desvio do alvo: {limpo['alvo_aj'].std():.2f} (era {desvio:.2f})\n")

for col, rotulo in [("s_momento", "momento da parada"),
                    ("d_composto", "composto de largada")]:
    medias = limpo.pivot_table(index=col, columns="d_perfil", values="alvo_aj",
                               aggfunc="mean", observed=True)
    medias = medias.reindex(columns=[p for p in perfis if p in medias.columns])
    print(f"{rotulo}:")
    print(medias.to_string(float_format=lambda x: f"{x:+.3f}"))
    print()

print("Nada foi salvo.")