"""
Diagnóstico da H2 (afinidade piloto x tipo de pista) antes de escrever a SD.

Pergunta central: a base tem corridas suficientes para sustentar padrões do
tipo "o piloto X rende acima do próprio normal no grupo Y"? E a referência
do que é "o próprio normal" aguenta o desenvolvimento do carro ao longo da
temporada?

Nada aqui decide nada. Só mede, pra decisão ser tomada com número na mão.
Lê a base já montada, não usa internet, roda em segundos.

Definição do alvo usada aqui:
  alvo = referência - posição final        (positivo = foi melhor que o normal)

Duas referências, calculadas lado a lado para comparação:
  temporada  media das OUTRAS corridas do piloto naquele ano
  movel      media das outras corridas vizinhas (janela de 11, mesmo ano)

Em ambas a própria corrida fica de fora da conta. Se entrasse, o alvo
encolheria em direção a zero, e mais para quem correu pouco.

Só pilotos que terminaram entram: abandono não tem posição, e tratá-lo como
último lugar transformaria quebra de carro em "antipatia pela pista".
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BASE = Path("data/processed/base_piloto_corrida.csv")
SAIDA = Path("data/processed")

# Quantas linhas um par piloto x grupo precisa ter para o padrão valer a pena.
LIMIARES = [5, 10, 15, 20, 30]
# Mínimo de corridas numa temporada para a referência do piloto fazer sentido.
MIN_CORRIDAS_TEMPORADA = 8
JANELA_MOVEL = 11

base = pd.read_csv(BASE)
print(f"Base: {len(base)} linhas, {base.groupby(['ano','round']).ngroups} corridas\n")


# ---------------------------------------------------------------------------
# A. Tamanho real de cada grupo, e se o abandono é desigual entre eles
# ---------------------------------------------------------------------------
print("=" * 68)
print("A. OS GRUPOS")
print("=" * 68)

grupos = base.groupby("cluster").agg(
    corridas=("round", lambda s: base.loc[s.index].groupby(["ano", "round"]).ngroups),
    circuitos=("circuito", "nunique"),
    linhas=("piloto", "size"),
    terminaram=("terminou", "sum"),
)
grupos["pct_abandono"] = 1 - grupos["terminaram"] / grupos["linhas"]
print(grupos.to_string())

print("\nCircuitos de cada grupo:")
for c, g in base.groupby("cluster"):
    print(f"  {int(c)}: {', '.join(sorted(g['circuito'].unique()))}")

print("\n>> Se a taxa de abandono variar muito entre grupos, excluir quem não")
print("   terminou encolhe os grupos de forma desigual - precisa ir escrito.")


# ---------------------------------------------------------------------------
# B. Os pilotos: quem tem temporada cheia o bastante para ter uma referência
# ---------------------------------------------------------------------------
print("\n" + "=" * 68)
print("B. OS PILOTOS")
print("=" * 68)

corridas_piloto_ano = (
    base.groupby(["piloto", "ano"]).size().rename("corridas").reset_index()
)
validos = corridas_piloto_ano[corridas_piloto_ano["corridas"] >= MIN_CORRIDAS_TEMPORADA]

print(f"Pilotos distintos: {base['piloto'].nunique()}")
print(f"Pares piloto-temporada: {len(corridas_piloto_ano)}")
print(f"  com {MIN_CORRIDAS_TEMPORADA}+ corridas (referência utilizável): {len(validos)}")
print(f"  abaixo disso (substitutos, temporadas parciais): "
      f"{len(corridas_piloto_ano) - len(validos)}")

total_piloto = base.groupby("piloto").size().sort_values(ascending=False)
print(f"\nCorridas por piloto - maior: {total_piloto.max()}, "
      f"mediana: {int(total_piloto.median())}, menor: {total_piloto.min()}")
print(f"Pilotos com menos de 20 corridas no total: {(total_piloto < 20).sum()}")


# ---------------------------------------------------------------------------
# C. O teste que importa: piloto x grupo tem linhas suficientes?
# ---------------------------------------------------------------------------
print("\n" + "=" * 68)
print("C. PILOTO x GRUPO - A PERGUNTA CENTRAL")
print("=" * 68)

# Só quem terminou, que é a amostra que a H2 vai usar de fato.
fin = base[base["terminou"]].copy()

matriz = fin.pivot_table(
    index="piloto", columns="cluster", values="ano", aggfunc="size", fill_value=0
)

print(f"Amostra (só quem terminou): {len(fin)} linhas\n")
print("Pares piloto x grupo que alcançam cada limiar:")
print(f"{'limiar':>8} {'pares':>7} {'g0':>5} {'g1':>5} {'g2':>5} {'g3':>5}")
for lim in LIMIARES:
    passa = matriz >= lim
    por_grupo = [int(passa[c].sum()) if c in passa.columns else 0 for c in [0, 1, 2, 3]]
    print(f"{lim:>8} {int(passa.values.sum()):>7} "
          + " ".join(f"{v:>5}" for v in por_grupo))

print("\nMaior amostra possível por grupo (o piloto mais presente):")
for c in sorted(matriz.columns):
    print(f"  grupo {int(c)}: {matriz[c].max()} linhas "
          f"({matriz[c].idxmax()})")

matriz.to_csv(SAIDA / "diag_piloto_x_grupo.csv")
print(f"\nMatriz completa salva em {SAIDA}/diag_piloto_x_grupo.csv")

print("\n>> Leitura: cada par que passa do limiar é um padrão que a SD PODE")
print("   encontrar. Onde a contagem for baixa, ausência de achado não é")
print("   ausência de afinidade - é falta de corrida.")


# ---------------------------------------------------------------------------
# D. As duas referências, e o perigo do carro que evolui no meio do ano
# ---------------------------------------------------------------------------
print("\n" + "=" * 68)
print("D. A REFERÊNCIA DO PILOTO")
print("=" * 68)

fin = fin.sort_values(["piloto", "ano", "round"])


def referencias(g):
    """Calcula as duas referências para um piloto numa temporada."""
    y = g["posicao_final"].astype(float)
    n = len(y)

    # temporada: média das OUTRAS corridas do ano
    g["ref_temporada"] = (y.sum() - y) / (n - 1) if n > 1 else np.nan

    # móvel: média das outras corridas da vizinhança
    soma = y.rolling(JANELA_MOVEL, center=True, min_periods=4).sum()
    cont = y.rolling(JANELA_MOVEL, center=True, min_periods=4).count()
    g["ref_movel"] = (soma - y) / (cont - 1)
    return g


fin = (
    fin.groupby(["piloto", "ano"], group_keys=False)
    .apply(referencias, include_groups=False)
    .join(fin[["piloto", "ano"]])
)

fin["alvo_temporada"] = fin["ref_temporada"] - fin["posicao_final"]
fin["alvo_movel"] = fin["ref_movel"] - fin["posicao_final"]

for nome in ["alvo_temporada", "alvo_movel"]:
    s = fin[nome].dropna()
    print(f"{nome}: {len(s)} valores, média {s.mean():+.3f}, "
          f"desvio {s.std():.2f}, faixa {s.min():+.1f} a {s.max():+.1f}")

# O teste do perigo: o alvo anda junto com o avanço da temporada?
# Se andar, é o carro evoluindo se disfarçando de afinidade com pista.
print("\nTeste do carro que evolui - correlação entre o alvo e o round,")
print("dentro de cada equipe-temporada (Spearman):")

for nome in ["alvo_temporada", "alvo_movel"]:
    rhos = []
    for _, g in fin.dropna(subset=[nome]).groupby(["equipe", "ano"]):
        if len(g) >= 10:
            rho, _ = spearmanr(g["round"], g[nome])
            if not np.isnan(rho):
                rhos.append(rho)
    rhos = np.array(rhos)
    fortes = int((np.abs(rhos) > 0.4).sum())
    print(f"  {nome}: {len(rhos)} equipe-temporadas, "
          f"|rho| médio {np.abs(rhos).mean():.3f}, "
          f"{fortes} com |rho| > 0.4 ({fortes/len(rhos):.0%})")

print("\n>> Quanto menor essa correlação, menos o alvo está capturando")
print("   evolução de carro. A referência com o número menor é a mais segura.")

fin.to_csv(SAIDA / "diag_h2_alvos.csv", index=False)
print(f"\nSalvo em {SAIDA}/diag_h2_alvos.csv")