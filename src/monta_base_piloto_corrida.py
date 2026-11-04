"""
Etapa 2 - Monta a base de piloto-corrida a partir dos arquivos já baixados.

Lê os quatro arquivos por corrida que a coleta salvou, junta tudo numa tabela
só, calcula o que dá pra calcular, e cola o número do grupo da pista.

Não usa internet. Roda em segundos. Pode rodar quantas vezes quiser - se algum
cálculo sair errado, é só corrigir aqui e rodar de novo, sem recoletar nada.

Entra:
  data/interim/corridas/*.csv        o que a coleta baixou
  sessoes_incluidas.csv              ano/round -> circuito, evento, formato, data
  clusters_circuitos.csv             circuito -> número do grupo

Sai:
  data/processed/base_piloto_corrida.csv   uma linha por piloto por corrida
  data/processed/contexto_corridas.csv     uma linha por corrida (para conferir)

ATENÇÃO à coluna de cluster: clusters_circuitos.csv tem duas colunas de grupo e
a que se chama "cluster_final" NÃO é a partição adotada na Etapa 1 (ela separa
Monza/Spielberg dos demais circuitos de potência). A correta é
cluster_kmeans_so_dinamicas_k4, que reproduz a tipologia 11/8/5/2.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ORIGEM = Path("data/interim/corridas")
DESTINO = Path("data/processed")
DESTINO.mkdir(parents=True, exist_ok=True)

COLUNA_CLUSTER = "cluster_kmeans_so_dinamicas_k4"

# Códigos de bandeira do FastF1. Um mesmo registro pode trazer mais de um
# código grudado (ex. "24"), por isso a checagem é por "contém", não igualdade.
COD_SAFETY_CAR = "4"
COD_BANDEIRA_VERMELHA = "5"
COD_VSC = "6"


def acha(nome):
    """Procura o arquivo nas pastas prováveis de data/."""
    for pasta in ["data/raw", "data/processed", "data/external", "data/interim"]:
        p = Path(pasta) / nome
        if p.exists():
            return p
    raise FileNotFoundError(
        f"não achei {nome} em data/raw, data/processed, data/external "
        f"ou data/interim - ajuste o caminho no topo do script"
    )


def le_todos(sufixo):
    """Empilha todos os arquivos de um tipo (pilotos, stints, etc.)."""
    arquivos = sorted(ORIGEM.glob(f"*_{sufixo}.csv"))
    if not arquivos:
        raise FileNotFoundError(f"nenhum arquivo *_{sufixo}.csv em {ORIGEM}")
    partes = [pd.read_csv(a) for a in arquivos]
    partes = [p for p in partes if not p.empty]
    return pd.concat(partes, ignore_index=True), len(arquivos)


# ---------------------------------------------------------------------------
# 1. Carrega o que a coleta baixou
# ---------------------------------------------------------------------------
pilotos, n_arq = le_todos("pilotos")
stints, _ = le_todos("stints")
bandeiras, _ = le_todos("bandeiras")
clima, _ = le_todos("clima")

print(f"Corridas lidas: {n_arq}")
print(f"  linhas de piloto: {len(pilotos)}")
print(f"  linhas de stint:  {len(stints)}")


# ---------------------------------------------------------------------------
# 2. Contexto da corrida: safety car, VSC, chuva, temperatura
#    (uma linha por corrida, não por piloto)
# ---------------------------------------------------------------------------

# Fim da sessão: a última medição de clima serve de referência de duração.
fim_sessao = (
    clima.groupby(["ano", "round"])["tempo_s"].max().rename("duracao_s").reset_index()
)

band = bandeiras.merge(fim_sessao, on=["ano", "round"], how="left")
band["codigo"] = band["codigo"].astype(str)
band = band.sort_values(["ano", "round", "tempo_s"])

# Quanto durou cada período: até o próximo registro de bandeira da mesma corrida.
band["proximo_s"] = band.groupby(["ano", "round"])["tempo_s"].shift(-1)
band["proximo_s"] = band["proximo_s"].fillna(band["duracao_s"])
band["dur_s"] = (band["proximo_s"] - band["tempo_s"]).clip(lower=0)

band["eh_sc"] = band["codigo"].str.contains(COD_SAFETY_CAR)
band["eh_vsc"] = band["codigo"].str.contains(COD_VSC)
band["eh_vermelha"] = band["codigo"].str.contains(COD_BANDEIRA_VERMELHA)

ctx_band = band.groupby(["ano", "round"]).apply(
    lambda g: pd.Series({
        "duracao_s": g["duracao_s"].iloc[0],
        "n_periodos_sc": int(g["eh_sc"].sum()),
        "n_periodos_vsc": int(g["eh_vsc"].sum()),
        "teve_bandeira_vermelha": bool(g["eh_vermelha"].any()),
        "segundos_sob_sc": g.loc[g["eh_sc"], "dur_s"].sum(),
        "segundos_sob_vsc": g.loc[g["eh_vsc"], "dur_s"].sum(),
    }),
    include_groups=False,
).reset_index()

ctx_band["pct_tempo_sob_sc"] = ctx_band["segundos_sob_sc"] / ctx_band["duracao_s"]
ctx_band["pct_tempo_sob_vsc"] = ctx_band["segundos_sob_vsc"] / ctx_band["duracao_s"]


def resume_clima(g):
    """Chuva e temperatura. Guardo QUANDO a chuva começou, porque chuva que cai
    no meio da corrida é evento estratégico - e uma média esconderia isso."""
    chuva = g["chuva"].astype(bool)
    if chuva.any():
        inicio = g.loc[chuva, "tempo_s"].min()
        momento = inicio / g["tempo_s"].max() if g["tempo_s"].max() else np.nan
    else:
        momento = np.nan
    return pd.Series({
        "teve_chuva": bool(chuva.any()),
        "pct_tempo_chuva": float(chuva.mean()),
        "momento_inicio_chuva": momento,  # 0 = largada, 1 = bandeirada
        "temp_pista_media": g["temp_pista"].mean(),
        "temp_ar_media": g["temp_ar"].mean(),
    })


ctx_clima = (
    clima.groupby(["ano", "round"]).apply(resume_clima, include_groups=False).reset_index()
)

contexto = ctx_band.merge(ctx_clima, on=["ano", "round"], how="outer")


# ---------------------------------------------------------------------------
# 3. Estratégia de cada piloto, a partir dos stints
# ---------------------------------------------------------------------------
stints = stints.sort_values(["ano", "round", "piloto", "stint"])
stints["duracao_stint"] = stints["volta_fim"] - stints["volta_inicio"] + 1


def resume_stints(g):
    g = g.sort_values("stint")
    # Número de paradas = trocas de pneu = stints - 1.
    n_paradas = len(g) - 1
    primeira_parada = g["volta_fim"].iloc[0] if n_paradas >= 1 else np.nan
    return pd.Series({
        "n_paradas": n_paradas,
        "compostos": "-".join(g["composto"].astype(str)),
        "composto_largada": g["composto"].iloc[0],
        "n_compostos_distintos": g["composto"].nunique(),
        "volta_primeira_parada": primeira_parada,
        "stint_mais_longo": g["duracao_stint"].max(),
        "stint_mais_curto": g["duracao_stint"].min(),
    })


estrategia = (
    stints.groupby(["ano", "round", "piloto"])
    .apply(resume_stints, include_groups=False)
    .reset_index()
)


# ---------------------------------------------------------------------------
# 4. Junta tudo
# ---------------------------------------------------------------------------
sessoes = pd.read_csv(acha("sessoes_incluidas.csv"))
clusters = pd.read_csv(acha("clusters_circuitos.csv"))

if COLUNA_CLUSTER not in clusters.columns:
    raise KeyError(f"{COLUNA_CLUSTER} não existe em clusters_circuitos.csv")
clusters = clusters[["circuito", COLUNA_CLUSTER]].rename(
    columns={COLUNA_CLUSTER: "cluster"}
)

base = (
    pilotos
    .merge(estrategia, on=["ano", "round", "piloto"], how="left")
    .merge(contexto, on=["ano", "round"], how="left")
    .merge(
        sessoes[["ano", "round", "circuito", "pais", "evento", "formato", "data"]],
        on=["ano", "round"], how="left",
    )
    .merge(clusters, on="circuito", how="left")
)


# ---------------------------------------------------------------------------
# 5. Variáveis derivadas
# ---------------------------------------------------------------------------

# --- dureza relativa do composto de largada --------------------------------
# O nome do composto não é comparável entre anos: hypersoft, ultrasoft e
# supersoft só existiram em 2018, e mesmo hoje o "hard" de uma corrida pode ser
# mais macio que o "medium" de outra, porque a Pirelli escolhe três da escala
# C1-C5 por evento. Usar o nome cru faria a busca encontrar o ano disfarçado
# de estratégia.
# Solução: classificar pela posição DENTRO da escolha daquele fim de semana -
# o mais macio dos três disponíveis é "macio", o mais duro é "duro".
ORDEM_DUREZA = ["HYPERSOFT", "ULTRASOFT", "SUPERSOFT", "SOFT", "MEDIUM", "HARD"]


def mapa_dureza(compostos):
    presentes = [c for c in ORDEM_DUREZA if c in compostos]
    n = len(presentes)
    if n >= 3:
        rotulos = ["macio"] + ["médio"] * (n - 2) + ["duro"]
    elif n == 2:
        rotulos = ["macio", "duro"]
    elif n == 1:
        rotulos = ["médio"]
    else:
        # Corrida inteira na chuva - nenhum composto seco foi usado por
        # ninguém (Suzuka 2022, São Paulo 2024). Não há dureza relativa a
        # calcular, mas "largou de chuva" continua sendo a informação certa.
        return {"INTERMEDIATE": "chuva", "WET": "chuva"}
    m = dict(zip(presentes, rotulos))
    m.update({"INTERMEDIATE": "chuva", "WET": "chuva"})
    return m


mapas = {
    chave: mapa_dureza(set(g["composto"].dropna()))
    for chave, g in stints.groupby(["ano", "round"])
}
base["composto_largada_rel"] = [
    mapas.get((a, r), {}).get(c)
    for a, r, c in zip(base["ano"], base["round"], base["composto_largada"])
]

# --- chuva em faixas, não em sim/não ---------------------------------------
# A marca booleana disparava com uma única medição de chuva em duas horas, o
# que carimbava como molhada qualquer corrida com pancada rápida (Singapura
# quase sempre). A fração de tempo separa o que é corrida de chuva de verdade.
LIMIAR_CHUVA_PONTUAL = 0.15


def faixa_chuva(p):
    if pd.isna(p) or p == 0:
        return "seco"
    if p < LIMIAR_CHUVA_PONTUAL:
        return "chuva pontual"
    return "chuva significativa"


base["faixa_chuva"] = base["pct_tempo_chuva"].map(faixa_chuva)

# --- corridas sem disputa efetiva ------------------------------------------
# Spa 2021: largaram atrás do safety car, deram duas voltas, encerraram. A
# chegada foi a ordem do grid, ninguém parou nos boxes. Não é falha de coleta,
# é uma corrida que não houve - e as linhas dela são ruído puro no alvo.
# A regra é geral (ninguém parou), não uma exceção nomeada, para pegar sozinha
# qualquer caso parecido. As linhas FICAM na base, só marcadas; quem filtra é
# a análise.
pct_sem_parada = (
    base.assign(zero=base["n_paradas"].fillna(0) == 0)
    .groupby(["ano", "round"])["zero"].mean()
)
base["corrida_sem_disputa"] = base.set_index(["ano", "round"]).index.map(
    pct_sem_parada > 0.9
)

# Corrida encurtada é coisa diferente de corrida que não houve: Japão 2022
# rodou metade da distância mas teve corrida de verdade. Fica na base, marcada,
# porque a distância muda o que é uma estratégia razoável.
# ATENÇÃO: total_voltas é a distância PROGRAMADA, não a percorrida - o FastF1
# devolve o número de voltas previsto para o evento. Quando a corrida é
# encerrada antes (Suzuka 2022: 28 voltas de 53), as duas divergem.
# voltas_corrida é a distância que a corrida de fato teve: as voltas do
# vencedor. É esse o denominador certo para tudo que é proporção.
base["voltas_corrida"] = base.groupby(["ano", "round"])[
    "voltas_completadas"
].transform("max")
base["corrida_encurtada"] = base["voltas_corrida"] < 0.75 * base["total_voltas"]

# "terminou" veio do coletor com uma regra incompleta. O FastF1 usa dois
# vocabulários para quem chegou ao fim tomando volta: "Lapped" em alguns anos,
# "+1 Lap" / "+2 Laps" em outros. Recalculo aqui a partir do status cru - que
# está salvo justamente para permitir este tipo de correção sem recoletar.
PADRAO_TERMINOU = r"^(Finished|Lapped|\+\d+ Laps?)$"
base["terminou"] = base["status"].str.match(PADRAO_TERMINOU, na=False)

# Quem nunca largou não tem estratégia nenhuma para descrever. Fica marcado
# para a análise poder excluir essas linhas sem confundi-las com abandono.
base["nao_largou"] = base["status"].isin(["Did not start", "Withdrew"])

# Desclassificado completou a corrida mas foi excluído do resultado - caso
# separado, nem abandono nem chegada normal.
base["desclassificado"] = base["status"] == "Disqualified"

# Posições ganhas: fica em branco para quem largou do box, porque ali a posição
# de largada é 0 e a conta daria um número sem sentido. Quem decide o que fazer
# com essas linhas é a análise, não a montagem da base.
base["posicoes_ganhas"] = np.where(
    base["largou_do_pit"], np.nan, base["grid"] - base["posicao_final"]
)

base["top_5"] = base["posicao_final"] <= 5
base["top_10"] = base["posicao_final"] <= 10
base["pontuou"] = base["pontos"] > 0
# Ambas dividem pela distância REAL da corrida, não pela programada. Com a
# programada, uma parada na volta 20 de 28 em Suzuka 2022 daria 0,38 e seria
# lida como "parou cedo", quando foi nas voltas finais.
base["pct_voltas_completadas"] = base["voltas_completadas"] / base["voltas_corrida"]

# Em que altura da corrida foi a primeira parada (0 = largada, 1 = bandeirada).
base["momento_primeira_parada"] = (
    base["volta_primeira_parada"] / base["voltas_corrida"]
)

colunas = [
    # identificação - guardadas para conferência, NÃO para entrar na busca de padrões
    "ano", "round", "data", "pais", "evento", "formato", "circuito", "piloto", "equipe",
    # ponte com a Etapa 1
    "cluster",
    # resultado
    "grid", "largou_do_pit", "posicao_final", "posicao_classificada",
    "posicoes_ganhas", "pontos", "pontuou", "top_5", "top_10",
    "status", "terminou", "nao_largou", "desclassificado",
    "voltas_completadas", "voltas_corrida", "total_voltas", "pct_voltas_completadas",
    # estratégia
    "n_paradas", "compostos", "composto_largada", "composto_largada_rel",
    "n_compostos_distintos",
    "volta_primeira_parada", "momento_primeira_parada",
    "stint_mais_longo", "stint_mais_curto",
    # contexto da corrida
    "n_periodos_sc", "pct_tempo_sob_sc", "n_periodos_vsc", "pct_tempo_sob_vsc",
    "teve_bandeira_vermelha", "teve_chuva", "pct_tempo_chuva", "faixa_chuva",
    "momento_inicio_chuva", "temp_pista_media", "temp_ar_media",
    # marcas de corrida atípica - a análise decide o que fazer com elas
    "corrida_sem_disputa", "corrida_encurtada",
]
base = base[colunas].sort_values(["ano", "round", "posicao_final"])


# ---------------------------------------------------------------------------
# 6. Conferência
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("CONFERÊNCIA")
print("=" * 60)
print(f"Linhas: {len(base)}")
print(f"Corridas: {base.groupby(['ano', 'round']).ngroups}")
print(f"Circuitos: {base['circuito'].nunique()}")
print(f"Pilotos distintos: {base['piloto'].nunique()}")
print(f"Anos: {sorted(base['ano'].unique())}")

sem_cluster = base["cluster"].isna().sum()
print(f"\nLinhas sem número de grupo: {sem_cluster}")
if sem_cluster:
    print("  circuitos sem grupo:",
          base.loc[base["cluster"].isna(), "circuito"].unique().tolist())
    print("  >> a ponte com a Etapa 1 está quebrada nessas linhas, resolver antes de seguir")

print("\nLinhas por grupo:")
print(base["cluster"].value_counts().sort_index().to_string())

print("\nVazios por coluna (só as que têm):")
vazios = base.isna().sum()
vazios = vazios[vazios > 0]
print(vazios.to_string() if len(vazios) else "  nenhum")

print("\nValores extremos (sanidade):")
print(f"  grid: {base['grid'].min():.0f} a {base['grid'].max():.0f}")
print(f"  largadas do box: {int(base['largou_do_pit'].sum())}")
print(f"  paradas: {base['n_paradas'].min():.0f} a {base['n_paradas'].max():.0f}")
print(f"  terminaram: {int(base['terminou'].sum())} "
      f"({base['terminou'].mean():.1%})  <- esperado ~85%")
print(f"  não largaram: {int(base['nao_largou'].sum())}")
print(f"  desclassificados: {int(base['desclassificado'].sum())}")
print(f"  corridas com chuva: "
      f"{base.groupby(['ano','round'])['teve_chuva'].first().sum()}")
print(f"  corridas com safety car: "
      f"{(base.groupby(['ano','round'])['n_periodos_sc'].first() > 0).sum()}")

print("\nComposto de largada, por dureza relativa ao fim de semana:")
print(base["composto_largada_rel"].value_counts(dropna=False).to_string())

print("\nChuva em faixas (corridas, não linhas):")
print(base.groupby(["ano", "round"])["faixa_chuva"].first()
      .value_counts().to_string())

sem_disputa = base.loc[base["corrida_sem_disputa"], ["ano", "round", "circuito"]]
print(f"\nCorridas sem disputa efetiva: "
      f"{sem_disputa.drop_duplicates().shape[0]} "
      f"({len(sem_disputa)} linhas marcadas)")
if len(sem_disputa):
    for _, r in sem_disputa.drop_duplicates().iterrows():
        print(f"  {r['ano']} r{r['round']} {r['circuito']}")

encurtadas = base.loc[base["corrida_encurtada"], ["ano", "round", "circuito"]]
print(f"\nCorridas encurtadas: {encurtadas.drop_duplicates().shape[0]}")
for _, r in encurtadas.drop_duplicates().iterrows():
    print(f"  {r['ano']} r{r['round']} {r['circuito']}")

base.to_csv(DESTINO / "base_piloto_corrida.csv", index=False)
contexto.merge(
    sessoes[["ano", "round", "circuito"]], on=["ano", "round"], how="left"
).to_csv(DESTINO / "contexto_corridas.csv", index=False)

print(f"\nSalvo em {DESTINO}/base_piloto_corrida.csv")