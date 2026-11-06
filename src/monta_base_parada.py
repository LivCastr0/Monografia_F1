"""
Etapa 2 - Monta a base do caminho A1: uma linha por parada com duelo válido.

Entra: data/interim/voltas/*_voltas.csv
       data/interim/corridas/*_stints.csv
       data/processed/base_piloto_corrida.csv
Sai:   data/processed/base_parada.csv

Não usa internet. Precisa de avalia_piloto_a1.py na mesma pasta, porque usa as
mesmas regras de parada útil e de saldo de duelos.

Alvo: `saldo`. Positivo = quem parou primeiro ganhou posição nos duelos da parada
(o undercut deu certo). Negativo = perdeu posição (o rival que parou depois levou
a melhor). `saldo_alt` é a mesma medida com vizinhança de 2 posições e janela de 5
voltas, só para verificar robustez.

Corridas com dado de parada incompleto ficam de fora: se menos de 80% dos pilotos
têm o número de paradas igual ao dos stints, faltam registros de box e o saldo
daquela corrida não é confiável.

Descritores (prefixo d_), fixados antes de ver qualquer resultado:
  d_perfil       perfil do circuito (Etapa 1)
  d_fase         em que altura da corrida foi a parada (terços observados)
  d_troca_pneu   pneu que entrou comparado ao que saiu, em dureza
  d_posicao      posição antes da parada
  d_distancia    distância para o carro da frente antes da parada
  d_chuva        chuva da corrida
  d_safety_car   se a corrida teve safety car (a parada em si foi sob bandeira verde)
"""
from pathlib import Path

import numpy as np
import pandas as pd

import avalia_piloto_a1 as A

BASE_CORRIDAS = Path("data/processed/base_piloto_corrida.csv")
SAIDA = Path("data/processed/base_parada.csv")

MIN_BATE_PARADAS = 0.80
NOMES_PERFIL = {0: "alto downforce", 1: "stop-and-go", 2: "potência", 3: "urbano-técnico"}
DUREZA = ["HYPERSOFT", "ULTRASOFT", "SUPERSOFT", "SOFT", "MEDIUM", "HARD"]
CHUVA = {"INTERMEDIATE", "WET"}


def saldo_com(df, uteis, vizinhanca, janela):
    guardados = A.VIZINHANCA, A.JANELA
    A.VIZINHANCA, A.JANELA = vizinhanca, janela
    s = A.saldos(df, uteis)
    A.VIZINHANCA, A.JANELA = guardados
    return s


def troca_pneu(antes, depois):
    if antes in CHUVA or depois in CHUVA:
        return "pneu de chuva"
    if antes not in DUREZA or depois not in DUREZA:
        return np.nan
    a, b = DUREZA.index(antes), DUREZA.index(depois)
    return "pneu mais macio" if b < a else "pneu mais duro" if b > a else "mesmo pneu"


def faixa_posicao(p):
    return "até P5" if p <= 5 else "P6 a P10" if p <= 10 else "P11 em diante"


def faixa_distancia(d):
    if pd.isna(d):
        return np.nan
    return "até 1,5 s" if d <= 1.5 else "1,5 a 3 s" if d <= 3 else "mais de 3 s"


def corrida(arquivo):
    """Devolve as paradas da corrida e a fração de pilotos com paradas conferidas."""
    ano, rnd = (int(t) for t in arquivo.name.split("_")[:2])
    df = A.le_voltas(arquivo)
    stints = pd.read_csv(A.CORRIDAS / f"{ano}_{rnd:02d}_stints.csv")
    par = A.acha_paradas(df)

    esperadas = stints.groupby("piloto").size() - 1
    medidas = par[par["trocou_pneu"]].groupby("piloto").size()
    medidas = medidas.reindex(esperadas.index, fill_value=0)
    conferidos = (medidas == esperadas).mean()

    uteis = A.paradas_uteis(par, int(df["volta"].max()))
    s = A.saldos(df, uteis)
    if s.empty:
        return s, conferidos, (ano, rnd)
    alt = saldo_com(df, uteis, 2, 5).rename(columns={"saldo": "saldo_alt"})
    s = s.merge(alt[["piloto", "volta_parada", "saldo_alt"]],
                on=["piloto", "volta_parada"], how="left")

    voltas = df.set_index(["piloto", "volta"]).sort_index()
    linhas = []
    for x in s.itertuples():
        base = voltas.loc[(x.piloto, x.volta_parada - 1)]
        entrada = voltas.loc[(x.piloto, x.volta_parada)]
        saida = voltas.loc[(x.piloto, x.volta_parada + 1)]
        na_volta = df[df["volta"] == x.volta_parada - 1]
        frente = na_volta[na_volta["posicao"] == base["posicao"] - 1]
        if base["posicao"] == 1:
            distancia = "líder"
        elif frente.empty:
            distancia = np.nan
        else:
            distancia = faixa_distancia(base["fim_volta_s"] - frente["fim_volta_s"].iloc[0])
        linhas.append({
            "equipe": base["equipe"],
            "posicao_antes": base["posicao"],
            "composto_antes": entrada["composto"],
            "composto_depois": saida["composto"],
            "d_troca_pneu": troca_pneu(entrada["composto"], saida["composto"]),
            "d_posicao": faixa_posicao(base["posicao"]),
            "d_distancia": distancia,
        })
    s = pd.concat([s, pd.DataFrame(linhas)], axis=1)
    s.insert(0, "round", rnd)
    s.insert(0, "ano", ano)
    return s, conferidos, (ano, rnd)


def main():
    partes, fora = [], []
    arquivos = sorted(A.VOLTAS.glob("*_voltas.csv"))
    for a in arquivos:
        s, conferidos, chave = corrida(a)
        if conferidos < MIN_BATE_PARADAS:
            fora.append((chave, conferidos))
        elif len(s):
            partes.append(s)

    print(f"Corridas lidas: {len(arquivos)}")
    print(f"Fora por dado de parada incompleto: {len(fora)}")
    for (ano, rnd), c in fora:
        print(f"  {ano} r{rnd}: {c:.0%} dos pilotos conferidos")

    base = pd.concat(partes, ignore_index=True)

    ctx = pd.read_csv(BASE_CORRIDAS).drop_duplicates(["ano", "round"])[
        ["ano", "round", "data", "circuito", "cluster", "voltas_corrida",
         "faixa_chuva", "n_periodos_sc"]]
    base = base.merge(ctx, on=["ano", "round"], how="left")
    base["d_perfil"] = base["cluster"].map(NOMES_PERFIL)
    base["d_chuva"] = base["faixa_chuva"]
    base["d_safety_car"] = np.where(base["n_periodos_sc"] > 0,
                                    "com safety car", "sem safety car")

    base["fracao_corrida"] = base["volta_parada"] / base["voltas_corrida"]
    q1, q2 = base["fracao_corrida"].quantile([1 / 3, 2 / 3])
    base["d_fase"] = pd.cut(base["fracao_corrida"], [-np.inf, q1, q2, np.inf],
                            labels=["parada cedo", "parada no meio", "parada tarde"]
                            ).astype(str)

    descritores = [c for c in base.columns if c.startswith("d_")]
    antes = len(base)
    base = base.dropna(subset=descritores + ["saldo"])
    print(f"\nParadas com duelo: {antes}; sem algum descritor: {antes - len(base)}")
    print(f"Base final: {len(base)} paradas, "
          f"{base.groupby(['ano', 'round']).ngroups} corridas, "
          f"{base['circuito'].nunique()} circuitos")
    print(f"Cortes da fase: {q1:.3f} e {q2:.3f} da corrida")

    print("\nOs descritores:")
    for d in descritores:
        print(f"\n  {d}")
        print(base[d].value_counts().to_string())

    base.to_csv(SAIDA, index=False)
    print(f"\nSalvo em {SAIDA}")
    print("Nenhum valor do alvo dentro de nenhum grupo foi examinado aqui.")


if __name__ == "__main__":
    main()
