"""
Etapa 2 - Opção 5: busca de subgrupos sobre a decisão de parar.

Pergunta: em que situações as equipes decidem parar?

Entra: data/processed/base_decisao_parada.csv   (monta_base_decisao_parada.py)
Sai:   data/processed/subgrupos_decisao_parada.csv
       data/processed/subgrupos_decisao_parada_finais.csv

Não usa internet. Precisa de sd_permutacao.py na mesma pasta.

Mesmo procedimento da opção 4, com uma diferença: PROFUNDIDADE 2 em vez de 3. São
cerca de 130 mil linhas, quase sete vezes mais que nos duelos, e com profundidade 3
o cálculo do limiar levaria em torno de 40 minutos e vários GB de memória. A
profundidade 2 ainda cobre a pergunta sobre o perfil (perfil E cada condição).

  1. Taxa de parada por perfil, com p por permutação dos perfis entre circuitos.
  2. Busca: pysubgroup, StandardQFNumeric (a=0.5), suporte 200 voltas, os dois
     sentidos. Só concorrem subgrupos com 4+ circuitos e nenhum acima de 40%, também
     nas permutações. Cada repetição embaralha o alvo dentro da corrida, os perfis
     entre circuitos e a era entre corridas do mesmo circuito.
     Firme: passa do limiar, mesmo sentido nas duas eras, mesmo sentido sem o
     circuito mais frequente.
  3. O perfil acrescenta? Compara cada subgrupo firme com perfil ao mesmo sem perfil,
     com limiar corrigido para todos os testes juntos.
  4. Sem redundância: fica o firme que muda a taxa em 1 ponto percentual e 2 erros-
     padrão contra cada versão mais simples. O mínimo é 1 ponto, e não 3 como nos
     duelos, porque a taxa geral de parada é de cerca de 3% por volta.

Descoberta de subgrupos descreve associação, não causa.
"""
from pathlib import Path

import numpy as np
import pandas as pd

import sd_permutacao as S

ENTRADA = Path("data/processed/base_decisao_parada.csv")
SAIDA = Path("data/processed/subgrupos_decisao_parada.csv")
SAIDA_FINAIS = Path("data/processed/subgrupos_decisao_parada_finais.csv")
DESCRITORES = ["d_idade_pneu", "d_composto", "d_neutralizacao", "d_rival_frente",
               "d_rival_tras", "d_janela", "d_dist_frente", "d_paradas_feitas", "d_fase",
               "d_posicao", "d_perfil", "d_era"]
ALVO = "parou"
PROFUNDIDADE, SUPORTE = 2, 200
PERMUTACOES, PERMUTACOES_PERFIL, SEMENTE = 500, 5000, 16
MIN_CIRCUITOS, MAX_CIRCUITO = 4, 0.40
MIN_GANHO, MIN_Z = 0.01, 2.0
FMT = lambda x: f"{x:+.4f}"


def titulo(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def main():
    df = pd.read_csv(ENTRADA).sort_values(["circuito", "ano", "round"], kind="stable") \
        .reset_index(drop=True)
    rng = np.random.default_rng(SEMENTE)
    print(f"Base: {len(df)} voltas, {int(df[ALVO].sum())} paradas, "
          f"{df.groupby(['ano', 'round']).ngroups} corridas, {df['circuito'].nunique()} circuitos")

    titulo("1. TAXA DE PARADA POR PERFIL")
    S.taxa_por_perfil(df, ALVO, rng, PERMUTACOES_PERFIL)

    titulo(f"2. A BUSCA ({len(DESCRITORES)} descritores, profundidade {PROFUNDIDADE})")
    res = S.busca(df, DESCRITORES, ALVO, rng, PROFUNDIDADE, SUPORTE, PERMUTACOES,
                  MIN_CIRCUITOS, MAX_CIRCUITO)
    cols = ["descricao", "linhas", "circuitos", "media", "z", "firme"]
    print("\nMaiores |z|:")
    print(res.head(20)[cols].to_string(index=False, float_format=FMT, max_colwidth=70))

    titulo("3. O PERFIL ACRESCENTA?")
    res = S.perfil_acrescenta(df, res, ALVO, rng, PERMUTACOES)
    res.to_csv(SAIDA, index=False)

    titulo("4. SEM REDUNDÂNCIA")
    finais = S.sem_redundancia(df, res, ALVO, MIN_GANHO, MIN_Z)
    media = df[ALVO].mean()
    print(f"Firmes: {int(res['firme'].sum())}   ficam: {len(finais)}   taxa geral: {media:.4f}\n")
    if len(finais):
        for nome, parte in [("ACIMA DA MÉDIA", finais[finais["media"] > media]),
                            ("ABAIXO DA MÉDIA", finais[finais["media"] < media])]:
            print(f"{nome}: {len(parte)}")
            if len(parte):
                print(parte[cols + ["menor_ganho", "comparado_com"]].head(40).to_string(
                    index=False, float_format=FMT, max_colwidth=60))
            print()
        cp = finais[finais["tem_perfil"]]
        print(f"Finais com perfil: {len(cp)}; o perfil acrescenta em "
              f"{int((cp['perfil_acrescenta'] == True).sum())}")
    finais.to_csv(SAIDA_FINAIS, index=False)
    print(f"\nSalvo em {SAIDA} e {SAIDA_FINAIS}")


if __name__ == "__main__":
    main()
