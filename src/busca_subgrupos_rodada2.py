"""
Etapa 2 - Segunda rodada da busca de subgrupos, com as correções aplicadas.

Pergunta: quais escolhas de estratégia, em que perfil de circuito e sob quais
condições, acompanham desempenho acima ou abaixo do padrão do próprio piloto?

Entra: data/processed/base_analise_sd.csv
Sai:   data/processed/subgrupos_rodada2.csv
       data/processed/subgrupos_rodada2_sem_parou_mais.csv

Configuração da busca, igual à da primeira rodada:
  medida            StandardQFNumeric (a=0.5)
  profundidade      até 3 condições
  suporte mínimo    50 linhas

O que muda:
  alvo              `alvo_ajustado` (sem o efeito da posição de largada)
  descritores       paradas e momento relativos ao pelotão da própria corrida
  direção           os dois sentidos: acima e abaixo do padrão do piloto
  limiar            por permutação, descontando o número de subgrupos testados

Por que permutação: com centenas de subgrupos, alguns passam de dois erros-
padrão por acaso. Aqui o alvo é embaralhado entre os pilotos de cada corrida, a
busca é repetida, e guarda-se o maior valor encontrado. O limiar é o valor que
esse máximo só ultrapassa em 5% das repetições. Embaralhar dentro da corrida não
testa descritores que valem para a corrida inteira (perfil, chuva, safety car)
sozinhos, mas testa as combinações deles com a estratégia, que é a pergunta.

A busca roda duas vezes: com a base completa e sem as linhas de "parou mais que
o pelotão". Excluir essas linhas não é decisão tomada; parada extra costuma vir
de problema na corrida, e as duas versões mostram quanto o resultado depende dela.

Descoberta de subgrupos descreve associação, não causa.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pysubgroup as ps

ENTRADA = Path("data/processed/base_analise_sd.csv")
SAIDA = Path("data/processed/subgrupos_rodada2.csv")
SAIDA_SEM = Path("data/processed/subgrupos_rodada2_sem_parou_mais.csv")

ALVO = "alvo_ajustado"
SUPORTE_MINIMO = 50
PROFUNDIDADE = 3
PESO_TAMANHO = 0.5
PERMUTACOES = 1000
SEMENTE = 3
QUANTOS_MOSTRAR = 15
TODOS = 100_000        # maior que o espaço de busca: guarda todos os subgrupos

PAROU_MAIS = "parou mais que o pelotão"


def rodada(df, descritores, saida):
    """Busca, limiar por permutação e verificações. Devolve a tabela."""
    df = df.reset_index(drop=True)
    busca_df = df[descritores + [ALVO]].copy()

    tarefa = ps.SubgroupDiscoveryTask(
        busca_df,
        ps.NumericTarget(ALVO),
        ps.create_selectors(busca_df, ignore=[ALVO]),
        result_set_size=TODOS,
        depth=PROFUNDIDADE,
        qf=ps.StandardQFNumeric(a=PESO_TAMANHO),
        constraints=[ps.MinSupportConstraint(SUPORTE_MINIMO)],
    )
    achados = ps.Apriori().execute(tarefa).to_descriptions()

    cobre = np.array([np.asarray(sg.covers(busca_df)) for _, sg in achados])
    n = cobre.sum(axis=1)
    y = df[ALVO].to_numpy()

    def medida(valores):
        """Distância da média geral em erros-padrão, para cada subgrupo.
        É a mesma medida da busca (a=0,5), dividida pelo desvio do alvo."""
        return cobre @ (valores - valores.mean()) / np.sqrt(n) / valores.std(ddof=1)

    z = medida(y)

    rng = np.random.default_rng(SEMENTE)
    corridas = list(df.groupby(["ano", "round"]).indices.values())
    maximos = np.empty(PERMUTACOES)
    for i in range(PERMUTACOES):
        embaralhado = y.copy()
        for linhas in corridas:
            embaralhado[linhas] = y[rng.permutation(linhas)]
        maximos[i] = np.abs(medida(embaralhado)).max()
    limiar = np.quantile(maximos, 0.95)

    linhas = []
    for i, (qualidade, sg) in enumerate(achados):
        sub = df[cobre[i]]
        linhas.append({
            "descricao": str(sg),
            "condicoes": str(sg).count("AND") + 1,
            "linhas": int(n[i]),
            "media": sub[ALVO].mean(),
            "mediana": sub[ALVO].median(),
            "z": z[i],
            "acima_do_limiar": abs(z[i]) > limiar,
            "media_ref_temporada": sub["alvo_temporada_ajustado"].mean(),
            "media_pos_bruta": sub["alvo_pos_bruta_ajustado"].mean(),
            "circuitos": sub["circuito"].nunique(),
            "maior_circuito_pct": sub["circuito"].value_counts(normalize=True).iloc[0],
            "corridas": sub.groupby(["ano", "round"]).ngroups,
            "qualidade": qualidade,
        })
    res = pd.DataFrame(linhas)
    res = res.reindex(res["z"].abs().sort_values(ascending=False).index)

    print(f"Linhas: {len(df)}   Corridas: {len(corridas)}")
    print(f"Desvio do alvo: {df[ALVO].std():.2f}")
    print(f"Subgrupos com {SUPORTE_MINIMO}+ linhas: {len(res)}")
    print(f"Limiar (|z|, 5%, {PERMUTACOES} permutações): {limiar:.2f}")
    print(f"Maior |z| encontrado: {res['z'].abs().max():.2f}")
    print(f"Acima do limiar: {int(res['acima_do_limiar'].sum())}\n")

    print(res.head(QUANTOS_MOSTRAR)[
        ["descricao", "linhas", "media", "z", "acima_do_limiar"]
    ].to_string(index=False, float_format=lambda x: f"{x:+.2f}", max_colwidth=78))

    passou = res[res["acima_do_limiar"]]
    if len(passou):
        print(f"\nDos {len(passou)} acima do limiar:")
        mudam = passou[(np.sign(passou["media"]) != np.sign(passou["media_ref_temporada"]))
                       | (np.sign(passou["media"]) != np.sign(passou["media_pos_bruta"]))]
        print(f"  mudam de sinal em alguma versão do alvo: {len(mudam)}")
        print(f"  com um circuito acima de 50% das linhas: "
              f"{int((passou['maior_circuito_pct'] > 0.5).sum())}")
        for rotulo, texto in [("parou mais", PAROU_MAIS),
                              ("parou igual", "parou igual ao pelotão"),
                              ("parou menos", "parou menos que o pelotão")]:
            print(f"  contêm '{rotulo}': "
                  f"{int(passou['descricao'].str.contains(texto, regex=False).sum())}")
        outros = passou[~passou["descricao"].str.contains("d_paradas", regex=False)]
        print(f"  sem nenhuma condição de paradas: {len(outros)}")

    res.to_csv(saida, index=False)
    print(f"\nSalvo em {saida}")
    return res


base = pd.read_csv(ENTRADA)
descritores = [c for c in base.columns if c.startswith("d_")]
print(f"Descritores ({len(descritores)}): {', '.join(descritores)}")

print("\n" + "=" * 78)
print("BASE COMPLETA")
print("=" * 78)
rodada(base, descritores, SAIDA)

print("\n" + "=" * 78)
print("SEM AS LINHAS DE 'PAROU MAIS QUE O PELOTÃO'")
print("=" * 78)
# A média por grid é refeita: tirar linhas muda a média de cada posição de
# largada, e a coluna antiga deixaria de estar centrada.
sem = base[base["d_paradas"] != PAROU_MAIS].copy()
for ajustado, original in [("alvo_ajustado", "alvo"),
                           ("alvo_temporada_ajustado", "alvo_temporada"),
                           ("alvo_pos_bruta_ajustado", "alvo_pos_bruta")]:
    sem[ajustado] = sem[original] - sem.groupby("grid")[original].transform("mean")
rodada(sem, [d for d in descritores if d != "d_paradas"] + ["d_paradas"], SAIDA_SEM)
