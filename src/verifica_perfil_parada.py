"""
Etapa 2 - Verifica o que a busca no grão de parada encontrou.

Entra: data/processed/base_parada.csv
       data/processed/subgrupos_parada.csv
Não salva nada e não usa internet.

Esta análise foi definida DEPOIS de ver o resultado da busca e deve ser relatada
assim. Ela faz duas perguntas que a busca não responde.

1. O efeito é da parada ou da conta?
   O saldo é uma soma de duelos, e quem para cedo no meio do pelotão tende a ter
   mais duelos. Mais duelos dão um saldo maior mesmo com a mesma chance de ganhar
   cada um. Por isso a medida aqui é `por_duelo`: saldo ÷ número de duelos.

2. O perfil do circuito acrescenta alguma coisa?
   É a pergunta da monografia. Um subgrupo como "perto do carro da frente E
   potência" pode só herdar o efeito de "perto do carro da frente". O teste mede
   se a diferença entre as faixas de um descritor muda de um perfil para outro.
   A significância vem de embaralhar os perfis entre os circuitos.

Regra fixada antes de rodar: são 4 testes, então o perfil só acrescenta se algum
deles tiver p < 0,0125 (0,05 ÷ 4).
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("data/processed/base_parada.csv")
SUBGRUPOS = Path("data/processed/subgrupos_parada.csv")

MEDIDA = "por_duelo"
TESTADOS = ["d_distancia", "d_fase", "d_posicao", "d_troca_pneu"]
PERMUTACOES = 5000
SEMENTE = 5
LIMITE_P = 0.05 / len(TESTADOS)
QUANTOS_MOSTRAR = 19
ORDEM_PERFIL = ["alto downforce", "stop-and-go", "potência", "urbano-técnico"]


def interacao(y, faixa, perfil):
    """Soma dos desvios que nem a faixa nem o perfil explicam sozinhos."""
    d = pd.DataFrame({"y": y, "f": faixa, "p": perfil})
    celula = d.groupby(["f", "p"])["y"].agg(["size", "mean"]).reset_index()
    geral = d["y"].mean()
    media_f = celula["f"].map(d.groupby("f")["y"].mean())
    media_p = celula["p"].map(d.groupby("p")["y"].mean())
    desvio = celula["mean"] - media_f - media_p + geral
    return (celula["size"] * desvio ** 2).sum()


def testa(df, col, rng):
    y = df[MEDIDA].to_numpy()
    faixa = df[col].to_numpy()
    mapa = df.drop_duplicates("circuito").set_index("circuito")["d_perfil"]
    circuitos, perfis = mapa.index.to_numpy(), mapa.to_numpy()
    observado = interacao(y, faixa, df["d_perfil"].to_numpy())
    maiores = 0
    for _ in range(PERMUTACOES):
        novo = dict(zip(circuitos, rng.permutation(perfis)))
        maiores += interacao(y, faixa, df["circuito"].map(novo).to_numpy()) >= observado
    return (maiores + 1) / (PERMUTACOES + 1)


def filtro(df, descricao):
    """Transforma a descrição de um subgrupo em linhas da base."""
    mascara = pd.Series(True, index=df.index)
    for col, valor in re.findall(r"(d_\w+)=='([^']*)'", descricao):
        mascara &= df[col] == valor
    return mascara


def main():
    df = pd.read_csv(BASE)
    df[MEDIDA] = df["saldo"] / df["duelos"]
    rng = np.random.default_rng(SEMENTE)
    formato = lambda x: f"{x:+.3f}"

    print("=" * 78)
    print("1. O EFEITO É DA PARADA OU DA CONTA?")
    print("=" * 78)
    print(f"Geral: saldo {df['saldo'].mean():+.3f}, por duelo {df[MEDIDA].mean():+.3f}, "
          f"{df['duelos'].mean():.2f} duelos por parada\n")
    for col in ["d_distancia", "d_fase", "d_posicao"]:
        t = df.groupby(col).agg(paradas=("saldo", "size"), duelos=("duelos", "mean"),
                                saldo=("saldo", "mean"), por_duelo=(MEDIDA, "mean"))
        print(t.to_string(float_format=formato) + "\n")

    print("=" * 78)
    print("2. O PERFIL ACRESCENTA ALGUMA COISA?")
    print("=" * 78)
    resultados = {}
    for col in TESTADOS:
        p = testa(df, col, rng)
        resultados[col] = p
        print(f"\n{col}: p = {p:.4f}  (limite {LIMITE_P:.4f})")
        print("Saldo por duelo:")
        medias = df.pivot_table(index=col, columns="d_perfil", values=MEDIDA, aggfunc="mean")
        print(medias.reindex(columns=ORDEM_PERFIL).to_string(float_format=formato))
        print("Paradas:")
        n = df.pivot_table(index=col, columns="d_perfil", values=MEDIDA, aggfunc="size")
        print(n.reindex(columns=ORDEM_PERFIL).fillna(0).astype(int).to_string())

    print("\n" + "=" * 78)
    print("3. OS SUBGRUPOS COM PERFIL CONTRA OS MESMOS SEM PERFIL")
    print("=" * 78)
    sub = pd.read_csv(SUBGRUPOS)
    com = sub[sub["acima_do_limiar"] & sub["descricao"].str.contains("d_perfil", regex=False)]
    linhas = []
    for d in com["descricao"]:
        sem = " AND ".join(c for c in d.split(" AND ") if "d_perfil" not in c)
        if not sem:
            continue
        m_com, m_sem = filtro(df, d), filtro(df, sem)
        perfil = re.search(r"d_perfil=='([^']*)'", d).group(1)
        linhas.append({"perfil": perfil, "condicoes": re.sub(r"d_\w+==|'", "", sem),
                       "paradas": int(m_com.sum()),
                       "por_duelo_com_perfil": df.loc[m_com, MEDIDA].mean(),
                       "por_duelo_sem_perfil": df.loc[m_sem, MEDIDA].mean()})
    if linhas:
        t = pd.DataFrame(linhas)
        t["diferenca"] = t["por_duelo_com_perfil"] - t["por_duelo_sem_perfil"]
        t = t.sort_values("diferenca", key=abs, ascending=False)
        print(f"{len(t)} subgrupos. Diferença perto de zero = só herdou o efeito.\n")
        print(t.head(QUANTOS_MOSTRAR).to_string(index=False, float_format=formato,
                                                max_colwidth=46))
    else:
        print("Nenhum subgrupo com perfil acima do limiar.")

    menor = min(resultados, key=resultados.get)
    print("\n" + "=" * 78)
    print("CONCLUSÃO DA REGRA")
    print("=" * 78)
    if resultados[menor] < LIMITE_P:
        print(f"O perfil acrescenta: {menor}, p = {resultados[menor]:.4f}.")
    else:
        print(f"O perfil não acrescenta: menor p = {resultados[menor]:.4f} ({menor}), "
              f"limite {LIMITE_P:.4f}.")
    print("\nNada foi salvo.")


if __name__ == "__main__":
    main()
