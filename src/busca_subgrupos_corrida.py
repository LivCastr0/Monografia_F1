"""
Etapa 2 - Busca de subgrupos no grão de corrida (caminho B).

Pergunta: em que condições a posição de largada deixa de determinar a de chegada?

Entra: data/processed/base_corrida.csv
Sai:   data/processed/subgrupos_corrida.csv

Configuração: pysubgroup, StandardQFNumeric (a=0.5), profundidade 2, suporte mínimo
10 corridas, os dois sentidos. Profundidade e suporte são menores que no A1 porque
aqui são cerca de 160 linhas.

Corridas no mesmo circuito não são independentes. Por isso cada repetição do teste
faz duas coisas: embaralha o alvo entre as corridas de cada circuito, o que testa
chuva, safety car, era e paradas; e embaralha os perfis entre os circuitos, o que
testa o perfil. O maior valor de cada repetição define o limiar, descontando o
número de subgrupos testados.

Sucesso, fixado antes de rodar. Um subgrupo é FIRME se:
  1. passa do limiar;
  2. mantém o sinal nas duas eras (quando a era não é condição dele);
  3. nenhum circuito tem mais da metade das corridas dele;
  4. mantém o sinal sem o circuito mais frequente dele.

A resposta principal vem de rho_largada. trocas_por_volta é secundário.
"potência E com safety car" já foi visto no piloto: se reaparecer, é confirmação
nos mesmos dados, não descoberta nova.

Descoberta de subgrupos descreve associação, não causa.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pysubgroup as ps

ENTRADA = Path("data/processed/base_corrida.csv")
SAIDA = Path("data/processed/subgrupos_corrida.csv")

ALVOS = {"rho_largada": "principal", "trocas_por_volta": "secundário"}
SUPORTE_MINIMO = 10
PROFUNDIDADE = 2
PESO_TAMANHO = 0.5
PERMUTACOES = 5000
SEMENTE = 7
JA_VISTO = {"d_perfil": "potência", "d_safety_car": "com safety car"}


def condicoes(sg):
    return {s.attribute_name: s.attribute_value for s in sg.selectors}


def teste_perfil(df, alvo, rng):
    media = df[alvo].mean()

    def separacao(rotulos):
        g = df[alvo].groupby(rotulos).agg(["size", "mean"])
        return (g["size"] * (g["mean"] - media) ** 2).sum()

    mapa = df.drop_duplicates("circuito").set_index("circuito")["d_perfil"]
    circuitos, perfis = mapa.index.to_numpy(), mapa.to_numpy()
    observado = separacao(df["d_perfil"].to_numpy())
    maiores = 0
    for _ in range(PERMUTACOES):
        novo = dict(zip(circuitos, rng.permutation(perfis)))
        maiores += separacao(df["circuito"].map(novo).to_numpy()) >= observado
    return (maiores + 1) / (PERMUTACOES + 1)


def limiar_familia(df, alvo, grupos, rng):
    """Limiar de |z| para todos os subgrupos, por permutação."""
    y = df[alvo].to_numpy(dtype=float)
    circuito = df["circuito"].to_numpy()
    colunas = {c: df[c].to_numpy() for g in grupos for c in g if c != "d_perfil"}
    mapa = df.drop_duplicates("circuito").set_index("circuito")["d_perfil"]
    circuitos, perfis = mapa.index.to_numpy(), mapa.to_numpy()
    linhas_circuito = list(df.groupby("circuito").indices.values())

    resto = []
    for g in grupos:
        m = np.ones(len(df), dtype=bool)
        for c, v in g.items():
            if c != "d_perfil":
                m &= colunas[c] == v
        resto.append((m, g.get("d_perfil")))

    maximos = np.empty(PERMUTACOES)
    for i in range(PERMUTACOES):
        v = y.copy()
        for linhas in linhas_circuito:
            v[linhas] = y[rng.permutation(linhas)]
        novo = pd.Series(rng.permutation(perfis), index=circuitos)
        perfil = novo.reindex(circuito).to_numpy()
        centrado = v - v.mean()
        dp = v.std(ddof=1)
        maior = 0.0
        for m, valor in resto:
            cobre = m if valor is None else m & (perfil == valor)
            n = cobre.sum()
            if n >= SUPORTE_MINIMO:
                maior = max(maior, abs(centrado[cobre].sum()) / np.sqrt(n) / dp)
        maximos[i] = maior
    return np.quantile(maximos, 0.95)


def rodada(base, alvo, rng):
    df = base.dropna(subset=[alvo]).reset_index(drop=True)
    print("\n" + "=" * 78)
    print(f"ALVO {alvo} ({ALVOS[alvo]})")
    print("=" * 78)
    if len(df) < SUPORTE_MINIMO * 2:
        print(f"Só {len(df)} corridas com este alvo: rode coleta_voltas.py --todas.")
        return pd.DataFrame(columns=["firme"])
    print(f"Corridas: {len(df)}   circuitos: {df['circuito'].nunique()}")
    print(f"Média {df[alvo].mean():.3f}   desvio {df[alvo].std():.3f}\n")

    t = df.groupby("d_perfil").agg(corridas=(alvo, "size"), circuitos=("circuito", "nunique"),
                                   media=(alvo, "mean"), desvio=(alvo, "std"))
    print(t.to_string(float_format=lambda x: f"{x:.3f}"))
    p = teste_perfil(df, alvo, rng)
    print(f"\nO perfil sozinho muda o alvo? p = {p:.4f}")

    descritores = [c for c in df.columns if c.startswith("d_")]
    busca = df[descritores + [alvo]].copy()
    tarefa = ps.SubgroupDiscoveryTask(
        busca, ps.NumericTarget(alvo), ps.create_selectors(busca, ignore=[alvo]),
        result_set_size=100_000, depth=PROFUNDIDADE, qf=ps.StandardQFNumeric(a=PESO_TAMANHO),
        constraints=[ps.MinSupportConstraint(SUPORTE_MINIMO)],
    )
    achados = ps.Apriori().execute(tarefa).to_descriptions()
    grupos = [condicoes(sg) for _, sg in achados]
    limiar = limiar_familia(df, alvo, grupos, rng)

    y = df[alvo]
    linhas = []
    for (qualidade, sg), g in zip(achados, grupos):
        sub = df[np.asarray(sg.covers(busca))]
        n = len(sub)
        z = (sub[alvo].mean() - y.mean()) * np.sqrt(n) / y.std()
        sinal = np.sign(sub[alvo].mean() - y.mean())
        contagem = sub["circuito"].value_counts()
        sem_maior = sub[sub["circuito"] != contagem.index[0]]
        eras = sub.groupby("d_era")[alvo].mean()
        eras_ok = all(np.sign(e - y.mean()) == sinal for e in eras) if "d_era" not in g else True
        maior_pct = contagem.iloc[0] / n
        sem_ok = len(sem_maior) > 0 and np.sign(sem_maior[alvo].mean() - y.mean()) == sinal
        passou = abs(z) > limiar
        linhas.append({
            "alvo": alvo, "descricao": str(sg), "corridas": n,
            "circuitos": sub["circuito"].nunique(), "media": sub[alvo].mean(),
            "diferenca": sub[alvo].mean() - y.mean(), "z": z,
            "acima_do_limiar": passou, "sinal_nas_eras": eras_ok,
            "maior_circuito_pct": maior_pct, "sinal_sem_maior_circuito": sem_ok,
            "firme": passou and eras_ok and maior_pct <= 0.5 and sem_ok,
            "ja_visto_no_piloto": g == JA_VISTO, "qualidade": qualidade,
        })
    res = pd.DataFrame(linhas)
    res = res.reindex(res["z"].abs().sort_values(ascending=False).index)

    print(f"\nSubgrupos: {len(res)}   limiar |z| (5%, {PERMUTACOES} permutações): {limiar:.2f}")
    print(f"Acima do limiar: {int(res['acima_do_limiar'].sum())}   "
          f"firmes: {int(res['firme'].sum())}\n")
    print(res.head(12)[["descricao", "corridas", "circuitos", "media", "z",
                        "acima_do_limiar", "firme"]].to_string(
        index=False, float_format=lambda x: f"{x:+.3f}", max_colwidth=62))
    visto = res[res["ja_visto_no_piloto"]]
    if len(visto):
        v = visto.iloc[0]
        print(f"\nJá visto no piloto: {v['descricao']}: {v['corridas']} corridas, "
              f"média {v['media']:.3f}, z {v['z']:+.2f}, firme: {v['firme']}")
    return res


def main():
    base = pd.read_csv(ENTRADA)
    rng = np.random.default_rng(SEMENTE)
    partes = [rodada(base, alvo, rng) for alvo in ALVOS]
    pd.concat(partes, ignore_index=True).to_csv(SAIDA, index=False)
    firmes = partes[0]["firme"].sum()
    print("\n" + "=" * 78)
    print("CONCLUSÃO")
    print("=" * 78)
    if firmes:
        print(f"rho_largada: {int(firmes)} subgrupo(s) firme(s).")
    else:
        print("rho_largada: nenhum subgrupo firme.")
    print(f"\nSalvo em {SAIDA}")


if __name__ == "__main__":
    main()
