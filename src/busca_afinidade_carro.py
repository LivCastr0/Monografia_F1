"""
Etapa 2 - Hipótese (ii): afinidade carro × perfil pelo ritmo de corrida.

Pergunta: que carros andam acima do próprio padrão em cada perfil de pista?

Entra: data/processed/base_ritmo.csv, data/processed/base_ritmo_pilotos.csv
Sai:   data/processed/subgrupos_afinidade.csv

Não usa internet. Todos os testes embaralham os perfis entre os 26 circuitos.

Testes, fixados antes de rodar:

1. TESTE GERAL (o principal): a afinidade depende do perfil, carro a carro, mais do
   que o acaso? Estatística: soma, entre os pares carro × perfil com 2+ corridas, de
   (corridas × afinidade média²). Afinidade média de cada carro é zero por construção,
   então a estatística só cresce se cada carro render diferente conforme o perfil.

2. A BUSCA: pysubgroup, StandardQFNumeric (a=0.5), profundidade 2, descritores carro e
   perfil; concorrem só os pares carro E perfil com 3+ corridas de 2+ circuitos. O
   maior |z| de cada repetição define o limiar.
   Firme: passa do limiar; os dois pilotos do carro têm o mesmo sentido; mesmo sentido
   sem o circuito mais frequente.
   O urbano-técnico quase nunca chega a 3 corridas secas de um carro numa temporada.

3. PERSISTÊNCIA (descritiva): a afinidade de uma equipe por perfil num ano se parece
   com a do ano seguinte? Correlação de Spearman entre anos seguidos, com p por
   permutação.

Descoberta de subgrupos descreve associação, não causa.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pysubgroup as ps

ENTRADA = Path("data/processed/base_ritmo.csv")
PILOTOS = Path("data/processed/base_ritmo_pilotos.csv")
SAIDA = Path("data/processed/subgrupos_afinidade.csv")
ALVO = "afinidade"
SUPORTE, MIN_CIRCUITOS = 3, 2
PERMUTACOES, SEMENTE = 5000, 17
EQUIPES = {"Force India": "Aston Martin", "Racing Point": "Aston Martin",
           "Toro Rosso": "Racing Bulls", "AlphaTauri": "Racing Bulls", "RB": "Racing Bulls",
           "Alfa Romeo Racing": "Sauber", "Alfa Romeo": "Sauber", "Kick Sauber": "Sauber",
           "Renault": "Alpine"}


def main():
    df = pd.read_csv(ENTRADA).reset_index(drop=True)
    pil = pd.read_csv(PILOTOS)
    rng = np.random.default_rng(SEMENTE)
    y = df[ALVO].to_numpy(float)
    dp = y.std(ddof=1)
    carro = df["d_carro"].to_numpy()
    circ = df["circuito"].to_numpy()
    mapa = df.drop_duplicates("circuito").set_index("circuito")["d_perfil"]
    print(f"Base: {len(df)} carros-corrida, {df['d_carro'].nunique()} carros, "
          f"{df['circuito'].nunique()} circuitos. Desvio da afinidade {dp:.3f}%\n")

    def pares(perfis):
        g = pd.DataFrame({"c": carro, "p": perfis, "y": y, "circ": circ}).groupby(["c", "p"])
        return g.agg(n=("y", "size"), media=("y", "mean"), circuitos=("circ", "nunique"))

    def estatistica(perfis):
        t = pares(perfis)
        t = t[t["n"] >= 2]
        return (t["n"] * t["media"] ** 2).sum()

    def maior_z(perfis):
        t = pares(perfis)
        t = t[(t["n"] >= SUPORTE) & (t["circuitos"] >= MIN_CIRCUITOS)]
        return (t["media"].abs() * np.sqrt(t["n"]) / dp).max() if len(t) else 0.0

    print("=" * 78 + "\n1. TESTE GERAL\n" + "=" * 78)
    obs = estatistica(df["d_perfil"].to_numpy())
    nulos, maximos = np.empty(PERMUTACOES), np.empty(PERMUTACOES)
    for i in range(PERMUTACOES):
        novo = pd.Series(rng.permutation(mapa.to_numpy()), index=mapa.index)
        perfis = df["circuito"].map(novo).to_numpy()
        nulos[i] = estatistica(perfis)
        maximos[i] = maior_z(perfis)
    p_geral = ((nulos >= obs).sum() + 1) / (PERMUTACOES + 1)
    print(f"Estatística observada {obs:.2f}; média no acaso {nulos.mean():.2f}")
    print(f"A afinidade depende do perfil, carro a carro? p = {p_geral:.4f}\n")

    print("=" * 78 + "\n2. A BUSCA\n" + "=" * 78)
    dados = df[["d_carro", "d_perfil", ALVO]]
    tarefa = ps.SubgroupDiscoveryTask(
        dados, ps.NumericTarget(ALVO), ps.create_selectors(dados, ignore=[ALVO]),
        result_set_size=100_000, depth=2, qf=ps.StandardQFNumeric(a=0.5),
        constraints=[ps.MinSupportConstraint(SUPORTE)])
    grupos = [{s.attribute_name: s.attribute_value for s in sg.selectors}
              for _, sg in ps.Apriori().execute(tarefa).to_descriptions()]
    grupos = [g for g in grupos if len(g) == 2]
    limiar = np.quantile(maximos, 0.95)

    linhas = []
    for g in grupos:
        m = (carro == g["d_carro"]) & (df["d_perfil"].to_numpy() == g["d_perfil"])
        sub = df[m]
        if len(sub) < SUPORTE or sub["circuito"].nunique() < MIN_CIRCUITOS:
            continue
        media = sub[ALVO].mean()
        z = media * np.sqrt(len(sub)) / dp
        s = np.sign(media)
        n = sub["circuito"].value_counts()
        sem = sub[sub["circuito"] != n.index[0]]
        ps_ = pil[(pil["d_carro"] == g["d_carro"]) & (pil["d_perfil"] == g["d_perfil"])]
        por_piloto = ps_.groupby("piloto")[ALVO].agg(["size", "mean"])
        por_piloto = por_piloto[por_piloto["size"] >= 2]
        pilotos_ok = len(por_piloto) >= 2 and all(np.sign(por_piloto["mean"]) == s)
        sem_ok = len(sem) > 0 and np.sign(sem[ALVO].mean()) == s
        passou = abs(z) > limiar
        linhas.append({"carro": g["d_carro"], "perfil": g["d_perfil"], "corridas": len(sub),
                       "circuitos": sub["circuito"].nunique(), "afinidade_media": media,
                       "z": z, "acima_do_limiar": passou,
                       "firme": passou and pilotos_ok and sem_ok,
                       "pilotos": "; ".join(f"{k} {v:+.2f}" for k, v in por_piloto["mean"].items()),
                       "lista_circuitos": ", ".join(sorted(n.index))})
    res = pd.DataFrame(linhas).sort_values("z", key=abs, ascending=False)
    print(f"Pares carro E perfil que concorrem: {len(res)}")
    print(f"Limiar |z| (5%, {PERMUTACOES} permutações): {limiar:.2f}")
    print(f"Acima do limiar: {int(res['acima_do_limiar'].sum())}   firmes: {int(res['firme'].sum())}\n")
    print(res.head(15)[["carro", "perfil", "corridas", "circuitos", "afinidade_media", "z",
                        "firme", "pilotos"]].to_string(index=False, float_format=lambda x: f"{x:+.2f}",
                                                        max_colwidth=40))
    res.to_csv(SAIDA, index=False)

    print("\n" + "=" * 78 + "\n3. PERSISTÊNCIA ENTRE ANOS\n" + "=" * 78)
    df["equipe_atual"] = df["equipe"].replace(EQUIPES)
    t = df.groupby(["equipe_atual", "d_perfil", "ano"])[ALVO].mean().unstack("ano")

    def correl(tab):
        a, b = [], []
        anos = sorted(tab.columns)
        for x, z_ in zip(anos[:-1], anos[1:]):
            m = tab[[x, z_]].dropna()
            a += list(m[x])
            b += list(m[z_])
        return pd.Series(a).corr(pd.Series(b), method="spearman"), len(a)

    rho, npares = correl(t)
    nulo = []
    for _ in range(1000):
        tp = t.copy()
        for col in tp.columns:
            tp[col] = rng.permutation(tp[col].to_numpy())
        nulo.append(correl(tp)[0])
    p_pers = (np.sum(np.array(nulo) >= rho) + 1) / 1001
    print(f"Equipe × perfil, ano seguinte: Spearman {rho:+.3f} ({npares} pares), p = {p_pers:.4f}")
    print(f"\nSalvo em {SAIDA}")


if __name__ == "__main__":
    main()
