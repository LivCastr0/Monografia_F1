"""
Etapa 2 - Opção 1: a assinatura de cada cluster na corrida.

Pergunta: que comportamentos de corrida distinguem cada tipo de circuito?

Entra: data/processed/base_piloto_corrida.csv
       data/processed/base_corrida.csv     (monta_base_corrida.py)
       data/processed/base_duelos.csv      (monta_base_duelos.py)
       data/processed/base_stint.csv       (monta_base_stint.py)
Sai:   data/processed/base_assinatura.csv
       data/processed/subgrupos_assinatura.csv

Não usa internet.

Grão: corrida seca. Uma busca para cada cluster, com o alvo "a corrida é de um
circuito deste cluster" (0 ou 1). A tipologia veio da classificação; aqui se
verifica se ela tem uma assinatura no que acontece na corrida, que é outro dado.

Descritores, em terços das corridas (baixa / média / alta):
  d_ultrapassagem  taxa de ultrapassagem por volta de duelo
  d_trocas         trocas de posição por volta sob bandeira verde
  d_largada        correlação entre largada e chegada
  d_paradas        número médio de paradas de quem terminou
  d_stint          maior stint mediano, como fração da corrida
  d_degradacao     degradação mediana do pneu nos stints válidos
  d_abandono       fração de quem largou e não terminou
e mais d_safety_car (sim ou não) e d_era (2018-21 ou 2022+).
Corridas sem alguma das medidas ficam de fora.

Busca: pysubgroup, StandardQFNumeric (a=0.5), profundidade 2, suporte mínimo 10
corridas, os dois sentidos.

Teste: os clusters são embaralhados entre os 26 circuitos, mantendo o tamanho de
cada cluster. O maior valor de cada repetição define o limiar.

Para concorrer (no resultado real e em cada repetição), o subgrupo precisa ter:
  - pelo menos MIN_CORRIDAS_CLUSTER corridas do cluster;
  - corridas de pelo menos 2 circuitos do cluster;
  - nenhum circuito com mais de 60% das corridas do cluster no subgrupo.
Firme, fixado antes de rodar: passa do limiar; mesmo sentido nas duas eras (quando a
era não é condição); mesmo sentido sem o circuito mais frequente do cluster.

Para os subgrupos firmes de 2 condições, a saída mostra também a fração do cluster em
cada condição sozinha. Se uma delas sozinha já dá quase o mesmo, a outra pode ser só
um recorte por acaso.

Ressalva: o urbano-técnico tem 2 circuitos, então qualquer resultado nele é frágil.
Descoberta de subgrupos descreve associação, não causa.
"""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pysubgroup as ps

PILOTO = Path("data/processed/base_piloto_corrida.csv")
CORRIDA = Path("data/processed/base_corrida.csv")
DUELOS = Path("data/processed/base_duelos.csv")
STINTS = Path("data/processed/base_stint.csv")
SAIDA_BASE = Path("data/processed/base_assinatura.csv")
SAIDA = Path("data/processed/subgrupos_assinatura.csv")

NOMES = {0: "alto downforce", 1: "stop-and-go", 2: "potência", 3: "urbano-técnico"}
MEDIDAS = {"d_ultrapassagem": "taxa_ultrapassagem", "d_trocas": "trocas_por_volta",
           "d_largada": "rho_largada", "d_paradas": "paradas", "d_stint": "stint_longo",
           "d_degradacao": "degradacao", "d_abandono": "abandono"}
SUPORTE, PROFUNDIDADE, PERMUTACOES, SEMENTE = 10, 2, 2000, 15
MIN_CORRIDAS_CLUSTER, MIN_CIRCUITOS, MAX_CIRCUITO = 5, 2, 0.60


def monta():
    b = pd.read_csv(PILOTO)
    b = b[~b["corrida_sem_disputa"] & (b["faixa_chuva"] == "seco")]
    largou = b[~b["nao_largou"]]
    fin = largou[largou["terminou"]].copy()
    fin["fracao"] = fin["stint_mais_longo"] / fin["voltas_corrida"]
    chave = ["ano", "round"]
    df = largou.groupby(chave).agg(circuito=("circuito", "first"), cluster=("cluster", "first"),
                                   sc=("n_periodos_sc", "first"),
                                   abandono=("terminou", lambda s: 1 - s.mean()))
    df = df.join(fin.groupby(chave).agg(paradas=("n_paradas", "mean"),
                                        stint_longo=("fracao", "median")))
    c = pd.read_csv(CORRIDA).set_index(chave)[["rho_largada", "trocas_por_volta"]]
    d = pd.read_csv(DUELOS).groupby(chave)["passou"].mean().rename("taxa_ultrapassagem")
    s = pd.read_csv(STINTS).groupby(chave)["deg_pct"].median().rename("degradacao")
    df = df.join(c).join(d).join(s).reset_index()
    antes = len(df)
    df = df.dropna(subset=list(MEDIDAS.values())).reset_index(drop=True)
    print(f"Corridas secas: {antes}; com todas as medidas: {len(df)}; "
          f"circuitos: {df['circuito'].nunique()}")
    df["perfil"] = df["cluster"].map(NOMES)
    df["d_era"] = np.where(df["ano"] >= 2022, "2022+", "2018-21")
    df["d_safety_car"] = np.where(df["sc"] > 0, "com safety car", "sem safety car")
    for d, m in MEDIDAS.items():
        q1, q2 = df[m].quantile([1 / 3, 2 / 3])
        df[d] = np.where(df[m] <= q1, "baixa", np.where(df[m] <= q2, "média", "alta"))
    df.to_csv(SAIDA_BASE, index=False)
    return df


def descritiva(df):
    print("\n" + "=" * 78 + "\n1. AS MEDIDAS POR CLUSTER (medianas)\n" + "=" * 78)
    t = df.groupby("perfil")[list(MEDIDAS.values())].median()
    t.insert(0, "corridas", df.groupby("perfil").size())
    t.insert(1, "circuitos", df.groupby("perfil")["circuito"].nunique())
    print(t.T.to_string(float_format=lambda x: f"{x:.3f}"))
    print(f"\nCorridas com safety car: "
          + "  ".join(f"{p} {v:.0%}" for p, v in
                      df.groupby("perfil")["d_safety_car"].apply(lambda s: (s == "com safety car").mean()).items()))


def main():
    df = monta()
    descritiva(df)
    descritores = [c for c in df.columns if c.startswith("d_")]
    rng = np.random.default_rng(SEMENTE)

    # Todos os subgrupos de 1 e 2 condições com suporte; a cobertura não muda nas
    # permutações, só o alvo.
    conds = sorted({(c, v) for c in descritores for v in df[c].unique()})
    grupos = []
    for k in range(1, PROFUNDIDADE + 1):
        for cb in combinations(conds, k):
            if len({c for c, _ in cb}) < k:
                continue
            m = np.logical_and.reduce([df[c].to_numpy() == v for c, v in cb])
            if SUPORTE <= m.sum() < len(df):
                grupos.append((cb, m))
    cobre = np.array([m for _, m in grupos])
    n = cobre.sum(axis=1)
    codigo, circuitos = pd.factorize(df["circuito"])
    um = np.zeros((len(df), len(circuitos)))
    um[np.arange(len(df)), codigo] = 1
    por_circuito = cobre.astype(float) @ um          # corridas de cada circuito no subgrupo
    cluster_do_circuito = df.drop_duplicates("circuito").set_index("circuito")["perfil"] \
        .reindex(circuitos).to_numpy()

    def avalia(rotulo_circuito, alvo):
        """z de cada subgrupo para o alvo 'é do cluster alvo', e quem concorre."""
        e = rotulo_circuito == alvo
        y = e[codigo].astype(float)
        do_cluster = por_circuito[:, e]
        total = do_cluster.sum(axis=1)
        ok = ((total >= MIN_CORRIDAS_CLUSTER) & ((do_cluster > 0).sum(axis=1) >= MIN_CIRCUITOS)
              & (do_cluster.max(axis=1, initial=0) <= MAX_CIRCUITO * np.maximum(total, 1)))
        z = np.where(ok, (cobre @ (y - y.mean())) / np.sqrt(n) / y.std(ddof=1), 0.0)
        return z, ok, y

    print("\n" + "=" * 78 + "\n2. A BUSCA, UMA POR CLUSTER\n" + "=" * 78)
    print(f"Subgrupos com {SUPORTE}+ corridas: {len(grupos)}")
    linhas = []
    for alvo in NOMES.values():
        z, ok, y = avalia(cluster_do_circuito, alvo)
        maximos = np.array([np.abs(avalia(rng.permutation(cluster_do_circuito), alvo)[0]).max()
                            for _ in range(PERMUTACOES)])
        limiar = np.quantile(maximos, 0.95)
        # Confere com o pysubgroup que a medida é a mesma da biblioteca
        tarefa = ps.SubgroupDiscoveryTask(
            df[descritores].assign(alvo=y), ps.NumericTarget("alvo"),
            ps.create_selectors(df[descritores]), result_set_size=5, depth=PROFUNDIDADE,
            qf=ps.StandardQFNumeric(a=0.5), constraints=[ps.MinSupportConstraint(SUPORTE)])
        melhor_ps = ps.Apriori().execute(tarefa).to_descriptions()[0][1]

        media = y.mean()
        for k in np.where(ok)[0]:
            cb, m = grupos[k]
            sub = df[m]
            taxa = y[m].mean()
            sinal = np.sign(taxa - media)
            eras_ok = True
            if not any(c == "d_era" for c, _ in cb):
                for era, g in df.groupby("d_era"):
                    ye = (g["perfil"] == alvo).astype(float)
                    me = m[g.index]
                    if me.sum() == 0 or np.sign(ye[me].mean() - ye.mean()) != sinal:
                        eras_ok = False
            do_cluster = sub[sub["perfil"] == alvo]["circuito"].value_counts()
            sem = sub[sub["circuito"] != do_cluster.index[0]]
            y_sem = (df.loc[df["circuito"] != do_cluster.index[0], "perfil"] == alvo).mean()
            sem_ok = len(sem) > 0 and np.sign((sem["perfil"] == alvo).mean() - y_sem) == sinal
            passou = abs(z[k]) > limiar
            sozinhas = "  ".join(
                f"{c[2:]}={v}: {y[df[c].to_numpy() == v].mean():.0%}" for c, v in cb) if len(cb) > 1 else ""
            linhas.append({"cada_condicao_sozinha": sozinhas,"cluster": alvo, "descricao": " E ".join(f"{c[2:]}={v}" for c, v in cb),
                           "corridas": int(n[k]), "do_cluster": int((sub["perfil"] == alvo).sum()),
                           "circuitos_do_cluster": len(do_cluster),
                           "fracao_do_cluster": taxa, "fracao_geral": media, "z": z[k],
                           "limiar": limiar, "acima_do_limiar": passou,
                           "firme": passou and eras_ok and sem_ok,
                           "circuitos": ", ".join(sorted(do_cluster.index))})
        r = pd.DataFrame([l for l in linhas if l["cluster"] == alvo])
        print(f"\n--- {alvo}: {int(y.sum())} de {len(y)} corridas ({media:.0%}); "
              f"concorrem {int(ok.sum())}; limiar |z| {limiar:.2f}; "
              f"acima {int(r['acima_do_limiar'].sum()) if len(r) else 0}; "
              f"firmes {int(r['firme'].sum()) if len(r) else 0}")
        print(f"    melhor subgrupo no pysubgroup (sem as exigências de circuito): {melhor_ps}")
        if len(r):
            r = r.sort_values("z", key=abs, ascending=False)
            print(r.head(8)[["descricao", "corridas", "do_cluster", "circuitos_do_cluster",
                             "fracao_do_cluster", "z", "firme"]].to_string(
                index=False, float_format=lambda x: f"{x:+.2f}", max_colwidth=50))
            f = r[r["firme"] & (r["cada_condicao_sozinha"] != "")]
            if len(f):
                print("    Firmes de 2 condições, com a fração do cluster em cada condição sozinha:")
                for _, l in f.head(8).iterrows():
                    print(f"      {l['descricao']} ({l['fracao_do_cluster']:.0%}) -> {l['cada_condicao_sozinha']}")

    res = pd.DataFrame(linhas)
    res.to_csv(SAIDA, index=False)
    print("\n" + "=" * 78 + "\nCONCLUSÃO\n" + "=" * 78)
    for alvo in NOMES.values():
        f = res[(res["cluster"] == alvo) & res["firme"]] if len(res) else res
        print(f"{alvo}: {len(f)} subgrupo(s) firme(s)")
    print(f"\nSalvo em {SAIDA}")


if __name__ == "__main__":
    main()
