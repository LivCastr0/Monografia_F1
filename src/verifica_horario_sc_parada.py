"""
Etapa 2 - Decisão de parar: verificação do horário do safety car.

Problema: em monta_base_decisao_parada.py a neutralização é a da própria volta L, e
vale para a volta inteira, juntando a bandeira de todos os pilotos naquela volta. Uma
parada feita antes da entrada do safety car (ou VSC) naquela volta conta como "sob SC"
e pode inflar o padrão mais forte da busca.

Pergunta fixada antes de ver os números:
  1. Das paradas marcadas como "safety car ou VSC", quantas aconteceram com a
     neutralização já ativa na entrada do box, quantas antes de ela começar e quantas
     depois de ela acabar.
  2. O padrão se mantém quando a neutralização é medida no horário de cada piloto?

Medida corrigida (a mesma regra para quem parou e para quem não parou):
  ponto de decisão = entrada do box (quem parou) ou fim da volta L (quem não parou).
  - versão A, "no ponto de decisão": SC ou VSC ativo naquele instante.
  - versão B, "visto antes da decisão": SC ou VSC ativo em algum momento entre o fim da
    volta L-1 do próprio piloto e o ponto de decisão. Inclui a parada feita logo depois
    de o SC sair da pista, que a equipe decidiu durante o SC.
  A bandeira vermelha (código 5) não conta como neutralização, como na base original.
  A assimetria (quem não parou é medido uns segundos depois, na linha) favorece
  "não neutralizado" para quem parou, então a correção é conservadora.

Entra: data/processed/base_decisao_parada.csv
       data/interim/voltas/*_voltas.csv
       data/interim/corridas/*_bandeiras.csv
Sai:   data/processed/verificacao_horario_sc_parada.csv (base + colunas corrigidas)

Situação: verificação feita uma vez, sobre a base ANTERIOR à correção (121.172 voltas).
A versão A foi depois incorporada a monta_base_decisao_parada.py, e a base atual
(base_decisao_parada.csv, 121.075 voltas) já sai corrigida e sem as voltas de bandeira
vermelha no ponto de decisão. A saída deste script é registro da verificação, não base
de análise. Rodar de novo sobre a base atual não reproduz a verificação.

Não usa internet. O z é o da busca: (taxa do subgrupo − taxa geral) × √n / desvio.
"""
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("data/processed/base_decisao_parada.csv")
VOLTAS = Path("data/interim/voltas")
CORRIDAS = Path("data/interim/corridas")
SAIDA = Path("data/processed/verificacao_horario_sc_parada.csv")
NEUTRO = {4, 6, 7}
SC = "safety car ou VSC"
LIMIAR_BUSCA = 8.28   # |z| da busca original (500 permutações); referência, não recalculado


def status_em(tempos, codigos, t):
    """Código de bandeira vigente no instante t (último aviso até t)."""
    i = np.searchsorted(tempos, t, side="right") - 1
    return codigos[max(i, 0)]


def neutro_no_intervalo(tempos, codigos, ini, fim):
    """SC/VSC ativo em algum momento de (ini, fim]."""
    if status_em(tempos, codigos, ini) in NEUTRO:
        return True
    m = (tempos > ini) & (tempos <= fim)
    return bool(np.isin(codigos[m], list(NEUTRO)).any())


def corrida(d, ano, rnd):
    v = pd.read_csv(VOLTAS / f"{ano}_{rnd:02d}_voltas.csv")
    b = pd.read_csv(CORRIDAS / f"{ano}_{rnd:02d}_bandeiras.csv").sort_values("tempo_s")
    tempos, codigos = b["tempo_s"].to_numpy(), b["codigo"].to_numpy()
    v["volta"] = v["volta"].astype(int)
    fim = v.set_index(["piloto", "volta"])["fim_volta_s"]
    entrada = v.set_index(["piloto", "volta"])["entrada_box_s"]

    out = []
    for r in d.itertuples(index=False):
        ini = fim.get((r.piloto, r.volta - 1), np.nan)
        fl = fim.get((r.piloto, r.volta), np.nan)
        ent = entrada.get((r.piloto, r.volta), np.nan) if r.parou else np.nan
        decisao = ent if r.parou and pd.notna(ent) else fl
        if pd.isna(ini) or pd.isna(decisao):
            out.append((np.nan, np.nan, "sem horário", ent))
            continue
        a = status_em(tempos, codigos, decisao) in NEUTRO
        bb = neutro_no_intervalo(tempos, codigos, ini, decisao)
        if r.parou and r.d_neutralizacao == SC:
            if a:
                cat = "SC/VSC ativo na entrada do box"
            elif bb:
                cat = "SC/VSC acabou antes da entrada"
            elif neutro_no_intervalo(tempos, codigos, decisao, fl if pd.notna(fl) else decisao + 60):
                cat = "SC/VSC começou depois da entrada"
            else:
                cat = "sem SC/VSC na volta do próprio piloto"
        else:
            cat = ""
        out.append((a, bb, cat, ent))
    d = d.copy()
    d["neutro_A"], d["neutro_B"], d["categoria_parada_sc"], d["entrada_box_s"] = zip(*out)
    return d


def z(y, m, base, dp):
    n = m.sum()
    return (y[m].mean() - base) * np.sqrt(n) / dp if n else np.nan


def tabela(df, col):
    y = df["parou"].to_numpy()
    base, dp = y.mean(), y.std()
    neutro = df[col].to_numpy(dtype=bool)
    casos = [
        ("SC/VSC", neutro),
        ("SC/VSC E 0 paradas", neutro & (df["d_paradas_feitas"] == "0 paradas").to_numpy()),
        ("SC/VSC E pneu 30+ voltas", neutro & (df["d_idade_pneu"] == "30+ voltas").to_numpy()),
        ("bandeira verde", ~neutro),
    ]
    linhas = []
    for nome, m in casos:
        linhas.append({"subgrupo": nome, "voltas": int(m.sum()), "paradas": int(y[m].sum()),
                       "taxa_%": 100 * y[m].mean(), "z": z(y, m, base, dp),
                       "circuitos": df.loc[m, "circuito"].nunique()})
    return pd.DataFrame(linhas)


def main():
    df = pd.read_csv(BASE)
    partes = [corrida(g, a, r) for (a, r), g in df.groupby(["ano", "round"])]
    df = pd.concat(partes).sort_index()
    df.to_csv(SAIDA, index=False)

    orig = df["d_neutralizacao"] == SC
    print(f"Base: {len(df)} voltas, {int(df['parou'].sum())} paradas, taxa geral "
          f"{100 * df['parou'].mean():.2f}%")
    print(f"Sem horário (fora das tabelas corrigidas): {int(df['neutro_A'].isna().sum())} voltas")

    print("\n1. PARADAS MARCADAS COMO 'SAFETY CAR OU VSC' NA BASE ORIGINAL")
    p = df[orig & (df["parou"] == 1)]
    print(f"   {len(p)} paradas")
    print(p["categoria_parada_sc"].value_counts().to_string())

    print("\n   Voltas sem parada marcadas como SC/VSC, mas sem SC/VSC no intervalo do "
          "próprio piloto (versão B):",
          int((orig & (df["parou"] == 0) & (df["neutro_B"] == False)).sum()), "de",
          int((orig & (df["parou"] == 0)).sum()))

    ok = df["neutro_A"].notna()
    print("\n2. O PADRÃO CORRIGIDO (taxa por volta; limiar da busca original |z| "
          f"{LIMIAR_BUSCA})")
    df_o = df[ok].assign(orig=orig[ok])
    for col, nome in [("orig", "original (volta inteira, todos os pilotos)"),
                      ("neutro_B", "versão B: SC/VSC visto antes da decisão"),
                      ("neutro_A", "versão A: SC/VSC ativo no ponto de decisão")]:
        print(f"\n   {nome}")
        print(tabela(df_o, col).to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print("\n3. ROBUSTEZ DA VERSÃO A (mesmo sentido nas eras e sem o circuito mais frequente)")
    m = df_o["neutro_A"].astype(bool)
    for era, g in df_o.groupby("d_era"):
        mm = g["neutro_A"].astype(bool)
        print(f"   {era}: SC/VSC {100 * g.loc[mm, 'parou'].mean():.2f}% contra "
              f"{100 * g['parou'].mean():.2f}% geral ({int(mm.sum())} voltas)")
    top = df_o.loc[m, "circuito"].value_counts()
    g = df_o[df_o["circuito"] != top.index[0]]
    mm = g["neutro_A"].astype(bool)
    print(f"   sem {top.index[0]} ({100 * top.iloc[0] / top.sum():.1f}% das voltas SC/VSC): "
          f"{100 * g.loc[mm, 'parou'].mean():.2f}% contra {100 * g['parou'].mean():.2f}%")

    print(f"\nSalvo em {SAIDA}")


if __name__ == "__main__":
    main()
