"""
Etapa 2 - Tarefa 4: poder do teste de partições nos dados reais.

Pergunta: qual é a menor diferença de efeito entre perfis que o teste de partições da
tarefa 3 consegue detectar? Sem isso, "não detectamos diferença" não diz se a
diferença não existe ou se o teste não a veria.

Entra: data/processed/base_duelos.csv, base_decisao_parada.csv
       data/processed/regras_globais_duelos.csv, regras_globais_parada.csv
Sai:   data/processed/abrangencia/poder_<analise>.csv

Desenho fixado antes de rodar
-----------------------------
- Regras: as regras globais acima da média, menos a regra marginal dos duelos (pneu 10+
  mais novo e primeiro terço, abaixo do suporte em todos os perfis).
- Tipologia sorteada (ajuste depois da primeira rodada, commit 80ef1ca; declarar): em
  cada repetição, os 26 circuitos são divididos ao acaso com os tamanhos reais
  (2/5/8/11), e o efeito é plantado num dos grupos de 5, 8 ou 11. O teste compara essa
  divisão com as partições aleatórias de referência, como na tarefa 3.
  Motivo: plantando nos perfis reais, o resultado misturava o efeito plantado com as
  diferenças que já existem entre os perfis (com Δ = 0 o "poder" dava 100% em algumas
  regras, e plantar no perfil de menor efeito escondia a diferença). Com a tipologia
  sorteada, Δ = 0 dá cerca de 5% e os dados ficam como são, com toda a variação real
  entre circuitos.
- Efeito plantado: nas voltas da regra dentro do grupo em que o alvo é 0, o alvo vira 1
  com probabilidade Δ / (1 − taxa da regra no grupo). A taxa da regra no grupo sobe Δ
  p.p. em média; o resto da base fica como é.
- Δ (p.p.): duelos 0, 5, 10, 15, 20, 30; parada 0, 1, 2, 3, 5, 8.
- Detectou = p < 0,05 para aquela regra (sem a correção entre regras, que exigiria
  efeitos ainda maiores). 200 repetições por regra, tamanho de grupo e Δ; 1.000
  partições de referência sorteadas uma vez. Poder = fração das repetições que detectam.
- Também é guardado quanto a diferença de efeito (taxa na regra − taxa do grupo) subiu
  de fato, porque a taxa do grupo também sobe um pouco.

Associação, não causa.
"""
from pathlib import Path

import numpy as np
import pandas as pd

P = Path("data/processed")
SAIDA = P / "abrangencia"
URBANO = "urbano-técnico"
PARTICOES, REPETICOES, SEMENTE = 1000, 200, 2026
ANALISES = {
    "duelos": dict(base="base_duelos.csv", regras="regras_globais_duelos.csv", alvo="passou",
                   suporte=100, deltas=[0, 5, 10, 15, 20, 30],
                   fora=["dif_idade=atacante 10+ voltas mais novo E fase=primeiro terço"]),
    "parada": dict(base="base_decisao_parada.csv", regras="regras_globais_parada.csv",
                   alvo="parou", suporte=200, deltas=[0, 1, 2, 3, 5, 8], fora=[]),
}


def mascara(df, descricao):
    m = np.ones(len(df), dtype=bool)
    for cond in descricao.split(" E "):
        c, v = cond.split("=", 1)
        m &= df["d_" + c].to_numpy() == v
    return m


def q_efeito(S, suporte):
    """S: (..., grupos, 4) com n, y, m, ym. Devolve Q/gl do efeito entre grupos."""
    n, y, m, ym = S[..., 0], S[..., 1], S[..., 2], S[..., 3]
    ok = m >= suporte
    with np.errstate(invalid="ignore", divide="ignore"):
        p = y / n
        d = np.where(ok, ym / m - p, 0.0)
        w = np.where(ok, m / np.maximum(p * (1 - p), 1e-9), 0.0)
        dbar = (w * d).sum(-1, keepdims=True) / w.sum(-1, keepdims=True)
        k = ok.sum(-1)
        q = (w * (d - dbar) ** 2).sum(-1) / (k - 1)
    return np.where(k >= 2, q, np.nan)


def roda(nome, cfg, rng):
    df = pd.read_csv(P / cfg["base"])
    regras = pd.read_csv(P / cfg["regras"])
    regras = regras[(regras["sentido"] == "acima") & ~regras["regra"].isin(cfg["fora"])]
    y0 = df[cfg["alvo"]].to_numpy(dtype=float)
    circuitos = np.array(sorted(df["circuito"].unique()))
    cod = pd.Categorical(df["circuito"], categories=circuitos).codes
    perfil_c = df.drop_duplicates("circuito").set_index("circuito")["d_perfil"] \
        .reindex(circuitos).to_numpy()
    tamanhos = pd.Series(perfil_c).value_counts()
    testados = sorted(g for g in tamanhos.index if g != URBANO)
    rot = np.concatenate([[g] * k for g, k in tamanhos.items()])
    parts = np.array([rng.permutation(rot) for _ in range(PARTICOES)])
    H_parts = np.stack([(parts == g) for g in testados], axis=1).astype(float)  # P×3×26
    nc = np.bincount(cod, minlength=len(circuitos)).astype(float)
    print(f"\n{'=' * 78}\n{nome.upper()}: {len(regras)} regras; grupos plantados com o tamanho de "
          f"{', '.join(f'{g} ({tamanhos[g]})' for g in testados)}\n{'=' * 78}")

    linhas = []
    for r in regras.itertuples():
        m = mascara(df, r.regra)
        mc = np.bincount(cod, weights=m, minlength=len(circuitos))
        for g in testados:
            for delta in cfg["deltas"]:
                det, subida, orig = 0, [], []
                for _ in range(REPETICOES):
                    pseudo = rng.permutation(rot)                 # tipologia sorteada
                    no_grupo = (pseudo == g)[cod]
                    alvo_l = m & no_grupo
                    taxa_rg = y0[alvo_l].mean()
                    d0 = taxa_rg - y0[no_grupo].mean()
                    y = y0.copy()
                    prob = min(delta / 100 / max(1 - taxa_rg, 1e-9), 1.0)
                    if prob > 0:
                        cand = np.flatnonzero(alvo_l & (y0 == 0))
                        y[cand[rng.random(len(cand)) < prob]] = 1.0
                    tot = np.column_stack([nc, np.bincount(cod, weights=y, minlength=len(circuitos)),
                                           mc, np.bincount(cod, weights=y * m, minlength=len(circuitos))])
                    H = np.stack([(pseudo == t) for t in testados]).astype(float)
                    q_obs = q_efeito(H @ tot, cfg["suporte"])
                    q_ref = q_efeito(H_parts @ tot, cfg["suporte"])
                    det += (np.sum(q_ref >= q_obs) + 1) / (PARTICOES + 1) < 0.05
                    subida.append(y[alvo_l].mean() - y[no_grupo].mean() - d0)
                    orig.append(d0)
                linhas.append({"regra": r.regra, "ordem": r.ordem, "grupo_plantado": g,
                               "circuitos_no_grupo": int(tamanhos[g]),
                               "efeito_medio_no_grupo_pp": 100 * np.mean(orig), "delta_pp": delta,
                               "subida_efetiva_pp": 100 * np.mean(subida),
                               "poder": det / REPETICOES})
    t = pd.DataFrame(linhas)
    t.to_csv(SAIDA / f"poder_{nome}.csv", index=False)

    pd.set_option("display.width", 250)
    tab = t.pivot_table(index=["ordem", "regra", "circuitos_no_grupo"], columns="delta_pp",
                        values="poder").reset_index()
    tab.columns = [f"Δ{c}" if isinstance(c, (int, np.integer)) else c for c in tab.columns]
    print("Poder (fração das repetições com p < 0,05), por Δ em p.p.:")
    print(tab.to_string(index=False, max_colwidth=60, float_format=lambda x: f"{x:.2f}"))

    menor = []
    for (o, regra, k), s in t.groupby(["ordem", "regra", "circuitos_no_grupo"]):
        ok = s[(s["delta_pp"] > 0) & (s["poder"] >= 0.8)]
        menor.append({"ordem": o, "regra": regra, "circuitos_no_grupo": k,
                      "efeito_medio_no_grupo_pp": s["efeito_medio_no_grupo_pp"].iloc[0],
                      "poder_delta0": s.loc[s["delta_pp"] == 0, "poder"].iloc[0],
                      "menor_delta_80pct": ok["delta_pp"].min() if len(ok) else np.nan,
                      "subida_efetiva_pp": ok["subida_efetiva_pp"].iloc[0] if len(ok) else np.nan})
    menor = pd.DataFrame(menor)
    menor["como_pct_do_efeito"] = 100 * menor["subida_efetiva_pp"] / menor["efeito_medio_no_grupo_pp"]
    print("\nMenor Δ com poder de 80% (NaN = nenhum Δ testado chegou a 80%):")
    print(menor.to_string(index=False, max_colwidth=60, float_format=lambda x: f"{x:.2f}"))
    menor.to_csv(SAIDA / f"poder_{nome}_resumo.csv", index=False)


def main():
    rng = np.random.default_rng(SEMENTE)
    SAIDA.mkdir(parents=True, exist_ok=True)
    for nome, cfg in ANALISES.items():
        roda(nome, cfg, rng)


if __name__ == "__main__":
    main()
