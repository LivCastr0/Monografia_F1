"""
Etapa 2 - Piloto da afinidade carro × perfil: a medida de ritmo é confiável?

Entra: data/interim/voltas/*_voltas.csv
       data/processed/base_piloto_corrida.csv
Não salva nada e não usa internet.

Pergunta da análise: que carros andam acima do próprio padrão em cada perfil de
pista? É a hipótese (ii) do relatório (afinidade de equipe com perfil de pista), com
o ritmo de volta no lugar da posição final.

Este piloto só verifica se a medida captura o carro. De propósito, NÃO mostra o
ritmo de nenhuma equipe por perfil ou por circuito. Usa as corridas secas entre as
10 sorteadas no piloto do A1.

A medida: ritmo do carro na corrida
-----------------------------------
1. Voltas limpas: precisas segundo a FastF1, bandeira verde na volta, fora de
   entrada e saída do box, não é a volta 1, a pelo menos TRAFEGO_S do carro da frente
   e até 107% da mediana da corrida.
2. Em cada corrida, um ajuste de mínimos quadrados com todas as voltas limpas do
   pelotão: tempo ~ composto + idade do pneu (inclinação por composto) + número da
   volta (combustível). O resto do ajuste é o ritmo da volta, descontados pneu e
   combustível.
3. Ritmo do piloto = mediana dos restos das voltas dele, em % do tempo mediano da
   corrida (negativo = mais rápido). Ritmo do carro = média dos dois pilotos.
Na análise final, o alvo será o ritmo do carro menos a média dele na temporada.

Critérios, fixados antes de rodar (mudar só antes de ver a saída):
  1. Cobertura: pelo menos 8 carros com ritmo (os dois pilotos com 10+ voltas limpas)
     por corrida, em média.
  2. A medida captura o carro: correlação de Spearman entre os ritmos dos dois
     pilotos do mesmo carro de pelo menos 0,70.
  3. Pouco ruído: o carro explica pelo menos 60% da variação entre pilotos na mesma
     corrida (correlação intraclasse entre companheiros).
  4. Sensibilidade: com 5 corridas de um carro num perfil, a análise detecta
     afinidades de até 0,30% do tempo de volta (2 × ruído do carro ÷ raiz de 5).
     O ruído do carro vem da diferença entre os companheiros.
  5. Tráfego: a ordem dos carros muda pouco sem o filtro de tráfego (Spearman de pelo
     menos 0,90 entre as duas versões).
"""
from pathlib import Path

import numpy as np
import pandas as pd

VOLTAS = Path("data/interim/voltas")
BASE = Path("data/processed/base_piloto_corrida.csv")
PILOTO = [(2021, 13), (2022, 15), (2022, 2), (2019, 2), (2022, 8),
          (2023, 22), (2020, 7), (2023, 12), (2021, 5), (2025, 8)]
TRAFEGO_S = 1.5
LIMITE_LENTA = 1.07
MIN_VOLTAS = 10
SECOS = ["HYPERSOFT", "ULTRASOFT", "SUPERSOFT", "SOFT", "MEDIUM", "HARD"]
MIN_CARROS, MIN_SPEARMAN, MIN_ICC, MAX_DETECTAVEL, MIN_TRAFEGO = 8, 0.70, 0.60, 0.30, 0.90


def verdadeiro(s):
    return s.astype(str).str.lower() == "true"


def voltas_limpas(v, trafego):
    v = v.dropna(subset=["volta", "posicao", "tempo_volta_s", "vida_pneu"]).copy()
    v["volta"] = v["volta"].astype(int)
    v = v.sort_values(["volta", "posicao"])
    v["dist_frente"] = v.groupby("volta")["fim_volta_s"].diff()
    v.loc[v["posicao"] == v.groupby("volta")["posicao"].transform("min"), "dist_frente"] = np.inf
    verde = v.groupby("volta")["bandeira"].agg(lambda s: (s.astype(str) == "1").mean() >= 0.8)
    v = v[v["volta"].map(verde) & (v["volta"] > 1) & v["composto"].isin(SECOS)
          & verdadeiro(v["volta_precisa"]) & ~verdadeiro(v["gerada_pela_fastf1"])
          & v["entrada_box_s"].isna() & v["saida_box_s"].isna()
          & (v["dist_frente"] >= trafego)]
    return v[v["tempo_volta_s"] <= LIMITE_LENTA * v["tempo_volta_s"].median()]


def ritmo_por_piloto(v, trafego=TRAFEGO_S):
    """Ritmo de cada piloto, em % do tempo mediano, descontados pneu e combustível."""
    l = voltas_limpas(v, trafego)
    if len(l) < 50:
        return pd.DataFrame()
    comp = pd.get_dummies(l["composto"]).astype(float)
    X = np.column_stack([comp.to_numpy(),
                         comp.to_numpy() * l["vida_pneu"].to_numpy()[:, None],
                         l["volta"].to_numpy(float)])
    y = l["tempo_volta_s"].to_numpy(float)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    l = l.assign(resto=(y - X @ coef) / np.median(y) * 100)
    g = l.groupby(["equipe", "piloto"])["resto"].agg(["median", "size"]).reset_index()
    g = g[g["size"] >= MIN_VOLTAS]
    return g.rename(columns={"median": "ritmo", "size": "voltas"})


def pares(r):
    """Uma linha por carro com os dois pilotos."""
    linhas = []
    for equipe, g in r.groupby("equipe"):
        if len(g) == 2:
            a, b = g.sort_values("piloto")["ritmo"].to_numpy()
            linhas.append({"equipe": equipe, "a": a, "b": b, "carro": (a + b) / 2})
    return pd.DataFrame(linhas)


def main():
    base = pd.read_csv(BASE).drop_duplicates(["ano", "round"]).set_index(["ano", "round"])
    todos, trafego, carros_por_corrida = [], [], []
    for ano, rnd in PILOTO:
        if base.at[(ano, rnd), "faixa_chuva"] != "seco":
            print(f"{ano} r{rnd}: corrida com chuva, fora")
            continue
        arquivo = VOLTAS / f"{ano}_{rnd:02d}_voltas.csv"
        v = pd.read_csv(arquivo, dtype={"bandeira": str})
        p = pares(ritmo_por_piloto(v))
        p0 = pares(ritmo_por_piloto(v, trafego=0.0))
        carros_por_corrida.append(len(p))
        print(f"{ano} r{rnd}: {len(p)} carros com ritmo")
        if len(p):
            todos.append(p.assign(corrida=f"{ano}_{rnd:02d}"))
            m = p.merge(p0, on="equipe", suffixes=("", "_sem"))
            if len(m) >= 4:
                trafego.append(m["carro"].corr(m["carro_sem"], method="spearman"))

    d = pd.concat(todos, ignore_index=True)
    # Dentro de cada corrida, ritmo relativo à média dos carros
    for c in ["a", "b", "carro"]:
        d[c] = d[c] - d.groupby("corrida")[c].transform("mean")
    spearman = d["a"].corr(d["b"], method="spearman")
    entre = d["carro"].var(ddof=1)
    dentro = ((d["a"] - d["b"]) ** 2 / 2).mean()        # variância por piloto
    icc = (entre - dentro / 2) / (entre + dentro / 2)
    ruido_carro = np.sqrt(dentro / 2)                     # erro do ritmo do carro (média de 2)
    detectavel = 2 * ruido_carro / np.sqrt(5)

    print(f"\nCarros-corrida: {len(d)}")
    print(f"Diferença típica entre companheiros: {np.sqrt(2 * dentro):.3f}% do tempo de volta")
    print(f"Variação entre carros: desvio {np.sqrt(entre):.3f}%")

    v = lambda ok: "PASSOU" if ok else "NÃO PASSOU"
    media_carros = np.mean(carros_por_corrida)
    traf = np.median(trafego) if trafego else np.nan
    c = [media_carros >= MIN_CARROS, spearman >= MIN_SPEARMAN, icc >= MIN_ICC,
         detectavel <= MAX_DETECTAVEL, traf >= MIN_TRAFEGO]
    print("\n" + "=" * 72 + "\nCRITÉRIOS\n" + "=" * 72)
    print(f"1. Carros com ritmo por corrida: {media_carros:.1f} (mínimo {MIN_CARROS})  {v(c[0])}")
    print(f"2. Companheiros concordam:       {spearman:.2f} (mínimo {MIN_SPEARMAN:.2f})  {v(c[1])}")
    print(f"3. O carro explica:              {icc:.1%} (mínimo {MIN_ICC:.0%})  {v(c[2])}")
    print(f"4. Afinidade detectável (n=5):   {detectavel:.3f}% (máximo {MAX_DETECTAVEL:.2f}%)  {v(c[3])}")
    print(f"5. Sem filtro de tráfego:        {traf:.2f} (mínimo {MIN_TRAFEGO:.2f})  {v(c[4])}")
    print()
    print("Decisão: a medida serve. Seguir para a base inteira." if all(c) else
          "Decisão: entender a falha antes de seguir.")
    print("\nNada foi salvo.")


if __name__ == "__main__":
    main()
