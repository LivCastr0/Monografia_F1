"""
Etapa 2 - Avalia a medida de degradação nas 10 corridas do piloto.

Entra: data/processed/base_stint.csv
Não salva nada e não usa internet.

A pergunta é só se a medida é confiável. De propósito, este script NÃO mostra a
degradação por perfil, abrasão, composto ou qualquer descritor.

Só entram as corridas secas das 10 sorteadas no piloto do A1.

Critérios, fixados antes de rodar (mudar só antes de ver a saída):
  1. Quantidade: pelo menos 10 stints válidos por corrida, em média. Abaixo disso, a
     medida usa uma parte pequena da corrida.
  2. Sentido físico: pelo menos 65% dos stints com degradação positiva. Pneu que
     "melhora" com a idade indica ruído, poupança de pneu ou correção errada.
  3. Sinal: a corrida explica pelo menos 25% da variação entre stints do mesmo
     composto. Se dois carros na mesma pista e pneu dão números sem relação, a medida
     é ruído. O limite está acima do que o acaso daria, e o acaso é mostrado na saída.
  4. Combustível: a ordem das corridas muda pouco com a suposição de 0,02 ou
     0,04 s/kg (correlação de Spearman de pelo menos 0,80).

Passando nos quatro, vale fazer a busca de subgrupos com a base inteira. Falhando,
a medida não serve e a hipótese fica registrada assim.
"""
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("data/processed/base_stint.csv")
PILOTO = [(2021, 13), (2022, 15), (2022, 2), (2019, 2), (2022, 8),
          (2023, 22), (2020, 7), (2023, 12), (2021, 5), (2025, 8)]
MEDIDA = "deg_pct"
MIN_STINTS, MIN_POSITIVOS, MIN_SINAL, MIN_COMBUSTIVEL = 10, 0.65, 0.25, 0.80
PERMUTACOES, SEMENTE = 2000, 10


def parcela_corrida(d, corrida):
    """Parcela da variação, dentro de cada composto, explicada pela corrida."""
    y = d[MEDIDA] - d.groupby("d_composto")[MEDIDA].transform("mean")
    m = y.groupby([corrida, d["d_composto"].to_numpy()]).transform("mean")
    return ((m - y.mean()) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def main():
    df = pd.read_csv(BASE)
    d = df[[(a, r) in PILOTO for a, r in zip(df["ano"], df["round"])]].reset_index(drop=True)
    if d.empty:
        raise ValueError("nenhum stint das corridas do piloto: conferir base_stint.csv")
    d["corrida"] = d["ano"].astype(str) + "_" + d["round"].astype(str)
    n_corridas = d["corrida"].nunique()
    por_corrida = d.groupby("corrida").size()

    print(f"Corridas secas do piloto com stint válido: {n_corridas} de {len(PILOTO)}")
    print(f"Stints: {len(d)}, voltas limpas por stint: {d['voltas_limpas'].median():.0f}\n")
    print(por_corrida.to_string())

    media_stints = len(d) / n_corridas
    positivos = (d[MEDIDA] > 0).mean()

    rng = np.random.default_rng(SEMENTE)
    corrida = d["corrida"].to_numpy()
    sinal = parcela_corrida(d, corrida)
    acaso = np.mean([parcela_corrida(d, rng.permutation(corrida)) for _ in range(PERMUTACOES)])

    medias = d.groupby("corrida")[["deg_pct_baixo", "deg_pct_alto"]].mean()
    combustivel = medias["deg_pct_baixo"].corr(medias["deg_pct_alto"], method="spearman")

    print(f"\nDesvio entre stints: {d[MEDIDA].std():.3f}% por volta")
    print(f"Efeito detectável com 200 stints: {2 * d[MEDIDA].std() / np.sqrt(200):.3f}%\n")

    c1, c2 = media_stints >= MIN_STINTS, positivos >= MIN_POSITIVOS
    c3, c4 = sinal >= MIN_SINAL, combustivel >= MIN_COMBUSTIVEL
    v = lambda ok: "PASSOU" if ok else "NÃO PASSOU"
    print("=" * 72 + "\nCRITÉRIOS\n" + "=" * 72)
    print(f"1. Stints por corrida:     {media_stints:.1f} (mínimo {MIN_STINTS})  {v(c1)}")
    print(f"2. Degradação positiva:    {positivos:.1%} (mínimo {MIN_POSITIVOS:.0%})  {v(c2)}")
    print(f"3. Sinal entre corridas:   {sinal:.1%} (mínimo {MIN_SINAL:.0%}; acaso {acaso:.1%})  {v(c3)}")
    print(f"4. Efeito do combustível:  {combustivel:.2f} (mínimo {MIN_COMBUSTIVEL:.2f})  {v(c4)}")
    print()
    if c1 and c2 and c3 and c4:
        print("Decisão: a medida serve. Seguir para a busca de subgrupos.")
    else:
        print("Decisão: a medida não passou. Não seguir para a busca sem entender a falha.")
    print("\nNada foi salvo.")


if __name__ == "__main__":
    main()
