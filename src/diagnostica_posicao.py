"""
Etapa 2 - Diagnóstico do critério 1 do piloto do A1 (posição x resultado oficial).

Entra: data/interim/voltas/*_voltas.csv
       data/interim/corridas/*_pilotos.csv
       data/processed/contexto_corridas.csv
Sai:   data/processed/diag_posicao_final.csv   (os carros fora da ordem comum)

Não usa internet.

Na base inteira o critério 1 deu 86,6% e não passou. Esse resultado continua
registrado como está. O critério compara número com número: se um carro for
desclassificado ou punido depois da bandeirada, todos os que estavam atrás dele
mudam de número sem que nenhuma volta esteja errada. Em Spa 2024 uma única
desclassificação derruba a corrida inteira para 0 de 18.

Este script faz a pergunta que importa: a ordem na pista e a ordem oficial são
a mesma, tirando poucos carros? Para cada corrida, ele acha a maior sequência de
pilotos que aparece na mesma ordem nas duas listas. Quem fica de fora é um carro
"movido". Com Zandvoort 2022, seis números diferentes viram dois carros movidos.

Regra fixada antes de rodar nos dados reais:
  A. Pelo menos 90% dos carros ficam na ordem comum (o mínimo do critério 1).
  B. Pelo menos 90% dos carros movidos, entre os que têm a distância medida, estão
     a até 35 segundos dos carros com que trocaram de ordem. A distância vem do
     tempo no fim da última volta. Punições costumam ser de 5 a 30 segundos, então
     uma distância maior não se explica assim e pede conferência manual.

Passando em A e B, a posição volta a volta é confiável e a falha do critério 1
vem do resultado oficial, não dos dados. A causa de cada carro não é confirmada
aqui: isso exigiria o registro de punições, que não foi coletado.
"""
from pathlib import Path

import pandas as pd

VOLTAS = Path("data/interim/voltas")
CORRIDAS = Path("data/interim/corridas")
CONTEXTO = Path("data/processed/contexto_corridas.csv")
SAIDA = Path("data/processed/diag_posicao_final.csv")

MIN_ORDEM_COMUM = 0.90
MAX_DISTANCIA_S = 35
MIN_MOVIDOS_PERTO = 0.90
PADRAO_TERMINOU = r"^(Finished|Lapped|\+\d+ Laps?)$"


def ordem_comum(sequencia):
    """Índices da maior subsequência crescente de `sequencia`."""
    n = len(sequencia)
    if n == 0:
        return []
    tamanho, anterior = [1] * n, [-1] * n
    for i in range(n):
        for j in range(i):
            if sequencia[j] < sequencia[i] and tamanho[j] + 1 > tamanho[i]:
                tamanho[i], anterior[i] = tamanho[j] + 1, j
    i = max(range(n), key=lambda k: tamanho[k])
    indices = []
    while i != -1:
        indices.append(i)
        i = anterior[i]
    return indices[::-1]


def carros_movidos(d):
    """Recebe a corrida ordenada pela pista. Devolve uma linha por carro movido."""
    d = d.reset_index(drop=True)
    comum = set(ordem_comum(d["posicao_final"].tolist()))
    linhas = []
    for i in d.index.difference(list(comum)):
        trocas = d[((d.index > i) & (d["posicao_final"] < d.at[i, "posicao_final"]))
                   | ((d.index < i) & (d["posicao_final"] > d.at[i, "posicao_final"]))]
        mesma_volta = trocas[trocas["volta"] == d.at[i, "volta"]]
        distancia = (mesma_volta["fim_volta_s"] - d.at[i, "fim_volta_s"]).abs().max()
        linhas.append({
            "piloto": d.at[i, "piloto"],
            "ordem_na_pista": i + 1,
            "posicao_oficial": d.at[i, "posicao_final"],
            "carros_trocados": len(trocas),
            "distancia_s": distancia,
        })
    return len(d), linhas


def avalia(arquivo):
    ano, rnd = (int(t) for t in arquivo.name.split("_")[:2])
    v = pd.read_csv(arquivo, dtype={"bandeira": str}).dropna(subset=["volta", "posicao"])
    ultima = v.sort_values("volta").groupby("piloto").tail(1)
    p = pd.read_csv(CORRIDAS / f"{ano}_{rnd:02d}_pilotos.csv")
    terminou = p[p["status"].astype(str).str.match(PADRAO_TERMINOU)]
    d = ultima.merge(terminou[["piloto", "posicao_final"]], on="piloto")
    d = d.sort_values(["volta", "posicao"], ascending=[False, True])
    total, linhas = carros_movidos(d)
    return total, [{"ano": ano, "round": rnd, **x} for x in linhas]


def main():
    total, movidos = 0, []
    arquivos = sorted(VOLTAS.glob("*_voltas.csv"))
    for a in arquivos:
        n, linhas = avalia(a)
        total += n
        movidos += linhas

    m = pd.DataFrame(movidos, columns=["ano", "round", "piloto", "ordem_na_pista",
                                       "posicao_oficial", "carros_trocados", "distancia_s"])
    contexto = pd.read_csv(CONTEXTO)[["ano", "round", "circuito", "teve_bandeira_vermelha"]]
    m = m.merge(contexto, on=["ano", "round"], how="left")
    m.to_csv(SAIDA, index=False)

    pct_comum = 1 - len(m) / total
    medidos = m["distancia_s"].dropna()
    perto = (medidos <= MAX_DISTANCIA_S).mean() if len(medidos) else 1.0

    print(f"Corridas: {len(arquivos)}   carros que terminaram: {total}")
    print(f"Carros movidos: {len(m)}, em {m.groupby(['ano', 'round']).ngroups} corridas")
    print(f"Movidos sem distância medida (voltas diferentes): {m['distancia_s'].isna().sum()}")

    longe = m[m["distancia_s"] > MAX_DISTANCIA_S]
    print(f"\nMovidos a mais de {MAX_DISTANCIA_S} s: {len(longe)}")
    if len(longe):
        print(longe[["ano", "round", "circuito", "piloto", "ordem_na_pista", "posicao_oficial",
                     "distancia_s", "teve_bandeira_vermelha"]]
              .sort_values("distancia_s", ascending=False).head(25)
              .to_string(index=False, float_format=lambda x: f"{x:.1f}"))

    por_corrida = m.groupby(["ano", "round", "circuito"]).size().sort_values(ascending=False)
    print("\nCorridas com mais carros movidos:")
    print(por_corrida.head(10).to_string())

    a = pct_comum >= MIN_ORDEM_COMUM
    b = perto >= MIN_MOVIDOS_PERTO
    print("\n" + "=" * 72)
    print("REGRA DO DIAGNÓSTICO")
    print("=" * 72)
    print(f"A. Carros na ordem comum:   {pct_comum:.1%} (mínimo {MIN_ORDEM_COMUM:.0%})  "
          f"{'PASSOU' if a else 'NÃO PASSOU'}")
    print(f"B. Movidos até {MAX_DISTANCIA_S} s:        {perto:.1%} "
          f"(mínimo {MIN_MOVIDOS_PERTO:.0%})  {'PASSOU' if b else 'NÃO PASSOU'}")
    print()
    if a and b:
        print("Conclusão: a posição volta a volta é confiável; a falha do critério 1 "
              "vem do resultado oficial.")
    else:
        print("Conclusão: a falha não se explica só pelo resultado oficial. Conferir "
              "os casos listados antes de seguir.")
    print(f"\nSalvo em {SAIDA}")


if __name__ == "__main__":
    main()
