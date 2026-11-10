"""
Etapa 2 - Piloto da análise de duelos: a medida de ultrapassagem é confiável?

Entra: data/interim/voltas/*_voltas.csv
Não salva nada e não usa internet.

Pergunta da análise: em que condições o carro de trás consegue passar?
Este piloto só verifica se dá para identificar duelos e ultrapassagens a partir das
voltas. De propósito, NÃO mostra a taxa de ultrapassagem por perfil, era, pneu ou
qualquer outro descritor.

Usa as mesmas 10 corridas sorteadas no piloto do A1.

Definições
----------
Volta de duelo: ao fim da volta anterior, o carro B está logo atrás do carro A (uma
posição) a até DISTANCIA_S segundos, e:
  - a volta anterior e a atual são de bandeira verde para a pista inteira;
  - nenhum dos dois entra ou sai do box na volta anterior ou na atual;
  - não é a volta 1;
  - os dois completam a volta atual.
Ultrapassagem: ao fim da volta de duelo, B está à frente de A.
Duelo: voltas de duelo seguidas do mesmo par, na mesma ordem. O duelo termina na
ultrapassagem ou quando a condição deixa de valer. Alvo futuro: o duelo terminou em
ultrapassagem.
Possível erro: na volta da ultrapassagem, o tempo de A passa de LIMITE_ERRO vezes a
mediana das voltas dele. Indica que A pode ter errado ou ter tido problema, e não
que B passou na disputa.

Critérios, fixados antes de rodar (mudar só antes de ver a saída):
  1. Pelo menos 30 duelos por corrida, em média.
  2. Taxa de ultrapassagem por duelo entre 5% e 50%. Fora disso, o alvo fica raro ou
     comum demais para a busca de subgrupos.
  3. No máximo 25% das ultrapassagens com possível erro.
  4. Pelo menos 90% das ultrapassagens com composto e idade do pneu dos dois carros.
  5. Pelo menos 95% das ultrapassagens com a ordem dos tempos igual à das posições
     no fim da volta (posição e cronometragem coerentes).
"""
from pathlib import Path

import numpy as np
import pandas as pd

VOLTAS = Path("data/interim/voltas")
PILOTO = [(2021, 13), (2022, 15), (2022, 2), (2019, 2), (2022, 8),
          (2023, 22), (2020, 7), (2023, 12), (2021, 5), (2025, 8)]
DISTANCIA_S = 1.0
LIMITE_ERRO = 1.07
MIN_DUELOS, TAXA_MIN, TAXA_MAX = 30, 0.05, 0.50
MAX_ERRO, MIN_PNEU, MIN_COERENTE = 0.25, 0.90, 0.95


def voltas_verdes(v):
    return v.groupby("volta")["bandeira"].agg(lambda s: (s.astype(str) == "1").mean() >= 0.8)


def duelos_da_corrida(v):
    v = v.dropna(subset=["volta", "posicao"]).copy()
    v["volta"] = v["volta"].astype(int)
    v["box"] = v["entrada_box_s"].notna() | v["saida_box_s"].notna()
    verde = voltas_verdes(v)
    mediana = v.groupby("piloto")["tempo_volta_s"].median()
    por_volta = {k: g.set_index("piloto") for k, g in v.groupby("volta")}

    linhas = []
    for volta in sorted(por_volta):
        antes = volta - 1
        if antes < 1 or antes not in por_volta or not (verde.get(volta, False) and verde.get(antes, False)):
            continue
        a_ant, a_atu = por_volta[antes].sort_values("posicao"), por_volta[volta]
        ordem = a_ant.index.to_list()
        for frente, tras in zip(ordem[:-1], ordem[1:]):
            if a_ant.at[tras, "posicao"] - a_ant.at[frente, "posicao"] != 1:
                continue
            if frente not in a_atu.index or tras not in a_atu.index:
                continue
            distancia = a_ant.at[tras, "fim_volta_s"] - a_ant.at[frente, "fim_volta_s"]
            if not (0 < distancia <= DISTANCIA_S):
                continue
            if a_ant.at[frente, "box"] or a_ant.at[tras, "box"] or a_atu.at[frente, "box"] or a_atu.at[tras, "box"]:
                continue
            passou = a_atu.at[tras, "posicao"] < a_atu.at[frente, "posicao"]
            linhas.append({
                "volta": volta, "frente": frente, "tras": tras, "distancia": distancia,
                "passou": passou,
                "coerente": (a_atu.at[tras, "fim_volta_s"] < a_atu.at[frente, "fim_volta_s"]) == passou,
                "erro": passou and a_atu.at[frente, "tempo_volta_s"] > LIMITE_ERRO * mediana.get(frente, np.inf),
                "pneu_ok": all(pd.notna(a_atu.at[p, c]) for p in (frente, tras)
                               for c in ("composto", "vida_pneu")),
            })
    d = pd.DataFrame(linhas, columns=["volta", "frente", "tras", "distancia", "passou",
                                      "coerente", "erro", "pneu_ok"])
    if d.empty:
        return d
    # Voltas seguidas do mesmo par viram um duelo
    d = d.sort_values(["frente", "tras", "volta"])
    novo = (d["frente"] != d["frente"].shift()) | (d["tras"] != d["tras"].shift()) \
        | (d["volta"] != d["volta"].shift() + 1) | d["passou"].shift(fill_value=False)
    d["duelo"] = novo.cumsum()
    return d


def main():
    por_corrida, duelos, voltas_passe = [], [], []
    for ano, rnd in PILOTO:
        arquivo = VOLTAS / f"{ano}_{rnd:02d}_voltas.csv"
        if not arquivo.exists():
            print(f"Falta {arquivo}")
            continue
        d = duelos_da_corrida(pd.read_csv(arquivo, dtype={"bandeira": str}))
        if d.empty:
            por_corrida.append({"corrida": f"{ano}_{rnd:02d}", "voltas_de_duelo": 0, "duelos": 0,
                                "ultrapassagens": 0})
            continue
        e = d.groupby("duelo").agg(voltas=("volta", "size"), passou=("passou", "any"))
        por_corrida.append({"corrida": f"{ano}_{rnd:02d}", "voltas_de_duelo": len(d),
                            "duelos": len(e), "ultrapassagens": int(e["passou"].sum())})
        duelos.append(e)
        voltas_passe.append(d[d["passou"]])

    t = pd.DataFrame(por_corrida)
    print(t.to_string(index=False))
    e = pd.concat(duelos)
    p = pd.concat(voltas_passe)

    media_duelos = t["duelos"].mean()
    taxa = e["passou"].mean()
    erro = p["erro"].mean() if len(p) else 0.0
    pneu = p["pneu_ok"].mean() if len(p) else 0.0
    coerente = p["coerente"].mean() if len(p) else 0.0

    print(f"\nDuelos: {len(e)}, ultrapassagens: {int(e['passou'].sum())}")
    print(f"Voltas por duelo: mediana {e['voltas'].median():.0f}, máximo {e['voltas'].max()}")

    v = lambda ok: "PASSOU" if ok else "NÃO PASSOU"
    c = [media_duelos >= MIN_DUELOS, TAXA_MIN <= taxa <= TAXA_MAX,
         erro <= MAX_ERRO, pneu >= MIN_PNEU, coerente >= MIN_COERENTE]
    print("\n" + "=" * 72 + "\nCRITÉRIOS\n" + "=" * 72)
    print(f"1. Duelos por corrida:          {media_duelos:.1f} (mínimo {MIN_DUELOS})  {v(c[0])}")
    print(f"2. Taxa de ultrapassagem:       {taxa:.1%} (entre {TAXA_MIN:.0%} e {TAXA_MAX:.0%})  {v(c[1])}")
    print(f"3. Ultrapassagens c/ possível erro: {erro:.1%} (máximo {MAX_ERRO:.0%})  {v(c[2])}")
    print(f"4. Ultrapassagens c/ dado de pneu:  {pneu:.1%} (mínimo {MIN_PNEU:.0%})  {v(c[3])}")
    print(f"5. Posição e tempo coerentes:   {coerente:.1%} (mínimo {MIN_COERENTE:.0%})  {v(c[4])}")
    print()
    if all(c):
        print("Decisão: a medida serve. Seguir para a montagem da base de duelos.")
    else:
        print("Decisão: entender a falha antes de montar a base de duelos.")
    print("\nNada foi salvo.")


if __name__ == "__main__":
    main()
