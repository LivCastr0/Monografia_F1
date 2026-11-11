"""
Etapa 2 - H9: quem ganha e quem perde posição com o safety car.

Critérios fixados antes de montar a base: claude/safety-car-chuva-msi2.md (projeto no
claude.ai). Exploratória declarada.

Grão: piloto x período de safety car (código 4), em corridas secas.
  Referência: posição no fim da volta anterior à do início do SC (volta do líder).
  Resultado:  posição no fim da 3.ª volta depois da relargada.
  Alvo `ganho`: posições ganhas, reordenadas entre os pilotos presentes nos dois momentos.
Fora: período que vira vermelha; outra neutralização ou vermelha antes da volta de
resultado; corrida que termina antes dela; SC iniciado na volta 1 (não há volta anterior
para a referência; definido ao montar a base, antes de ver o alvo).

Entra: data/interim/voltas, data/interim/corridas, data/processed/base_piloto_corrida.csv
Sai:   data/processed/base_safety_car.csv, data/processed/sd_safety_car/
Uso:   python src/sd_safety_car.py [--sintetico]
Associação, não causa.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import avalia_piloto_a1 as A  # noqa: E402
import sd_eventos as S  # noqa: E402

P = Path("data/processed")
BASE = P / "base_safety_car.csv"
SAIDA = P / "sd_safety_car"
SEMENTE = 2026
VOLTAS_DEPOIS = 3
MIN_CONFERIDOS = 0.80
DESCRITORES = ["d_posicao", "d_parou", "d_frente_parou", "d_idade_pneu", "d_paradas_feitas",
               "d_dist_frente", "d_dist_tras", "d_fase"]


def faixa(x, cortes, rotulos):
    for c, r in zip(cortes, rotulos):
        if x <= c:
            return r
    return rotulos[-1]


def periodos(b):
    """Lista de (inicio, fim, codigo_que_encerra) dos períodos de SC."""
    t, c = b["tempo_s"].to_numpy(), b["codigo"].astype(str).to_numpy()
    out, i = [], 0
    while i < len(c):
        if c[i] == "4" and (i == 0 or c[i - 1] != "4"):
            j = i + 1
            while j < len(c) and c[j] == "4":
                j += 1
            out.append((t[i], t[j] if j < len(c) else np.nan, c[j] if j < len(c) else None))
            i = j
        else:
            i += 1
    return out


def corrida(ano, rnd, info):
    v = A.le_voltas(A.VOLTAS / f"{ano}_{rnd:02d}_voltas.csv").dropna(subset=["posicao"]).copy()
    par = A.acha_paradas(v)
    trocas = par[par["trocou_pneu"]]
    s = pd.read_csv(A.CORRIDAS / f"{ano}_{rnd:02d}_stints.csv")
    esperadas = s.groupby("piloto").size() - 1
    medidas = trocas.groupby("piloto").size().reindex(esperadas.index, fill_value=0)
    if (medidas == esperadas).mean() < MIN_CONFERIDOS:
        return None, "paradas"
    b = pd.read_csv(A.CORRIDAS / f"{ano}_{rnd:02d}_bandeiras.csv").sort_values("tempo_s")
    ent = v.set_index(["piloto", "volta"])["entrada_box_s"]
    trocas = trocas.assign(t_box=[ent.get((p, l), np.nan) for p, l in
                                  zip(trocas["piloto"], trocas["volta_parada"])])
    lider = v[v["posicao"] == 1].set_index("volta")["fim_volta_s"].sort_index()
    ultima = int(v["volta"].max())
    pos = v.pivot_table(index="volta", columns="piloto", values="posicao")
    fim = v.pivot_table(index="volta", columns="piloto", values="fim_volta_s")
    vida = v.pivot_table(index="volta", columns="piloto", values="vida_pneu")
    linhas, fora = [], []

    def volta_em(t):
        k = np.flatnonzero(lider.to_numpy() >= t)
        return int(lider.index[k[0]]) if len(k) else None

    for n_p, (t0, t1, cod_fim) in enumerate(periodos(b), 1):
        if cod_fim is None or cod_fim == "5":
            fora.append("vermelha ou sem fim")
            continue
        l0, l1 = volta_em(t0), volta_em(t1)
        if l0 is None or l1 is None:
            fora.append("sem volta")
            continue
        if l0 <= 1:
            fora.append("volta 1")
            continue
        lref, lres = l0 - 1, l1 + VOLTAS_DEPOIS
        if lres > ultima:
            fora.append("fim da corrida")
            continue
        t_res = lider.get(lres, np.nan)
        depois = b[(b["tempo_s"] > t1) & (b["tempo_s"] <= t_res)]["codigo"].astype(str)
        if depois.isin(["4", "5", "6", "7"]).any():
            fora.append("outra neutralização")
            continue
        p_ref, p_res = pos.loc[lref].dropna(), pos.loc[lres].dropna()
        presentes = p_ref.index.intersection(p_res.index)
        if len(presentes) < 5:
            fora.append("poucos pilotos")
            continue
        r_ref, r_res = p_ref[presentes].rank(), p_res[presentes].rank()
        sob = trocas[(trocas["t_box"] >= t0) & (trocas["t_box"] <= t1)]["piloto"]
        parou = set(sob)
        ordem = r_ref.sort_values()
        lista = list(ordem.index)
        f_ref = fim.loc[lref]
        for pil in presentes:
            k = lista.index(pil)
            frente3 = lista[max(0, k - 3):k]
            d_f = f_ref[pil] - f_ref[lista[k - 1]] if k > 0 else np.nan
            d_t = f_ref[lista[k + 1]] - f_ref[pil] if k + 1 < len(lista) else np.nan
            feitas = int(((trocas["piloto"] == pil) & (trocas["volta_parada"] <= lref)).sum())
            idade = vida.loc[lref].get(pil, np.nan)
            linhas.append({
                "ano": ano, "round": rnd, "circuito": info["circuito"],
                "evento": f"{ano}_{rnd:02d}_sc{n_p}", "piloto": pil,
                "volta_inicio": l0, "volta_relargada": l1,
                "pos_ref": int(r_ref[pil]), "pos_res": int(r_res[pil]),
                "ganho": float(r_ref[pil] - r_res[pil]),
                "d_posicao": "líder" if r_ref[pil] == 1 else
                faixa(r_ref[pil], [5, 10], ["2.º a 5.º", "6.º a 10.º", "11.º em diante"]),
                "d_parou": "parou no SC" if pil in parou else "não parou no SC",
                "d_frente_parou": "algum da frente parou" if parou & set(frente3)
                else "nenhum da frente parou",
                "d_idade_pneu": faixa(idade, [9, 19], ["0–9 voltas", "10–19 voltas", "20+ voltas"])
                if pd.notna(idade) else np.nan,
                "d_paradas_feitas": "0 paradas" if feitas == 0 else "1+ paradas",
                "d_dist_frente": faixa(d_f, [1, 3], ["até 1 s", "1 a 3 s", "mais de 3 s"])
                if pd.notna(d_f) else "mais de 3 s",
                "d_dist_tras": faixa(d_t, [1, 3], ["até 1 s", "1 a 3 s", "mais de 3 s"])
                if pd.notna(d_t) else "mais de 3 s",
                "d_fase": "primeiro terço" if l0 <= info["voltas"] / 3 else
                "segundo terço" if l0 <= 2 * info["voltas"] / 3 else "último terço",
                "d_era": "2022+" if ano >= 2022 else "2018-21"})
    return pd.DataFrame(linhas), fora


def monta():
    base = pd.read_csv(P / "base_piloto_corrida.csv").drop_duplicates(["ano", "round"])
    base = base[(base["faixa_chuva"] == "seco") & ~base["corrida_sem_disputa"]]
    partes, motivos, sem_paradas = [], [], 0
    for _, r in base.iterrows():
        info = {"circuito": r["circuito"], "voltas": r["voltas_corrida"]}
        d, fora = corrida(int(r["ano"]), int(r["round"]), info)
        if d is None:
            sem_paradas += 1
            continue
        motivos += fora
        if len(d):
            partes.append(d)
    df = pd.concat(partes, ignore_index=True)
    df.to_csv(BASE, index=False)
    print(f"Corridas secas: {len(base)}; fora por dado de parada incompleto: {sem_paradas}")
    print(f"Períodos usados: {df['evento'].nunique()} em {df.groupby(['ano', 'round']).ngroups} "
          f"corridas e {df['circuito'].nunique()} circuitos; linhas {len(df)}")
    print("Períodos fora:", pd.Series(motivos).value_counts().to_dict())
    for c in DESCRITORES:
        print(f"  {c}: {df[c].value_counts().to_dict()}")
    return df


def sintetico(df):
    rng = np.random.default_rng(5)
    for nome, efeito in [("sem_efeito", 0.0), ("com_efeito", 1.0)]:
        x = df.copy()
        x["ganho"] = x.groupby("evento")["ganho"].transform(lambda s: rng.permutation(s.to_numpy()))
        alvo = (x["d_parou"] == "não parou no SC") & (x["d_frente_parou"] == "algum da frente parou")
        x.loc[alvo, "ganho"] += efeito
        res = S.busca(x, "ganho", DESCRITORES, "evento", 40, 0.3, np.random.default_rng(SEMENTE),
                      permutacoes=200)
        S.resume(f"sintético {nome}", res, 6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sintetico", action="store_true")
    args = ap.parse_args()
    df = monta()
    if args.sintetico:
        sintetico(df)
        return
    SAIDA.mkdir(parents=True, exist_ok=True)
    res = S.busca(df, "ganho", DESCRITORES, "evento", 40, 0.3, np.random.default_rng(SEMENTE))
    S.resume("H9 safety car", res, 25)
    res.drop(columns="chave").to_csv(SAIDA / "subgrupos.csv", index=False)
    fin = res[res["final"]].drop(columns="chave")
    fin.to_csv(SAIDA / "subgrupos_finais.csv", index=False)
    print(f"\nSubgrupos finais: {len(fin)}")


if __name__ == "__main__":
    main()
