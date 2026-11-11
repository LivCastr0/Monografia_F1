"""
Etapa 2 - H11: decisão no safety car (e troca na chuva) x resultado no fim da corrida.

Critérios fixados antes de rodar: claude/safety-car-chuva-msi2.md, seção H11 (projeto no
claude.ai). Definida depois da H9; exploratória declarada.

Alvo `ganho_final`: posições ganhas da referência até a classificação final, reordenadas
entre os pilotos que estavam na referência e terminaram a corrida.
  Safety car: referência = fim da volta anterior à do início do SC (mesma base da H9).
  Chuva:      referência = fim da volta anterior à troca (mesmas trocas da H10c).

Entra: data/processed/base_safety_car.csv, base_chuva_trocas.csv, data/interim/...
Sai:   data/processed/sd_resultado_final/
Uso:   python src/sd_resultado_final.py [--sintetico]
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
import sd_safety_car as SC  # noqa: E402
import sd_chuva as CH  # noqa: E402

P = Path("data/processed")
SAIDA = P / "sd_resultado_final"
SEMENTE = 2026


def final_por_corrida(ano, rnd):
    v = A.le_voltas(A.VOLTAS / f"{ano}_{rnd:02d}_voltas.csv").dropna(subset=["posicao"])
    pos = v.pivot_table(index="volta", columns="piloto", values="posicao")
    pl = pd.read_csv(A.CORRIDAS / f"{ano}_{rnd:02d}_pilotos.csv")
    term = pl[pl["terminou"] == True].dropna(subset=["posicao_final"])  # noqa: E712
    return pos, term.set_index("piloto")["posicao_final"]


def acrescenta_final(df, coluna_volta_ref, evento):
    out = []
    for (ano, rnd), g in df.groupby(["ano", "round"]):
        pos, fin = final_por_corrida(int(ano), int(rnd))
        for ev, h in g.groupby(evento):
            lref = int(h[coluna_volta_ref].iloc[0])
            if lref not in pos.index:
                continue
            ref = pos.loc[lref].dropna()
            pres = ref.index.intersection(fin.index)
            r0, r1 = ref[pres].rank(), fin[pres].rank()
            h = h[h["piloto"].isin(pres)].copy()
            h["ganho_final"] = [float(r0[p] - r1[p]) for p in h["piloto"]]
            out.append(h)
    return pd.concat(out, ignore_index=True)


def base_sc():
    b = pd.read_csv(P / "base_safety_car.csv")
    b["volta_ref"] = b["volta_inicio"] - 1
    return acrescenta_final(b, "volta_ref", "evento")


def base_chuva():
    t = pd.read_csv(P / "base_chuva_trocas.csv")
    t["volta_ref"] = t["volta"] - 1
    t["evento_troca"] = t["evento"] + "_" + t["piloto"] + "_" + t["volta"].astype(str)
    # cada troca é avaliada contra o pelotão da sua volta; a permutação é dentro da corrida
    partes = []
    for _, linha in t.iterrows():
        partes.append(acrescenta_final(linha.to_frame().T.assign(
            ano=int(linha["ano"]), round=int(linha["round"]), volta_ref=int(linha["volta_ref"])),
            "volta_ref", "evento_troca"))
    out = pd.concat(partes, ignore_index=True)
    out["ganho_final"] = out["ganho_final"].astype(float)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sintetico", action="store_true")
    args = ap.parse_args()
    sc = base_sc()
    ch = base_chuva()
    print(f"Safety car: {len(sc)} linhas (de 836 na H9; saem os que abandonaram depois), "
          f"{sc['evento'].nunique()} períodos; média do ganho final {sc['ganho_final'].mean():+.3f}")
    print(f"Chuva: {len(ch)} trocas (de 255 na H10c)")
    if args.sintetico:
        rng = np.random.default_rng(5)
        for nome, efeito in [("embaralhado", 0.0), ("com_efeito", 1.0)]:
            x = sc.copy()
            x["ganho_final"] = x.groupby("evento")["ganho_final"].transform(
                lambda s: rng.permutation(s.to_numpy()))
            x.loc[x["d_parou"] == "parou no SC", "ganho_final"] += efeito
            r = S.busca(x, "ganho_final", SC.DESCRITORES, "evento", 40, 0.3,
                        np.random.default_rng(SEMENTE), permutacoes=200)
            S.resume(f"sintético {nome}", r, 5)
        return
    SAIDA.mkdir(parents=True, exist_ok=True)
    sc.to_csv(SAIDA / "base_safety_car_final.csv", index=False)
    ch.to_csv(SAIDA / "base_chuva_trocas_final.csv", index=False)
    r = S.busca(sc, "ganho_final", SC.DESCRITORES, "evento", 40, 0.3, np.random.default_rng(SEMENTE))
    S.resume("H11 safety car -> fim da corrida", r, 25)
    r.drop(columns="chave").to_csv(SAIDA / "sc_subgrupos.csv", index=False)

    # Comparação descritiva: subgrupos finais da H9, janela curta x fim da corrida
    h9 = pd.read_csv(P / "sd_safety_car" / "subgrupos_finais.csv")
    linhas = []
    for d in h9["descricao"]:
        m = np.ones(len(sc), dtype=bool)
        for cond in d.split(" E "):
            c, v = cond.split("=", 1)
            m &= sc[f"d_{c}"].astype(str).to_numpy() == v
        linhas.append({"descricao": d, "linhas": int(m.sum()),
                       "ganho_3_voltas": sc.loc[m, "ganho"].mean(),
                       "ganho_fim_da_corrida": sc.loc[m, "ganho_final"].mean()})
    comp = pd.DataFrame(linhas)
    comp.to_csv(SAIDA / "comparacao_h9.csv", index=False)
    print("\nSubgrupos da H9 (só quem terminou): janela curta x fim da corrida")
    print(comp.round(2).to_string(index=False))

    rc = S.busca(ch, "ganho_final", CH.DESC_TROCA, "evento", 20, 0.5,
                 np.random.default_rng(SEMENTE), min_era=10)
    S.resume("H11 chuva (troca -> fim da corrida)", rc, 12)
    if not rc.empty:
        rc.drop(columns="chave").to_csv(SAIDA / "chuva_subgrupos.csv", index=False)


if __name__ == "__main__":
    main()
