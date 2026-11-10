"""
Etapa 2 - Piloto do ar sujo: dá para medir, volta a volta, quanto um carro perde
(ou ganha) por começar a volta logo atrás de outro?

Entra: data/interim/voltas/<ano>_<round>_voltas.csv
       data/processed/base_duelos.csv   (só para a lista de corridas secas e o perfil)
Sai:   nada (só imprime). Não usa internet.

Uso:   python src/avalia_piloto_ar_sujo.py              (as 10 corridas do piloto)
       python src/avalia_piloto_ar_sujo.py --sintetico  (teste da medida)

Pergunta da análise futura: em que condições seguir outro carro custa ou rende tempo?
Hipótese ligada à Etapa 1: perda em pistas de curvas rápidas, ganho de vácuo em pistas
de reta. Este piloto só verifica se a MEDIDA é confiável. De propósito, NÃO mostra a
medida por perfil, por corrida, por era ou por qualquer outro descritor; só os
critérios de qualidade e a média de todas as corridas juntas por faixa de distância.

Corridas do piloto
------------------
Sorteio novo, uma vez, com default_rng(2026), entre as 122 corridas secas da base de
duelos: 3 alto downforce, 3 stop-and-go, 3 potência e 1 urbano-técnico, com as duas
eras em cada perfil de 3. Motivo de não reaproveitar a lista dos pilotos anteriores:
ela tem só 1 corrida de potência e 1 corrida fora da base seca, e o vácuo nas retas
pode mudar o comportamento da medida justamente na potência. Decidido antes de rodar.

Definições
----------
Volta válida (do piloto X, volta L):
  - L >= 3 (DRS desligado nas duas primeiras voltas);
  - as voltas L-1 e L são de bandeira verde para todos os carros;
  - X não entra nem sai do box em L; tempo preciso (volta_precisa) e não gerado pela
    FastF1; pneu seco;
  - tempo até 107% da mediana das voltas de X que cumprem o resto (tira rodadas,
    problemas e danos, como no limite de erro dos duelos);
  - o carro à frente (abaixo) não entra nem sai do box na volta que ele começa ali.
Carro à frente na pista: o último carro (qualquer um, inclusive retardatário) a cruzar
a linha antes de X no fim de L-1. Distância = diferença dos horários de cruzamento.
Faixa, SÓ pela distância no início da volta (a do fim dependeria do próprio tempo de
volta, que é o que se mede):
  seguindo    até 1,0 s
  perto       1,0 a 2,0 s
  médio       2,0 a 3,0 s
  limpo       3,0 s ou mais, ou sem carro à frente (líder)
Ritmo em ar limpo: por corrida, mínimos quadrados sobre as voltas limpas,
  log(tempo) = intercepto do stint do piloto + idade do pneu por composto
               + tendência da volta da corrida (linear + quadrática, comum a todos)
  O stint só recebe intercepto com 4+ voltas limpas. Combustível e idade do pneu não
  se separam bem (andam juntos no primeiro stint), mas o que importa aqui é a previsão,
  que é única mesmo quando os coeficientes não são.
Medida: excesso = 100 * (tempo / ritmo limpo previsto - 1), em % da volta.
  Positivo = perdeu tempo; negativo = ganhou.
Vantagem de ritmo: ritmo limpo previsto do carro da frente, na volta que ele começa
  ali, menos o de X, em %. Positiva = X é mais rápido em ar limpo. Quando X é mais
  rápido, ele pode estar preso atrás e o excesso mede "preso", não aerodinâmica. Quando
  não é mais rápido (vantagem <= 0), nada o segura e o excesso mede o efeito de seguir.

Critérios, fixados antes de rodar (mudar só antes de ver a saída)
-----------------------------------------------------------------
  C1 cobertura: em pelo menos 9 das 10 corridas, 85%+ das voltas válidas recebem
     previsão de ritmo limpo.
  C2 calibração: ajuste cruzado (voltas limpas alternadas dentro de cada stint; ajusta
     numa metade e prevê a outra). Em todas as corridas juntas, erro absoluto mediano
     <= 0,40% e erro médio entre -0,10% e +0,10%.
  C3 resolução: em pelo menos 8 das 10 corridas, 50+ voltas "seguindo" e erro-padrão
     do excesso médio dessas voltas <= 0,10%.
  C4 separação de "preso atrás": pelo menos 30% das voltas "seguindo" têm vantagem
     calculada e <= 0 (o de trás não é mais rápido em ar limpo).
  C5 especificação: o excesso das voltas "seguindo" calculado com outra especificação
     (efeito fixo por volta da corrida + idade do pneu quadrática por composto) tem
     correlação de Pearson >= 0,90 com a principal, em todas as corridas juntas.
Regra de decisão: passa nos 5 -> seguir para a base completa. Falha em C1, C2 ou C5
-> rever o modelo de ritmo. Falha em C3 ou C4 -> a medida por volta não tem
resolução; considerar a versão por setor.

Teste sintético (antes da base real): corridas simuladas com efeito de ar sujo
conhecido, carros presos atrás quando mais rápidos e ultrapassagens ocasionais. A
medida nas voltas com vantagem <= 0 deve recuperar o efeito plantado.

Associação, não causa.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

VOLTAS = Path("data/interim/voltas")
BASE = Path("data/processed/base_duelos.csv")
SEMENTE = 2026
SORTEIO = {"alto downforce": 3, "stop-and-go": 3, "potência": 3, "urbano-técnico": 1}
SECOS = {"HYPERSOFT", "ULTRASOFT", "SUPERSOFT", "SOFT", "MEDIUM", "HARD"}
FAIXAS = [("seguindo", 0.0, 1.0), ("perto", 1.0, 2.0), ("médio", 2.0, 3.0)]
LIMPO_S, VOLTA_MIN, MIN_LIMPAS_STINT, LIMITE_LENTA = 3.0, 3, 4, 1.07

C1_COB, C1_CORRIDAS = 0.85, 9
C2_MEDIANA, C2_MEDIA = 0.40, 0.10
C3_N, C3_EP, C3_CORRIDAS = 50, 0.10, 8
C4_MIN = 0.30
C5_MIN = 0.90


def titulo(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


# ---------------------------------------------------------------------------
def sorteia_corridas():
    b = pd.read_csv(BASE, usecols=["ano", "round", "circuito", "d_perfil", "d_era"]) \
        .drop_duplicates(["ano", "round"]).sort_values(["ano", "round"]).reset_index(drop=True)
    rng = np.random.default_rng(SEMENTE)
    escolha = []
    for perfil, n in SORTEIO.items():
        g = b[b["d_perfil"] == perfil]
        g = g.iloc[rng.permutation(len(g))]
        pega = []
        if n >= 2:                                   # uma de cada era primeiro
            for era in rng.permutation(sorted(g["d_era"].unique())):
                pega.append(g[g["d_era"] == era].index[0])
        for i in g.index:
            if len(pega) >= n:
                break
            if i not in pega:
                pega.append(i)
        escolha += pega[:n]
    return b.loc[escolha].reset_index(drop=True)


# ---------------------------------------------------------------------------
def prepara(v):
    """Uma linha por piloto-volta, com o carro à frente no início da volta e a faixa."""
    v = v.dropna(subset=["volta", "fim_volta_s"]).copy()
    v["volta"] = v["volta"].astype(int)
    v["bandeira"] = v["bandeira"].astype(str).str.replace(r"\.0$", "", regex=True)
    verde = v.groupby("volta")["bandeira"].agg(lambda s: bool((s == "1").all()))
    v["box"] = v["entrada_box_s"].notna() | v["saida_box_s"].notna()
    v["stint_id"] = v["piloto"] + "_" + v["stint"].astype(str)

    # cruzamentos ordenados: para cada um, o último cruzamento de outro carro antes dele
    cruz = v[["piloto", "volta", "fim_volta_s"]].sort_values("fim_volta_s").reset_index(drop=True)
    t, p, vo = cruz["fim_volta_s"].to_numpy(), cruz["piloto"].to_numpy(), cruz["volta"].to_numpy()
    f_p = np.full(len(cruz), None, dtype=object)
    f_v = np.full(len(cruz), -1)
    f_d = np.full(len(cruz), np.nan)
    for i in range(1, len(cruz)):
        j = i - 1
        while j >= 0 and p[j] == p[i]:
            j -= 1
        if j >= 0:
            f_p[i], f_v[i], f_d[i] = p[j], vo[j] + 1, t[i] - t[j]   # volta que o da frente começa
    cruz["frente"], cruz["frente_volta"], cruz["dist_ini"] = f_p, f_v, f_d
    cruz["volta"] = cruz["volta"] + 1                                # o cruzamento abre a volta seguinte
    v = v.merge(cruz.drop(columns="fim_volta_s"), on=["piloto", "volta"], how="left")

    box_frente = v.set_index(["piloto", "volta"])["box"]
    chave = list(zip(v["frente"], v["frente_volta"]))
    v["frente_box"] = [bool(box_frente.get(k, False)) if k[0] is not None and not pd.isna(k[0]) else False
                       for k in chave]

    preciso = v["volta_precisa"].astype(str).str.lower().eq("true") & \
        ~v["gerada_pela_fastf1"].astype(str).str.lower().eq("true")
    base = ((v["volta"] >= VOLTA_MIN) & v["volta"].map(verde).fillna(False).astype(bool)
            & (v["volta"] - 1).map(verde).fillna(False).astype(bool) & ~v["box"] & preciso
            & v["composto"].isin(SECOS) & v["tempo_volta_s"].notna() & v["vida_pneu"].notna())
    mediana = v[base].groupby("piloto")["tempo_volta_s"].median()
    lenta = v["tempo_volta_s"] > LIMITE_LENTA * v["piloto"].map(mediana)
    v["valida"] = base & ~lenta & ~v["frente_box"]

    sem_frente = v["frente"].isna()
    d = v["dist_ini"]
    v["faixa"] = np.select([sem_frente | (d >= LIMPO_S)] + [(d >= a) & (d < b) for _, a, b in FAIXAS],
                           ["limpo"] + [n for n, _, _ in FAIXAS], default=None)
    return v


def desenho(v, spec, voltas_ref):
    cols = {}
    for s in sorted(v["stint_id"].unique()):
        cols[f"st_{s}"] = (v["stint_id"] == s).astype(float)
    for c in sorted(v["composto"].dropna().unique()):
        m = (v["composto"] == c).astype(float)
        cols[f"vida_{c}"] = m * v["vida_pneu"].fillna(0)
        if spec == "alternativa":
            cols[f"vida2_{c}"] = m * v["vida_pneu"].fillna(0) ** 2
    if spec == "principal":
        L = v["volta"].astype(float)
        cols["volta"], cols["volta2"] = L, L ** 2
    else:
        for L in voltas_ref[1:]:
            cols[f"v_{L}"] = (v["volta"] == L).astype(float)
    return pd.DataFrame(cols, index=v.index)


def ajusta(limpas, spec):
    ref = sorted(limpas["volta"].unique())
    X = desenho(limpas, spec, ref)
    coef, *_ = np.linalg.lstsq(X.to_numpy(), np.log(limpas["tempo_volta_s"].to_numpy()), rcond=None)
    return dict(zip(X.columns, coef)), ref


def preve(coef, v, spec, ref):
    """log do ritmo limpo previsto; NaN para stint sem intercepto ou volta sem efeito."""
    st = v["stint_id"].map(lambda s: coef.get(f"st_{s}", np.nan))
    vc = v["composto"].map(lambda c: coef.get(f"vida_{c}", np.nan))
    y = st + vc * v["vida_pneu"]
    if spec == "principal":
        y = y + coef["volta"] * v["volta"] + coef["volta2"] * v["volta"] ** 2
    else:
        y = y + v["composto"].map(lambda c: coef.get(f"vida2_{c}", 0.0)) * v["vida_pneu"] ** 2
        ef = {ref[0]: 0.0, **{L: coef[f"v_{L}"] for L in ref[1:]}}
        y = y + v["volta"].map(lambda L: ef.get(L, np.nan))
    return y.to_numpy(dtype=float)


def mede(v):
    """Ajusta o ritmo limpo e devolve (voltas válidas com a medida, erros do ajuste cruzado)."""
    ok = v["valida"] & v["faixa"].notna()
    limpas = v[ok & (v["faixa"] == "limpo")]
    conta = limpas.groupby("stint_id").size()
    limpas = limpas[limpas["stint_id"].isin(conta[conta >= MIN_LIMPAS_STINT].index)]

    coef, ref = ajusta(limpas, "principal")
    v = v.copy()
    v["prev"] = np.exp(preve(coef, v, "principal", ref))          # todas as voltas, para a vantagem
    coef_a, ref_a = ajusta(limpas, "alternativa")
    v["prev_alt"] = np.exp(preve(coef_a, v, "alternativa", ref_a))

    ritmo = v.set_index(["piloto", "volta"])["prev"]
    rf = [ritmo.get(k, np.nan) if k[0] is not None and not pd.isna(k[0]) else np.nan
          for k in zip(v["frente"], v["frente_volta"])]
    v["vantagem"] = 100 * (np.asarray(rf, dtype=float) / v["prev"] - 1)

    val = v[ok].copy()
    val["excesso"] = 100 * (val["tempo_volta_s"] / val["prev"] - 1)
    val["excesso_alt"] = 100 * (val["tempo_volta_s"] / val["prev_alt"] - 1)

    limpas = limpas.sort_values(["stint_id", "volta"]).copy()
    limpas["metade"] = limpas.groupby("stint_id").cumcount() % 2
    erros = []
    for m in (0, 1):
        treino, teste = limpas[limpas["metade"] == m], limpas[limpas["metade"] != m]
        c, r = ajusta(treino, "principal")
        erros.append(100 * (teste["tempo_volta_s"].to_numpy() / np.exp(preve(c, teste, "principal", r)) - 1))
    erros = np.concatenate(erros)
    return val, erros[~np.isnan(erros)]


# ---------------------------------------------------------------------------
def avalia(corridas, leitor):
    linhas, todas, erros = [], [], []
    for _, c in corridas.iterrows():
        val, err = mede(prepara(leitor(c)))
        seg = val[(val["faixa"] == "seguindo") & val["excesso"].notna()]
        linhas.append({"corrida": f"{c['ano']}_{int(c['round']):02d}",
                       "voltas_validas": len(val),
                       "cobertura": val["prev"].notna().mean(),
                       "voltas_seguindo": len(seg),
                       "ep_seguindo": seg["excesso"].std(ddof=1) / np.sqrt(len(seg)) if len(seg) > 1 else np.nan})
        todas.append(val)
        erros.append(err)
    t = pd.DataFrame(linhas)
    todas = pd.concat(todas, ignore_index=True)
    erros = np.concatenate(erros)
    seg = todas[(todas["faixa"] == "seguindo") & todas["excesso"].notna()]

    titulo("CRITÉRIOS POR CORRIDA (sem a medida em si)")
    print(t.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    c1 = int((t["cobertura"] >= C1_COB).sum())
    med, media = np.median(np.abs(erros)), erros.mean()
    c3 = int(((t["voltas_seguindo"] >= C3_N) & (t["ep_seguindo"] <= C3_EP)).sum())
    nao_mais_rapido = seg["vantagem"] <= 0
    c4 = nao_mais_rapido.mean() if len(seg) else 0.0
    ok5 = seg[["excesso", "excesso_alt"]].dropna()
    c5 = ok5["excesso"].corr(ok5["excesso_alt"]) if len(ok5) > 2 else np.nan

    titulo("CRITÉRIOS")
    crit = [
        ("C1 cobertura", f"{c1} de {len(t)} corridas com {C1_COB:.0%}+", f">= {C1_CORRIDAS}", c1 >= C1_CORRIDAS),
        ("C2 calibração", f"|erro| mediano {med:.3f}%, erro médio {media:+.3f}% ({len(erros)} voltas)",
         f"<= {C2_MEDIANA}% e |média| <= {C2_MEDIA}%", bool(med <= C2_MEDIANA and abs(media) <= C2_MEDIA)),
        ("C3 resolução", f"{c3} de {len(t)} corridas", f">= {C3_CORRIDAS}", c3 >= C3_CORRIDAS),
        ("C4 de trás não mais rápido", f"{c4:.1%} de {len(seg)} voltas seguindo", f">= {C4_MIN:.0%}", bool(c4 >= C4_MIN)),
        ("C5 especificação", f"Pearson {c5:.3f} ({len(ok5)} voltas)", f">= {C5_MIN}", bool(c5 >= C5_MIN)),
    ]
    for nome, valor, limite, ok in crit:
        print(f"{nome:28s} {'PASSA' if ok else 'FALHA':6s} {valor}   (limite {limite})")

    titulo("TODAS AS CORRIDAS JUNTAS, POR FAIXA DE DISTÂNCIA (informativo)")
    f = todas[todas["excesso"].notna()].groupby("faixa")["excesso"].agg(["size", "mean", "median", "std"])
    f["ep"] = f["std"] / np.sqrt(f["size"])
    print(f.reindex(["limpo", "médio", "perto", "seguindo"]).to_string(float_format=lambda x: f"{x:+.3f}"))
    if len(seg):
        s0 = seg.loc[nao_mais_rapido, "excesso"]
        s1 = seg.loc[seg["vantagem"] > 0, "excesso"]
        print(f"\nSeguindo, de trás não mais rápido: {s0.mean():+.3f}% (ep {s0.std(ddof=1)/np.sqrt(len(s0)):.3f}) "
              f"em {len(s0)} voltas")
        print(f"Seguindo, de trás mais rápido:     {s1.mean():+.3f}% (ep {s1.std(ddof=1)/np.sqrt(len(s1)):.3f}) "
              f"em {len(s1)} voltas")
    passou = all(ok for *_, ok in crit)
    print("\nDecisão pela regra fixada:", "seguir para a base completa" if passou
          else "ver a regra de decisão no cabeçalho")
    return t, crit, todas


# ---------------------------------------------------------------------------
def corrida_sintetica(rng, efeito_ar, n=20, voltas=55):
    """Corrida simulada. Começar a volta a até 1 s de outro carro soma efeito_ar % ao
    tempo; um carro mais rápido fica preso atrás (no mínimo 0,25 s) e passa com
    probabilidade 0,25 por volta quando é 0,3 s+ mais rápido. Uma parada por carro."""
    ritmo = 90 * (1 + rng.normal(0, 0.006, n))
    parada = rng.integers(18, 36, n)
    deg = {"SOFT": 0.06, "MEDIUM": 0.04, "HARD": 0.025}
    comp1 = rng.choice(["SOFT", "MEDIUM"], n)
    ultimo = np.arange(n) * 1.2
    linhas = []
    for L in range(1, voltas + 1):
        ordem = np.argsort(ultimo)
        novo = np.empty(n)
        for pos, i in enumerate(ordem):
            st = 1 if L <= parada[i] else 2
            comp = comp1[i] if st == 1 else "HARD"
            vida = L if st == 1 else L - parada[i]
            limpo = ritmo[i] - 0.06 * L + deg[comp] * vida
            t = limpo * (1 + rng.normal(0, 0.003))
            box_in, box_out = L == parada[i], L == parada[i] + 1
            if box_in:
                t += 20
            if pos > 0 and L >= 2 and not box_in:
                j = ordem[pos - 1]
                if ultimo[i] - ultimo[j] < 1.0:
                    t *= 1 + efeito_ar / 100
                    tj = novo[j] - ultimo[j]
                    frente_no_box = L == parada[j]
                    passa = (tj - t > 0.3) and rng.random() < 0.25
                    if not passa and not frente_no_box:
                        t = max(t, novo[j] + 0.25 - ultimo[i])
            novo[i] = ultimo[i] + t
            linhas.append({"ano": 2099, "round": 1, "piloto": f"P{i:02d}", "equipe": "X", "volta": L,
                           "posicao": pos + 1, "tempo_volta_s": t, "fim_volta_s": novo[i],
                           "entrada_box_s": 1.0 if box_in else np.nan,
                           "saida_box_s": 1.0 if box_out else np.nan,
                           "stint": st, "composto": comp, "vida_pneu": vida, "pneu_novo": True,
                           "bandeira": "1", "volta_precisa": not (box_in or box_out),
                           "gerada_pela_fastf1": False})
        ultimo = novo
    return pd.DataFrame(linhas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sintetico", action="store_true")
    args = ap.parse_args()
    if args.sintetico:
        for efeito in (0.0, 0.5):
            rng = np.random.default_rng(7)
            titulo(f"SINTÉTICO: efeito de ar sujo plantado = {efeito:+.1f}% (10 corridas)")
            corridas = pd.DataFrame({"ano": [2099] * 10, "round": range(1, 11)})
            cache = {r: corrida_sintetica(rng, efeito) for r in range(1, 11)}
            avalia(corridas, lambda c: cache[c["round"]])
        return
    corridas = sorteia_corridas()
    titulo("CORRIDAS DO PILOTO (sorteio com default_rng(2026))")
    print(corridas.to_string(index=False))
    avalia(corridas, lambda c: pd.read_csv(VOLTAS / f"{c['ano']}_{int(c['round']):02d}_voltas.csv"))


if __name__ == "__main__":
    main()
