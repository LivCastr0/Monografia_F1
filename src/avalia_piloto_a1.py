"""
Etapa 2 - Avalia o piloto do caminho A1 com critérios fixados antes de ver os dados.

Entra: data/interim/voltas/*_voltas.csv               (coleta_voltas.py)
       data/interim/corridas/*_pilotos.csv, *_stints.csv
       data/processed/base_analise_sd.csv             (corridas por perfil)

Não usa internet e não altera nenhum arquivo.

A pergunta deste script é só se vale recoletar a base inteira e reconstruir o
pipeline no grão de parada. Ele mede a qualidade dos dados e o ruído da medida.
De propósito, NÃO mostra o resultado das paradas por perfil nem a média da
medida: 10 corridas não respondem isso, e olhar agora contaminaria a análise
final, que usa as mesmas corridas.


A medida: saldo de duelos da parada
-----------------------------------
Comparar a posição antes e depois da parada engana. Quem para primeiro cai para
trás de quem ainda não parou, e essa perda desaparece quando o outro para. Por
isso a comparação é feita só contra os vizinhos que também pararam, depois que
os dois pararam:

  1. Na volta anterior à parada do piloto X, os vizinhos são os carros a até
     VIZINHANCA posições dele.
  2. Vale o duelo com o vizinho Y que parar depois de X, até JANELA voltas depois.
  3. Compara-se a ordem dos dois antes da parada de X com a ordem duas voltas
     depois da parada de Y.
  4. Y estava à frente e ficou atrás: +1. Y estava atrás e ficou à frente: -1.

O saldo da parada é a soma dos duelos. Cada duelo conta uma vez, para quem parou
primeiro. Se contasse para os dois, os valores se anulariam dentro da corrida e o
ruído pareceria menor do que é.


Paradas que entram na conta
---------------------------
Fora: volta 3 ou antes (costuma ser dano), últimas 3 voltas, parada sem troca de
pneu (punição), pneu anterior com menos de MIN_VOLTAS_STINT voltas (costuma ser
furo ou dano) e bandeira de safety car, VSC ou vermelha na volta de entrada ou na
seguinte, na parada de X ou de Y. Limitação: safety car entre as duas paradas não
é filtrado, e dano fora desses casos não é identificado.


Critérios (fixados antes de rodar; mudar só antes de ver a saída)
-----------------------------------------------------------------
  1. Posição confiável: a posição na última volta bate com o resultado oficial
     em pelo menos 90% dos pilotos que terminaram. Punição depois da bandeirada
     explica alguma diferença, por isso não é 100%.
  2. Paradas confiáveis: o número de paradas bate com os stints já coletados em
     pelo menos 95% dos pilotos.
  3. Sensibilidade: na célula mediana (perfil × 3 faixas, como na análise atual)
     a base inteira detecta efeitos de até 0,25 no saldo por parada. O valor sai
     do limite atual: 0,45 posição por corrida ÷ 1,8 parada por corrida.
  4. Espaço para o efeito: o saldo é diferente de zero em pelo menos 25% das
     paradas. Uma medida quase sempre zero teria desvio baixo e passaria no
     critério 3 sem informar nada. O valor segue do critério 3: um efeito de 0,25
     por parada exige que ao menos 1 parada em cada 4 mude a ordem.

Passando nos quatro, vale recoletar e reconstruir. Falhando 1 ou 2, o problema é
a medida ou os dados e precisa ser entendido antes. Falhando 3 ou 4, o A1 não
enxerga mais que a análise atual e não compensa o custo.

O ruído é corrigido pela dependência entre paradas da mesma corrida (efeito de
desenho, 1 + (m-1) × correlação intraclasse). Com 10 corridas essa correlação é
estimada com pouca precisão, então a sensibilidade é uma ordem de grandeza.
"""
from pathlib import Path

import numpy as np
import pandas as pd

VOLTAS = Path("data/interim/voltas")
CORRIDAS = Path("data/interim/corridas")
BASE = Path("data/processed/base_analise_sd.csv")

VIZINHANCA = 3         # posições à frente ou atrás
JANELA = 8             # voltas até o vizinho parar
VOLTAS_APOS = 2        # voltas depois da parada de Y para comparar a ordem
MIN_VOLTAS_STINT = 4
VOLTA_MINIMA = 4
VOLTAS_FINAIS = 3
CODIGOS_NEUTRALIZACAO = set("4567")   # safety car, vermelha, VSC, VSC encerrando
FAIXAS = 3             # faixas do descritor, como na análise atual

MIN_POSICAO = 0.90
MIN_PARADAS = 0.95
MAX_DETECTAVEL = 0.25
MIN_SALDO_NAO_ZERO = 0.25


def neutralizada(codigo):
    return bool(set(str(codigo)) & CODIGOS_NEUTRALIZACAO)


def le_voltas(arquivo):
    df = pd.read_csv(arquivo, dtype={"bandeira": str})
    df = df.dropna(subset=["volta"]).copy()
    df["volta"] = df["volta"].astype(int)
    return df


def acha_paradas(df):
    """Uma linha por parada: entrou nos boxes e voltou para a pista."""
    df = df.sort_values(["piloto", "volta"])
    seg = df.groupby("piloto")
    df = df.assign(
        prox_volta=seg["volta"].shift(-1),
        prox_stint=seg["stint"].shift(-1),
        prox_bandeira=seg["bandeira"].shift(-1),
        prox_saida=seg["saida_box_s"].shift(-1),
    )
    # Sem volta seguinte é quem abandonou nos boxes, não uma parada.
    par = df[df["entrada_box_s"].notna()
             & (df["prox_volta"] == df["volta"] + 1)
             & df["prox_saida"].notna()].copy()
    par = par.rename(columns={"volta": "volta_parada"})
    # Stint vazio não conta como troca: se faltar dado, o critério 2 acusa.
    par["trocou_pneu"] = (par["prox_stint"].notna() & par["stint"].notna()
                          & (par["stint"] != par["prox_stint"]))
    par["limpa"] = ~(par["bandeira"].map(neutralizada)
                     | par["prox_bandeira"].map(neutralizada))

    # Duração do pneu que saiu: desde a parada anterior, ou desde a largada.
    par["anterior"] = par.groupby("piloto")["volta_parada"].shift(1).fillna(0)
    par["voltas_pneu"] = par["volta_parada"] - par["anterior"]
    return par[["piloto", "volta_parada", "trocou_pneu", "limpa", "voltas_pneu"]]


def paradas_uteis(par, ultima_volta):
    return par[par["trocou_pneu"] & par["limpa"]
               & (par["voltas_pneu"] >= MIN_VOLTAS_STINT)
               & (par["volta_parada"] >= VOLTA_MINIMA)
               & (par["volta_parada"] <= ultima_volta - VOLTAS_FINAIS)]


def saldos(df, uteis):
    """Saldo de duelos de cada parada útil. Vazio = parada sem duelo válido."""
    pos = df.pivot_table(index="piloto", columns="volta", values="posicao", aggfunc="first")

    def p(piloto, volta):
        if piloto not in pos.index or volta not in pos.columns:
            return np.nan
        return pos.at[piloto, volta]

    linhas = []
    for x in uteis.itertuples():
        volta_base = x.volta_parada - 1
        px = p(x.piloto, volta_base)
        if pd.isna(px):
            continue
        duelos, saldo = 0, 0
        depois = uteis[(uteis["piloto"] != x.piloto)
                       & (uteis["volta_parada"] > x.volta_parada)
                       & (uteis["volta_parada"] <= x.volta_parada + JANELA)]
        # Se o vizinho parou duas vezes na janela, vale a primeira.
        for y in depois.sort_values("volta_parada").drop_duplicates("piloto").itertuples():
            py = p(y.piloto, volta_base)
            if pd.isna(py) or abs(py - px) > VIZINHANCA:
                continue
            fim = y.volta_parada + VOLTAS_APOS
            fx, fy = p(x.piloto, fim), p(y.piloto, fim)
            if pd.isna(fx) or pd.isna(fy):
                continue
            duelos += 1
            if py < px and fy > fx:
                saldo += 1
            elif py > px and fy < fx:
                saldo -= 1
        if duelos:
            linhas.append({"piloto": x.piloto, "volta_parada": x.volta_parada,
                           "duelos": duelos, "saldo": saldo})
    return pd.DataFrame(linhas, columns=["piloto", "volta_parada", "duelos", "saldo"])


def correlacao_intraclasse(valores, grupos):
    """Quanto das variações da medida é comum às paradas da mesma corrida."""
    d = pd.DataFrame({"v": valores, "g": grupos})
    g = d.groupby("g")["v"]
    k, n, ni = g.ngroups, len(d), g.size()
    if k < 2 or n <= k:
        return 0.0
    entre = (ni * (g.mean() - d["v"].mean()) ** 2).sum() / (k - 1)
    dentro = ((d["v"] - g.transform("mean")) ** 2).sum() / (n - k)
    n0 = (n - (ni ** 2).sum() / n) / (k - 1)
    denominador = entre + (n0 - 1) * dentro
    return max(0.0, (entre - dentro) / denominador) if denominador > 0 else 0.0


def avalia_corrida(arquivo):
    ano, rnd = (int(t) for t in arquivo.name.split("_")[:2])
    df = le_voltas(arquivo)
    pilotos = pd.read_csv(CORRIDAS / f"{ano}_{rnd:02d}_pilotos.csv")
    stints = pd.read_csv(CORRIDAS / f"{ano}_{rnd:02d}_stints.csv")

    # Critério 1: posição da última volta contra o resultado oficial
    ultima = df.sort_values("volta").groupby("piloto")["posicao"].last()
    status = pilotos["status"].astype(str)
    term = pilotos[status.str.match(r"^(Finished|Lapped|\+\d+ Laps?)$")]
    oficial = term.set_index("piloto")["posicao_final"]
    comum = oficial.index.intersection(ultima.dropna().index)
    bate_posicao = int((ultima.loc[comum] == oficial.loc[comum]).sum())

    # Critério 2: paradas contra os stints já coletados
    par = acha_paradas(df)
    esperadas = stints.groupby("piloto").size() - 1
    medidas = par[par["trocou_pneu"]].groupby("piloto").size()
    medidas = medidas.reindex(esperadas.index, fill_value=0)
    bate_paradas = int((medidas == esperadas).sum())

    uteis = paradas_uteis(par, int(df["volta"].max()))
    s = saldos(df, uteis)
    if len(s):
        s = s.assign(corrida=f"{ano}_{rnd:02d}")

    # Medida simples, só para comparar o ruído: posição 5 voltas depois da saída
    pos = df.set_index(["piloto", "volta"])["posicao"]
    delta = []
    for x in uteis.itertuples():
        chaves = (x.piloto, x.volta_parada - 1), (x.piloto, x.volta_parada + 6)
        if all(c in pos.index for c in chaves):
            delta.append(pos.loc[chaves[0]] - pos.loc[chaves[1]])

    resumo = {
        "corrida": f"{ano}_{rnd:02d}",
        "term_comparados": len(comum), "bate_posicao": bate_posicao,
        "pilotos": len(esperadas), "bate_paradas": bate_paradas,
        "paradas": int(par["trocou_pneu"].sum()),
        "uteis": len(uteis), "com_duelo": len(s),
        "duelos": int(s["duelos"].sum()) if len(s) else 0,
    }
    return resumo, s, delta


def veredito(passou):
    return "PASSOU" if passou else "NÃO PASSOU"


def main():
    arquivos = sorted(VOLTAS.glob("*_voltas.csv"))
    if not arquivos:
        raise FileNotFoundError(f"nenhum arquivo em {VOLTAS}: rode coleta_voltas.py")

    resumos, partes, deltas = [], [], []
    for a in arquivos:
        r, s, d = avalia_corrida(a)
        resumos.append(r)
        deltas += d
        if len(s):
            partes.append(s)

    res = pd.DataFrame(resumos)
    print(f"Corridas avaliadas: {len(res)}\n")
    print(res.to_string(index=False))
    if not partes:
        raise ValueError("nenhuma parada com duelo válido: conferir os dados")
    todos = pd.concat(partes, ignore_index=True)

    pct_pos = res["bate_posicao"].sum() / res["term_comparados"].sum()
    pct_par = res["bate_paradas"].sum() / res["pilotos"].sum()

    desvio = todos["saldo"].std()
    icc = correlacao_intraclasse(todos["saldo"], todos["corrida"])
    por_corrida = res["com_duelo"].mean()
    efeito_desenho = 1 + (por_corrida - 1) * icc

    base = pd.read_csv(BASE)
    corridas = base.groupby("d_perfil")[["ano", "round"]].apply(
        lambda g: g.drop_duplicates().shape[0]
    )
    linhas_celula = corridas * por_corrida / FAIXAS
    detectavel = 2 * desvio * np.sqrt(efeito_desenho / linhas_celula)
    # cada perfil aparece FAIXAS vezes, então a mediana das células é a dos perfis
    det_mediana = detectavel.median()

    print("\nAs paradas")
    print(f"  total {res['paradas'].sum()}, úteis {res['uteis'].sum()}, "
          f"com duelo {res['com_duelo'].sum()}")
    print(f"  com duelo por corrida: {por_corrida:.1f}; duelos por parada: "
          f"{todos['duelos'].mean():.2f}")

    print("\nO ruído da medida")
    print(f"  desvio do saldo: {desvio:.2f}")
    nao_zero = (todos["saldo"] != 0).mean()
    print(f"  paradas em que o saldo não é zero: {nao_zero:.1%}")
    print(f"  correlação intraclasse: {icc:.3f}; efeito de desenho: {efeito_desenho:.2f}")
    if deltas:
        print(f"  desvio da posição simples (5 voltas depois, só comparação): "
              f"{np.nanstd(deltas, ddof=1):.2f}")

    print("\nProjeção para a base inteira, por célula perfil × faixa")
    tabela = pd.DataFrame({"corridas": corridas,
                           "paradas_na_celula": linhas_celula.round(0).astype(int),
                           "detectavel": detectavel.round(2)})
    print(tabela.to_string())

    c1 = pct_pos >= MIN_POSICAO
    c2 = pct_par >= MIN_PARADAS
    c3 = det_mediana <= MAX_DETECTAVEL
    c4 = nao_zero >= MIN_SALDO_NAO_ZERO
    print("\n" + "=" * 72)
    print("CRITÉRIOS")
    print("=" * 72)
    print(f"1. Posição bate com o resultado: {pct_pos:.1%} "
          f"(mínimo {MIN_POSICAO:.0%})  {veredito(c1)}")
    print(f"2. Paradas batem com os stints:  {pct_par:.1%} "
          f"(mínimo {MIN_PARADAS:.0%})  {veredito(c2)}")
    print(f"3. Efeito detectável na mediana: {det_mediana:.2f} "
          f"(máximo {MAX_DETECTAVEL:.2f})  {veredito(c3)}")
    print(f"4. Saldo diferente de zero:      {nao_zero:.1%} "
          f"(mínimo {MIN_SALDO_NAO_ZERO:.0%})  {veredito(c4)}")

    print()
    if c1 and c2 and c3 and c4:
        print("Decisão: vale recoletar a base inteira e reconstruir no grão de parada.")
    elif c1 and c2:
        print("Decisão: dados bons, mas o A1 não enxerga mais que a análise atual.")
    else:
        print("Decisão: entender a falha nos dados antes de qualquer conclusão sobre o A1.")
    print("\nNada foi salvo.")


if __name__ == "__main__":
    main()
