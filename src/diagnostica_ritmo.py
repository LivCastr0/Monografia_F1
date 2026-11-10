"""
Etapa 2 - Diagnóstico do critério 2 do piloto de ritmo (companheiros concordam).

Entra: data/interim/voltas/*_voltas.csv
       data/processed/base_piloto_corrida.csv
Não salva nada e não usa internet. Precisa de avalia_piloto_ritmo.py na mesma pasta.

No piloto, a concordância entre companheiros deu 0,64 (mínimo 0,70) e não passou.
Esse resultado continua registrado como está. Este diagnóstico foi definido DEPOIS
de ver a falha e deve ser relatado assim.

A hipótese: parte da diferença entre companheiros é fixa (um piloto é mais rápido
que o outro a temporada inteira). Essa parte não afeta a análise final, porque o
alvo é o ritmo do carro (média dos dois) menos a média do próprio carro na
temporada: uma diferença fixa entre os pilotos se cancela. O que afeta é a parte
que muda de corrida para corrida.

Para medir as duas partes são precisas várias corridas do mesmo par na mesma
temporada, por isso o diagnóstico usa todas as corridas secas. Ele só mostra medidas
de qualidade: nenhum ritmo por perfil, circuito ou equipe.

Regra, fixada antes de rodar este script:
  A. Com o ruído que muda de corrida para corrida, a afinidade detectável com 5
     corridas fica em até 0,30% do tempo de volta (o mesmo limite do critério 4).
  B. Os casos de diferença muito grande (mais de 3 vezes a diferença típica) são no
     máximo 5% dos carros-corrida.
Passando em A e B, a medida segue para a base inteira. Senão, a hipótese (ii) fica
registrada como sem medida confiável.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from avalia_piloto_ritmo import BASE, VOLTAS, ritmo_por_piloto

MAX_DETECTAVEL, MAX_EXTREMOS = 0.30, 0.05


def main():
    base = pd.read_csv(BASE).drop_duplicates(["ano", "round"])
    secas = base[(base["faixa_chuva"] == "seco") & ~base["corrida_sem_disputa"]]
    linhas = []
    for ano, rnd in zip(secas["ano"], secas["round"]):
        arquivo = VOLTAS / f"{ano}_{rnd:02d}_voltas.csv"
        if not arquivo.exists():
            continue
        r = ritmo_por_piloto(pd.read_csv(arquivo, dtype={"bandeira": str}))
        if r.empty:
            continue
        r["ritmo"] = r["ritmo"] - r["ritmo"].mean()
        for equipe, g in r.groupby("equipe"):
            if len(g) == 2:
                g = g.sort_values("piloto")
                (pa, ra), (pb, rb) = g[["piloto", "ritmo"]].to_numpy()
                linhas.append({"ano": ano, "round": rnd, "equipe": equipe,
                               "par": f"{pa}/{pb}", "diferenca": ra - rb})
    d = pd.DataFrame(linhas)
    print(f"Carros-corrida: {len(d)}, corridas: {d.groupby(['ano', 'round']).ngroups}")

    grupo = ["equipe", "ano", "par"]
    d["corridas_do_par"] = d.groupby(grupo)["diferenca"].transform("size")
    u = d[d["corridas_do_par"] >= 3].copy()
    u["fixa"] = u.groupby(grupo)["diferenca"].transform("mean")
    u["variavel"] = u["diferenca"] - u["fixa"]
    graus = len(u) - u.groupby(grupo).ngroups
    var_total = u["diferenca"].var(ddof=1)
    var_variavel = (u["variavel"] ** 2).sum() / graus
    print(f"Pares com 3+ corridas na mesma temporada: {u.groupby(grupo).ngroups}, "
          f"carros-corrida: {len(u)}")
    print(f"\nDiferença entre companheiros (desvio):")
    print(f"  total:                          {np.sqrt(var_total):.3f}%")
    print(f"  só a parte que muda por corrida: {np.sqrt(var_variavel):.3f}%")
    print(f"  parte fixa na temporada:        {1 - var_variavel / var_total:.0%} da variação")

    # Ruído do ritmo do carro (média dos dois pilotos) vindo da parte variável
    ruido_carro = np.sqrt(var_variavel) / 2
    detectavel = 2 * ruido_carro / np.sqrt(5)
    tipico = np.sqrt(var_variavel)
    extremos = (u["variavel"].abs() > 3 * tipico).mean()

    print(f"\nCarros-corrida com diferença extrema (mais de {3 * tipico:.2f}% fora do "
          f"padrão do par): {extremos:.1%}")
    maiores = u.reindex(u["variavel"].abs().sort_values(ascending=False).index).head(10)
    print(maiores[["ano", "round", "equipe", "par", "diferenca", "fixa", "variavel"]]
          .to_string(index=False, float_format=lambda x: f"{x:+.2f}"))

    a, b = detectavel <= MAX_DETECTAVEL, extremos <= MAX_EXTREMOS
    v = lambda ok: "PASSOU" if ok else "NÃO PASSOU"
    print("\n" + "=" * 72 + "\nREGRA DO DIAGNÓSTICO\n" + "=" * 72)
    print(f"A. Afinidade detectável (n=5), só ruído variável: {detectavel:.3f}% "
          f"(máximo {MAX_DETECTAVEL:.2f}%)  {v(a)}")
    print(f"B. Diferenças extremas: {extremos:.1%} (máximo {MAX_EXTREMOS:.0%})  {v(b)}")
    print()
    print("Conclusão: a diferença entre companheiros é sobretudo fixa; a medida segue para a "
          "base inteira." if a and b else
          "Conclusão: a medida não é confiável para a afinidade carro × perfil.")
    print("\nNada foi salvo.")


if __name__ == "__main__":
    main()
