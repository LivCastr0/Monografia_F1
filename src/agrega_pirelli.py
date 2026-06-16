"""
Etapa 1 - Derivação: agrega as notas Pirelli por circuito.

Lê as notas 1-5 coletadas manualmente ano a ano (cards Pirelli) e colapsa
para um valor único por circuito via mediana, ignorando os anos faltantes.

Sobre os faltantes: NÃO são perda de dado, são mudança de formato do card.
  - 2018-2019: sem traction, braking, track_evolution
  - 2020:      sem track_evolution
  - 2021+:     os 8 ratings presentes
A mediana é calculada sobre os anos disponíveis de CADA feature, então a
janela efetiva difere por coluna. Aceitável e documentado no caderno.

Limitação conhecida: Sochi pode não ter track_evolution (correu até 2021).
Mantido como NaN - o tratamento é decidido a jusante, não aqui.
"""
import re
import pandas as pd

ENTRADA = "data/external/notas_pirelli_por_ano.csv"
SAIDA = "data/processed/pirelli_agregado_por_circuito.csv"


def separa_circuito_ano(rotulo):
    """'Austin 2025' -> ('Austin', 2025). Linhas-seção (só nome) -> ano None."""
    m = re.match(r"^(.*?)\s+(\d{4})$", str(rotulo).strip())
    return (m.group(1).strip(), int(m.group(2))) if m else (str(rotulo).strip(), None)


# Detecta separador automaticamente (sep=None) e tolera utf-8/latin-1.
# utf-8 PRIMEIRO: ler utf-8 como latin-1 corrompe acentos (Montréal -> MontrÃ©al).
try:
    raw = pd.read_csv(ENTRADA, sep=None, engine="python", encoding="utf-8")
except (UnicodeDecodeError, UnicodeError):
    raw = pd.read_csv(ENTRADA, sep=None, engine="python", encoding="latin-1")

raw = raw.rename(columns={raw.columns[0]: "rotulo"})
raw["circuito"], raw["ano"] = zip(*raw["rotulo"].map(separa_circuito_ano))

# Linhas sem ano são separadores visuais da planilha -> descartar
dados = raw[raw["ano"].notna()].copy()

# As colunas de feature são todas exceto rótulo/circuito/ano - detectadas, não fixadas
features = [c for c in raw.columns if c not in ("rotulo", "circuito", "ano")]
for c in features:
    dados[c] = pd.to_numeric(dados[c], errors="coerce")

print("Colunas de feature detectadas:")
for c in features:
    print(f"  - {c}")

# Mediana por circuito (pula NaN). n_anos = nº de anos do circuito (robusto)
agregado = dados.groupby("circuito")[features].median()
agregado.insert(0, "n_anos", dados.groupby("circuito")["ano"].nunique())

agregado.round(2).reset_index().to_csv(SAIDA, index=False)
print(f"\nOK: {SAIDA} ({agregado.shape[0]} circuitos)")

# Auditoria das lacunas reais (célula sem nenhum ano)
contagem = dados.groupby("circuito")[features].count()
vazias = [(c, f) for c in contagem.index for f in features if contagem.loc[c, f] == 0]
if vazias:
    print("ATENCAO - celulas sem nenhum dado (decidir tratamento a jusante):")
    for c, f in vazias:
        print(f"  {c} / {f}")