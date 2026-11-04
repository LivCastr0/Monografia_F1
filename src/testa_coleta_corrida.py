"""
Etapa 2 - Teste antes da coleta.

Duas perguntas, nessa ordem:

1. Numa corrida que a gente sabe que funciona (2023, round 1), todos os campos
   que a Etapa 2 precisa realmente vêm do FastF1? No formato esperado?
2. As corridas de 2022 carregam? Na Etapa 1 a classificação de 2022 falhava
   ("Failed to load session_info data!"). Corrida pode ser diferente - não dá
   pra assumir nem que sim nem que não.

Nada aqui é salvo em disco. É só diagnóstico: roda, lê o relatório, decide.
Gasta ~4 sessões do limite de 500 requisições por hora.

Diferente da Etapa 1: telemetry=False. Os dados de sensor do carro (velocidade
e acelerador ponto a ponto) não servem pra nada aqui e são o que pesa no download.
"""
import fastf1
import pandas as pd

fastf1.Cache.enable_cache("data/cache")

ANO_REFERENCIA, ROUND_REFERENCIA = 2023, 1
ROUNDS_2022_TESTE = [1, 10, 20]

# Campos que a base de piloto-corrida precisa, por tabela do FastF1.
CAMPOS_RESULTADOS = [
    "DriverNumber", "Abbreviation", "TeamName",
    "GridPosition", "Position", "ClassifiedPosition", "Status", "Points",
]
CAMPOS_VOLTAS = [
    "Driver", "LapNumber", "LapTime",
    "Stint", "Compound", "TyreLife", "FreshTyre",
    "PitInTime", "PitOutTime",
]
CAMPOS_CLIMA = ["Time", "Rainfall", "TrackTemp", "AirTemp"]
CAMPOS_BANDEIRAS = ["Time", "Status"]


def carregar_corrida(ano, rnd):
    """Carrega a corrida sem os dados de sensor do carro."""
    ses = fastf1.get_session(int(ano), int(rnd), "R")
    ses.load(laps=True, telemetry=False, weather=True, messages=False)
    return ses


def confere_campos(nome_tabela, df, esperados):
    """Diz quais campos vieram e quais faltaram."""
    if df is None or len(df) == 0:
        print(f"  {nome_tabela}: VAZIA")
        return False
    faltando = [c for c in esperados if c not in df.columns]
    if faltando:
        print(f"  {nome_tabela}: {len(df)} linhas - FALTANDO {faltando}")
        return False
    print(f"  {nome_tabela}: {len(df)} linhas - todos os campos presentes")
    return True


# ---------------------------------------------------------------------------
# Parte 1 - a corrida de referência tem tudo que a Etapa 2 pede?
# ---------------------------------------------------------------------------
print("=" * 70)
print(f"PARTE 1 - corrida de referência: {ANO_REFERENCIA} round {ROUND_REFERENCIA}")
print("=" * 70)

ses = carregar_corrida(ANO_REFERENCIA, ROUND_REFERENCIA)
print(f"Evento: {ses.event['EventName']} ({ses.event['Location']})")
print(f"Total de voltas da corrida: {ses.total_laps}\n")

print("Campos por tabela:")
ok_resultados = confere_campos("resultados", ses.results, CAMPOS_RESULTADOS)
ok_voltas = confere_campos("voltas", ses.laps, CAMPOS_VOLTAS)
ok_clima = confere_campos("clima", ses.weather_data, CAMPOS_CLIMA)
ok_bandeiras = confere_campos("bandeiras", ses.track_status, CAMPOS_BANDEIRAS)

# --- quantas linhas a base vai ter por corrida ---
print(f"\nPilotos na corrida: {len(ses.results)}")
print("  (é isso vezes ~164 corridas que dá o tamanho final da base)")

# --- a pegadinha do grid zero ---
grid_zero = (ses.results["GridPosition"] == 0).sum()
print(f"\nLargadas do box (GridPosition = 0): {grid_zero}")
if grid_zero:
    quem = ses.results.loc[ses.results["GridPosition"] == 0, "Abbreviation"].tolist()
    print(f"  pilotos: {quem}")
print("  lembrete: zero não é posição, é largada do box. Não entra cru na conta"
      " de posições ganhas.")

# --- dá pra reconstruir as trocas de pneu com número de volta? ---
print("\nTrocas de pneu - teste de reconstrução (3 pilotos de exemplo):")
stints = (
    ses.laps
    .groupby(["Driver", "Stint"])
    .agg(
        composto=("Compound", "first"),
        volta_inicio=("LapNumber", "min"),
        volta_fim=("LapNumber", "max"),
        pneu_novo=("FreshTyre", "first"),
    )
    .reset_index()
)
print(stints[stints["Driver"].isin(stints["Driver"].unique()[:3])].to_string(index=False))
print(f"\n  total de stints na corrida: {len(stints)}")
print("  se as colunas volta_inicio/volta_fim vieram preenchidas, a semana 3"
      " consegue calcular duração de stint e momento da parada.")

# --- safety car e chuva: dá pra medir intensidade, não só sim/não? ---
print("\nContexto da corrida:")
bandeiras = ses.track_status.copy()
# códigos do FastF1: 1 pista livre, 2 amarela, 4 safety car,
# 5 bandeira vermelha, 6 VSC ativado, 7 VSC encerrando
contagem = bandeiras["Status"].value_counts().sort_index()
print(f"  códigos de bandeira que apareceram: {dict(contagem)}")
print(f"  períodos de safety car (código 4): {(bandeiras['Status'] == '4').sum()}")
print(f"  períodos de VSC (código 6): {(bandeiras['Status'] == '6').sum()}")

if "Rainfall" in ses.weather_data.columns:
    chuva = ses.weather_data["Rainfall"]
    print(f"  medições de clima: {len(chuva)}")
    print(f"  fração das medições com chuva: {chuva.mean():.1%}")

print("\nStatus finais (quem terminou, quem abandonou e por quê):")
print(ses.results["Status"].value_counts().to_string())

todos_ok = all([ok_resultados, ok_voltas, ok_clima, ok_bandeiras])
print(f"\n>>> PARTE 1: {'tudo presente' if todos_ok else 'FALTOU COISA - ver acima'}")


# ---------------------------------------------------------------------------
# Parte 2 - 2022 carrega ou repete a falha da Etapa 1?
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PARTE 2 - 2022 entra ou não entra")
print("=" * 70)

resultado_2022 = []
for rnd in ROUNDS_2022_TESTE:
    try:
        s = carregar_corrida(2022, rnd)
        tem_voltas = not s.laps.empty
        tem_resultados = len(s.results) > 0
        tem_clima = not s.weather_data.empty
        ok = tem_voltas and tem_resultados and tem_clima
        print(f"  2022 round {rnd}: {'OK' if ok else 'INCOMPLETO'} - "
              f"{len(s.results)} pilotos, {len(s.laps)} voltas, "
              f"clima {'sim' if tem_clima else 'não'}")
        resultado_2022.append(ok)
    except Exception as e:
        print(f"  2022 round {rnd}: FALHOU - {type(e).__name__}: {e}")
        resultado_2022.append(False)

print()
if all(resultado_2022):
    print(">>> 2022 funciona nas corridas. Pode entrar: +22 corridas, ~440 linhas.")
    print("    Escrever na metodologia que a tipologia foi construída sem 2022")
    print("    mas é aplicada às corridas de 2022.")
elif any(resultado_2022):
    print(">>> 2022 funciona em parte. Vale testar os 22 rounds antes de decidir.")
else:
    print(">>> 2022 falha também nas corridas. Fica de fora, igual à Etapa 1.")
    print("    Base final: 142 corridas, ~2.840 linhas.")