# F1 Estratégia — Identificação de Padrões Estratégicos via Clustering e Subgroup Discovery

MSI

## Objetivo

Identificar padrões estratégicos na Fórmula 1 cruzando duas análises complementares:

- **Clustering de circuitos** (unidade de análise: circuito): agrupa pistas com perfis físicos semelhantes
- **Subgroup Discovery** (unidade de análise: piloto-corrida): encontra subgrupos com comportamentos estratégicos distintos

Os clusters de circuito foram planejados como ponte entre as duas análises. Na Etapa 2, eles são usados para testar se os padrões encontrados pela descoberta de subgrupos dependem do tipo de circuito.

---

## Estrutura do projeto

```
Monografia-f1/
├── src/                          # Scripts de coleta e processamento
│   ├── coleta_calendario.py      # Calendário de sessões
│   ├── circuitos_canonicos.py    # Definição dos 26 circuitos
│   ├── extrai_features_circuito.py  # Telemetria e features dinâmicas
│   └── status_extracao.py        # Conferência offline do cache
├── notebooks/                    # Análises exploratórias e experimentos
├── data/
│   ├── raw/
│   ├── calendario_2018_2025.csv
│   ├── circuitos_canonicos.csv
│   ├── sessoes_incluidas.csv
│   ├── voltas_fingerprint.csv
│   ├── temp_por_corrida.csv
│   └── falhas_extracao.csv
└── processed/
    └── features_fastf1_por_circuito.csv
```

---

## Etapas

### ✅ Etapa 1 — Caracterização dos circuitos 

Coleta e engenharia de features para o grão circuito.

**Features definidas:**

| # | Feature | Fonte | Sessão |
|---|---------|-------|--------|
| 1 | Perda de tempo no pit lane (s) | Manual (site F1) | — |
| 2 | Velocidade mínima de curva / % baixa velocidade | FastF1 telemetria | Quali |
| 3 | Carga longitudinal (dv/dt suavizado, zonas de freio) | FastF1 telemetria | Quali |
| 4 | % full throttle | FastF1 telemetria | Quali |
| 5 | Dificuldade de ultrapassagem | Manual (site F1) | — |
| 6 | Probabilidade de SC/VSC | Manual (site F1) | — |
| 7 | Abrasividade do asfalto | Pirelli (externo) | — |
| 8 | Temperatura de pista (°C) | FastF1 weather | Quali |
| 9 | Altitude (m) | Referência externa | — |

**Decisões metodológicas relevantes:**
- Sessões de **classificação** usadas para o fingerprint (combustível mínimo, sem ruído de estratégia)
- **2022 excluído integralmente**: a API da F1 não serve `session_info` dessa temporada, inviabilizando toda a telemetria
- Agregação por **mediana** das voltas mais rápidas por piloto por sessão
- Extremos calculados com `nanpercentile(1/99)` em vez de min/max, para robustez ao ruído
- 26 circuitos canônicos a partir de 142 sessões de quali (2018–2025)

**Artefatos gerados:**
- `data/raw/calendario_2018_2025.csv`
- `data/raw/circuitos_canonicos.csv`
- `data/raw/sessoes_incluidas.csv`
- `data/raw/voltas_fingerprint.csv`
- `data/raw/features_fastf1_por_circuito.csv`
- `data/raw/falhas_extracao.csv`
- `data/external/template_features_manuais.xlsx`

**Clusterização (concluída):** KMeans sobre 5 features de telemetria (`so_dinamicas`), k = 4, em `notebooks/02_clusterizacao_circuitos.ipynb`. Partição adotada: coluna `cluster_kmeans_so_dinamicas_k4` de `data/processed/clusters_circuitos.csv` (alto downforce 11, stop-and-go 8, potência 5, urbano-técnico 2).

### 🔄 Etapa 2 — Descoberta de subgrupos nas corridas (MSI II)

- Bases por grão (corrida, parada, stint, duelo, decisão de parar) a partir da coleta volta a volta de 163 corridas
- Análises principais: duelos (`base_duelos.csv`) e decisão de parar (`base_decisao_parada.csv`), com regras consolidadas em `regras_globais_*.csv`
- Validações com a tipologia da Etapa 1: abrangência por perfil, SD por cluster, teste de partições e de poder, estabilidade da descoberta
- Histórico das análises em `CHANGELOG.md`

---

## Fonte de dados

- **FastF1 3.8.3** — telemetria oficial da F1 (via API pública)
- **Site oficial da F1** — features manuais (#1, #5, #6)
- **Pirelli** — escala de abrasividade (#7)
- **Referências externas** — altitude (#9)

## Ambiente

```bash
# Criar e ativar ambiente virtual
python -m venv venv
source venv/bin/activate

# Instalar dependências
pip install -r requirements.txt
```

## Dependências principais

Ver `requirements.txt`. Bibliotecas centrais: `fastf1`, `pandas`, `numpy`, `scikit-learn`, `pysubgroup`, `matplotlib`, `seaborn`.
