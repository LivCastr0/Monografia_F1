# Changelog

Registro cronológico das decisões e marcos do projeto.

---

## Etapa 1 — Caracterização dos circuitos

### Concluído

- Definição das 9 features do grão circuito (dinâmicas, manuais, externas)
- Escolha de sessões de classificação como fonte do fingerprint
- Exclusão de 2022 por indisponibilidade de `session_info` na API
- Substituição de `nanmin`/`nanmax` por `nanpercentile(1/99)` para robustez ao ruído
- Substituição de carga lateral (G indisponível na API) por proxy de velocidade mínima de curva
- Pipeline resiliente com cache FastF1 e registro de falhas
- 26 circuitos canônicos × 10 features, zero nulos

- Features manuais e externas integradas (pit lane, altitude, Pirelli); EDA formal (`01_eda_features.ipynb`)
- Clusterização (`02_clusterizacao_circuitos.ipynb`): KMeans, `so_dinamicas` (5 features de telemetria), k = 4; silhouette 0,270, Davies-Bouldin 1,000, ARI bootstrap 0,818
- Partição adotada: coluna `cluster_kmeans_so_dinamicas_k4` de `clusters_circuitos.csv` (11/8/5/2). A coluna `cluster_final` não é a partição adotada

### Pendente

- Conferir o sinal do heatmap de centroides do cluster de potência (o pico de desaceleração é negativo nos dados brutos)

---

## Etapa 2 — Descoberta de subgrupos nas corridas

Registros detalhados de cada análise na pasta `claude/` do projeto no claude.ai.

### Bases

- Coleta de corridas e volta a volta (`coleta_corridas.py`, `coleta_voltas.py`): 163 de 164 corridas de 2018–2025 (Monza 2018 sem dados de tempo); 2022 incluído
- Base piloto-corrida (`base_piloto_corrida.csv`, `base_analise_sd.csv`)
- Bases por grão: parada (`base_parada.csv`, A1), corrida (`base_corrida.csv`), stint (`base_stint.csv`), duelo (`base_duelos.csv`), decisão de parar (`base_decisao_parada.csv`), ritmo (`base_ritmo.csv`)
- Decisão de parar: SC/VSC medido no ponto de decisão de cada piloto e voltas de bandeira vermelha excluídas (121.075 voltas). A saída da verificação feita antes da correção fica em `verificacao_horario_sc_parada.csv` (antes `base_decisao_parada_sc_corrigido.csv`), só como registro

### Análises

- Rodadas 1 e 2 (piloto-corrida): resultado dominado por vieses do alvo e por "parou mais que o pelotão"
- A1 (saldo de posições na parada), B (largada → chegada), pneu × circuito, carro × perfil, degradação, afinidade carro × perfil, assinatura do cluster
- Duelos (`busca_subgrupos_duelos.py`) e decisão de parar (`busca_subgrupos_decisao_parada.py`): análises principais; regras consolidadas em `regras_globais_duelos.csv` e `regras_globais_parada.csv`
- Abrangência por perfil, poder do teste de partições, SD por cluster (`valida_sd_por_cluster.py`): nenhum padrão depende do tipo de circuito
- Estabilidade da descoberta (`estabilidade_descoberta.py`): as regras reaparecem entre eras e em 100 divisões de circuitos
- Eficácia das paradas (`eficacia_regras_parada.py`): paradas feitas nas situações das regras rendem menos posições que as demais
- SD com as features da Etapa 1 como descritores, H1–H5 (`sd_features_etapa1.py`): nenhuma confirmada pela SD; H1 e H3 passam no critério contínuo
- Pilotos de ar sujo e de setores: medidas sem resolução para análise completa; setores com mais frenagens por km degradam mais
- Fichas descritivas dos clusters (`fichas_clusters.py`)
- Tipologia comportamental (`tipologia_comportamental.py`, critérios fixados antes de rodar): 8 medidas de corrida são estáveis por circuito, mas o agrupamento não se repete entre metades das corridas (ARI 0,08); a análise para no passo 4
- Perfil físico × medidas estáveis (`perfil_medidas_estaveis.py`, definido depois da tipologia comportamental): sem os urbanos, o perfil explica menos de 6% de cada uma das 8 medidas, abaixo do acaso; figura em `data/processed/perfil_medidas_estaveis/`
- H6, pit lane × decisão de parar (`sd_pit_lane_parada.py`, exploratória): não confirmada; pit lane alto sob SC tem +4,0 p.p. no sentido previsto, mas abaixo do limiar; esticar o stint não aparece
- H7, comportamento de parada por equipe (`equipe_decisao_parada.py`, exploratória): não confirmada; no top 5, as equipes de ponta respondem menos à parada do rival de trás (sentido oposto); as regras variam entre equipes mais que o acaso, mas confundido com a posição
- H8, zonas de DRS × ultrapassagem (`sd_drs_duelos.py`, exploratória; contagem de zonas em `data/external/zonas_drs_por_circuito.csv`): não confirmada; correlação com a taxa de ultrapassagem e com o efeito D3 perto de zero

### Correções em 2026-11-11

- `valida_sd_por_cluster.py` refeito com a base de decisão de parar atual (a rodada anterior usava a base antes da correção do SC); saídas versionadas em `data/processed/sd_por_cluster/`. Conclusão sem mudança (parada p = 0,485; duelos p = 0,542)
