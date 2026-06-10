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

### Pendente

- Coleta manual das features #1, #5, #6 (site F1) e integração com #7 (Pirelli) e #9 (altitude)
- EDA formal sobre o vetor completo de features
