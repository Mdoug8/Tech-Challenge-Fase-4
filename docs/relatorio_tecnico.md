# Relatório técnico — Tech Challenge Fase 4

## 1. Introdução e objetivo
## 2. Fluxo multimodal
Inserir o diagrama de arquitetura e descrever entrada → processamento → fusão → alerta.

## 3. Modelos aplicados por tipo de dado
| Dado | Modelo/serviço | Saída |
|---|---|---|
| Vídeo | YOLOv8-pose / OpenPose; YOLOv8 detecção | Ângulos, desvios, objetos fora da área |
| Áudio | Azure Speech to Text; Azure Language; librosa | Transcrição, sentimento, termos críticos, features vocais |
| Vitais | Limites clínicos, z-score móvel, Isolation Forest | Anomalias por sinal e multivariadas |
| Prescrições | Regras (dose, suspensão, interação) | Alterações inesperadas |
| Movimentação | Z-score móvel, imobilidade | Picos bruscos, imobilidade |
| Fusão | Late fusion ponderada + bônus de concordância | Score e severidade |

## 4. Datasets (PhysioNet, AudioSet, vídeos usados) e pré-processamento
**Áudio:** 3 consultas **simuladas** (síntese de voz TTS do macOS, voz Luciana pt-BR, convertidas para WAV 16 kHz mono)
em `data/samples/audio/`: `normal` (sem queixas), `alerta` (dor no peito, falta de ar, tontura, desmaio) e `lenta`
(fala pausada com fraqueza e cansaço). **Não são gravações de pacientes reais.**
## 5. Resultados
### 5.3 Áudio (3 consultas simuladas por TTS)
**Transcrição:** feita pela API Knowlez STT (`src/audio/http_transcriber.py`), **não** pelo Azure Speech
(`src/audio/transcriber.py` existe, mas não foi executado por falta de chave). Saídas em `data/samples/audio/transcricoes_knowlez.json`.

| Áudio | Transcrição obtida | Termos críticos detectados | Alertas de voz |
|---|---|---|---|
| normal | "Bom dia doutor. Estou me sentindo bem, dormi bem esta noite..." (idêntica ao roteiro) | nenhum | nenhum |
| alerta | "Doutor, desde ontem estou com dor no peito e falta de ar. Tive tontura e quase desmaiei..." (idêntica) | dor no peito (0,9), falta de ar (0,8), desmaiei (0,8), tontura (0,5), cansado (0,4) | nenhum |
| lenta | "Eu, estou com muita fraqueza, e cansaço, não o consigo, falar direito..." (1 erro de transcrição: "não o consigo") | fraqueza (0,5), cansaço (0,4) | nenhum |

**Features vocais (librosa):**

| Áudio | Duração | Pausas | Taxa de fala | RMS | Jitter F0 |
|---|---|---|---|---|---|
| normal | 8,4 s | 6% | 149 wpm | 0,167 | 0,031 |
| alerta | 9,3 s | 6% | 148 wpm | 0,163 | 0,029 |
| lenta | 8,4 s | 11% | 114 wpm | 0,155 | 0,024 |

**Leitura honesta dos resultados:** a detecção de termos críticos funcionou e não houve falso alarme no áudio normal.
Porém, como a voz é sintética, nenhum áudio ultrapassou os limiares de fadiga (pausas > 50%, < 80 wpm, RMS < 0,01,
jitter > 0,05), então **a detecção de fadiga/disartria por voz não foi validada**; só a extração das features.
Sentimento (Azure Language): **[PREENCHER — não executado, sem chave]**.

- Exemplos de anomalias detectadas (prints, tabelas, relatório HTML do vídeo)
- Métricas, se houver rótulos (precisão/recall nas anomalias injetadas)
## 6. Fluxo de alerta (Event Grid → Teams/painel)
## 7. Limitações, ética e LGPD
## 8. Próximos passos

# Roteiro do vídeo (até 15 min)
1. (1 min) Problema e arquitetura
2. (3 min) Vídeo: upload, pose, desvios, relatório
3. (3 min) Áudio: transcrição Azure, termos críticos, sentimento, features vocais
4. (3 min) Anomalias: vitais com SpO2 caindo, prescrição alterada
5. (3 min) Fusão, alerta no Teams e painel Streamlit
6. (2 min) Resultados e limitações
