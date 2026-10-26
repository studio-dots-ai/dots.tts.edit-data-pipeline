# Emotion/Style Editing Pipeline
# Stage I:   Input Generation (tagged text groups + edit instructions)
# Stage II:  Speech Synthesis (per-segment routing, concatenation, smoothing)
# Stage III: Evaluation — Content (ASR/WER), pair acoustic change, Quality (UTMOS), Speaker Similarity (WavLM)
# Stage IV:  Save — distributed parquet output sink

PIPELINE_TYPE = "emotion_style_edit"
PIPELINE_VERSION = "1.0.0"
