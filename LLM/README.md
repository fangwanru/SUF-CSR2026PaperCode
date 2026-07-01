# LLM Baseline Experiments

Few-shot LLM baselines for NER and RE on CitrusNER / CitrusRE (Tables 5–6).

## Paper correspondence

| Paper | Setting | Script |
|-------|---------|--------|
| Table 5 | Qwen-3-max NER (zero/few-shot) | `scripts/run_ner_eval.sh` → `NER/metrics_qwen.json` |
| Table 5 | DeepSeek-v4 NER | `scripts/run_ner_eval.sh` → `NER/metrics_deepseek.json` |
| Table 6 | Qwen-3-max-A (Scheme A, e2e RE) | `scripts/run_re_scheme_a.sh` |
| Table 6 | Qwen-3-max-B (Scheme B, staged RE) | `scripts/run_re_scheme_b.sh` |
| Table 6 | DeepSeek-v4-A / B | same scripts (DeepSeek outputs) |

**Scheme A**: single-stage end-to-end entity + relation extraction.  
**Scheme B**: Stage 1 NER → Stage 2 relation extraction (few-shot).

CSR in the paper uses Qwen for sentence splitting inside `../UIE/run_citrus_lowconf_split_pipeline.py`.

## Setup

```bash
cd LLM
pip install -r requirements.txt

cp config/qwen.json.example config/qwen.json
cp config/deepseek.json.example config/deepseek.json
# Edit API keys in config/*.json
```

## Run

```bash
bash scripts/run_ner_eval.sh      # Table 5 LLM NER
bash scripts/run_re_scheme_a.sh   # Table 6 Scheme A
bash scripts/run_re_scheme_b.sh   # Table 6 Scheme B
```

Metrics are written as JSON with per-type and overall P/R/F1.

## Directory layout

```
LLM/
├── config/           # API configs (not committed with keys)
├── NER/              # NER prediction scripts
├── RE/               # Scheme A (e2e RE)
├── two_stage/        # Scheme B (NER → RE)
└── scripts/          # Experiment runners
```

Data: `../CitrusNER/`, `../CitrusRE/` (few-shot examples from `CitrusNER.jsonl` / `CitrusRE.jsonl`).
