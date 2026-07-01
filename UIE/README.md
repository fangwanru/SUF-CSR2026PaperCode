# SUF-CSR UIE Experiments

Stage-wise Universal Information Extraction (SUF) and Complex Sentence Re-extraction (CSR) for citrus knowledge graph construction.

## Paper correspondence

| Paper section | Script | Output |
|---------------|--------|--------|
| Table 5 — UIE-base NER | `scripts/run_ner_eval.sh` (uie_base) | `outputs/citrus/ner_eval/report_uie_base.txt` |
| Table 5 — SUF-NER | `scripts/run_ner_eval.sh` (suf_ner) | `outputs/citrus/ner_eval/report_suf_ner.txt` |
| Table 6 — UIE-RE-tuned | `run_e2e_re.py` | `outputs/citrus/evaluate_e2e_no_ner.txt` |
| Table 6 — SUF-CSR | `scripts/run_suf_csr.sh` | `outputs/citrus/suf_csr/evaluate_report.txt` |
| Table 7 — w/o CSR | `scripts/run_suf_two_stage.sh` | `outputs/citrus/evaluate_two_stage_no_csr.txt` |
| Table 7 — w/o NER | `run_e2e_re.py` | `outputs/citrus/evaluate_e2e_no_ner.txt` |
| Figure 5 — threshold | `scripts/run_threshold_sweep.sh` | `outputs/citrus/threshold_sweep/` |

## Setup

```bash
cd UIE
pip install -r requirements.txt
pip install openai   # CSR sentence splitting via Qwen API
```

1. Download UIE-base to `checkpoints/uie-base-en/` (see `checkpoints/README.md`)
2. Fine-tune or copy SUF-NER / SUF-RE weights to `checkpoints/`
3. Configure Qwen API in `../LLM/config/qwen.json` for CSR

## Quick start

```bash
# Fine-tune (optional if you have checkpoints)
bash scripts/finetune_ner.sh
bash scripts/finetune_re.sh

# NER comparison (Table 5)
bash scripts/run_ner_eval.sh

# Full SUF-CSR (Table 6)
bash scripts/run_suf_csr.sh

# Ablation (Table 7)
bash scripts/run_ablation.sh

# Threshold sweep (Figure 5)
export DASHSCOPE_API_KEY=your_key
bash scripts/run_threshold_sweep.sh
```

## Core modules

| File | Role |
|------|------|
| `finetune.py` | Train SUF-NER / SUF-RE on CitrusNER / CitrusRE |
| `predict_citrus_ner.py` | NER inference with Schema-NER |
| `predict_citrus_re.py` | End-to-end RE (w/o NER stage) |
| `run_suf_two_stage.py` | SUF: NER → RE (no CSR) |
| `run_citrus_lowconf_split_pipeline.py` | CSR: LLM split + re-extract + merge |
| `run_e2e_re.py` | Single-stage RE baseline |
| `convert_for_uie_eval.py` | Convert predictions for official UIE metrics |
| `schema/citrus_schema.json` | Entity/relation ontology (Tables 1–2) |

## Method overview

```
Sentence X
  → SUF-NER(X, Schema-NER) → entities E
  → SUF-RE(X, E, Schema-RE) → triples T with confidence φ
  → if ∅T or min(φ) < θ (θ=0.8): LLM split X → {C₁…Cₖ}; SUF(Cᵢ) → T′
  → merge high-conf T ∪ T′ with conflict resolution
```

Dataset paths point to `../CitrusNER` and `../CitrusRE`.
