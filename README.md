# SUF-CSR

**S**tage-wise **U**niversal Information Extraction Framework with **C**omplex **S**entence **R**e-extraction for CitrusKG construction.

Reproducible code for the paper *SUF-CSR: A Stage-wise Universal Information Extraction Framework with Complex Sentence Re-extraction for CitrusKG Construction*.

## Repository layout

```
SUF-CSR/
├── CitrusNER/     # NER dataset (annotations + UIE train/dev/test splits)
├── CitrusRE/      # RE dataset (annotations + UIE train/dev/test splits)
├── UIE/           # SUF-NER, SUF-RE, SUF-CSR pipeline & experiments
└── LLM/           # Qwen / DeepSeek few-shot baselines (Tables 5–6)
```

## Quick start

### 1. UIE experiments (SUF-NER, SUF-RE, SUF-CSR)

```bash
cd UIE
pip install -r requirements.txt
# Place models in checkpoints/ — see UIE/checkpoints/README.md
bash scripts/run_ner_eval.sh    # Table 5
bash scripts/run_suf_csr.sh     # Table 6 SUF-CSR
bash scripts/run_ablation.sh    # Table 7
```

### 2. LLM baselines

```bash
cd LLM
pip install -r requirements.txt
cp config/qwen.json.example config/qwen.json   # add API key
bash scripts/run_ner_eval.sh
bash scripts/run_re_scheme_a.sh
bash scripts/run_re_scheme_b.sh
```

## Method summary

1. **SUF-NER**: entity recognition under Schema-NER (8 entity types)
2. **SUF-RE**: relation extraction under Schema-RE given NER outputs
3. **CSR**: re-extract from LLM-simplified sentences when confidence &lt; 0.8
4. **Aggregation**: entity alignment + conflict resolution → CitrusKG (Neo4j)

See `UIE/README.md` and `LLM/README.md` for experiment-to-paper mapping.

## Citation

If you use this code or datasets, please cite the SUF-CSR paper.

## License

UIE core code follows the [uie_pytorch](https://github.com/heiheiyoyo/uie_pytorch) Apache 2.0 license.
