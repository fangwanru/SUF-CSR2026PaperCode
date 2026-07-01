# CitrusRE

Relation extraction (RE) dataset for the citrus domain, with joint NER and RE annotations.

## Files

| File | Description |
|------|-------------|
| `CitrusRE.jsonl` | Full annotations in Doccano export format (**1742** documents, **1754** records) |
| `train.txt` | Training set (UIE format) |
| `dev.txt` | Validation set (UIE format) |
| `test.txt` | Test set (UIE format) |
| `test_sentences.txt` | Plain-text test sentences, one per line (LLM input) |

`train.txt`, `dev.txt`, and `test.txt` are split from `CitrusRE.jsonl` at roughly **7 : 1 : 2** and converted to UIE training format. One sentence may span multiple lines (expanded by entity-type `prompt`).

## Entity Types (8)

`compound` · `citrus` · `by-product` · `aroma` · `health` · `location` · `taste` · `mouth_feel`

## Relation Types (8)

| Relation | Meaning |
|----------|---------|
| `contain` | Citrus or by-product contains a compound |
| `produce` | Citrus produces a by-product |
| `produced_in` | Citrus is produced in a location |
| `has_aroma` | Has a certain aroma |
| `has_taste` | Has a certain taste |
| `has_mouth_feel` | Has a certain mouthfeel |
| `impact_on` | Impact on health |
| `category` | Entity category assignment |

## CitrusRE.jsonl Format

One JSON object per line:

```json
{
  "id": 1449,
  "text": "In the high-end fragrance and flavor industry, bergamot ...",
  "entities": [
    {"id": 19166, "label": "citrus", "start_offset": 47, "end_offset": 55},
    {"id": 19165, "label": "by-product", "start_offset": 47, "end_offset": 87}
  ],
  "relations": [
    {"id": 13071, "from_id": 19166, "to_id": 19165, "type": "produce"}
  ]
}
```

- `from_id` / `to_id`: reference entity `id` values in `entities`
- `type`: relation type

## train / dev / test Format

Same as CitrusNER: UIE format (`content` + `result_list` + `prompt`) for fine-tuning and evaluation.
