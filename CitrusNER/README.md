# CitrusNER

Named entity recognition (NER) dataset for the citrus domain.

## Files

| File | Description |
|------|-------------|
| `CitrusNER.jsonl` | Full annotations in Doccano export format (**1742** documents, **1754** records) |
| `train.txt` | Training set (UIE format) |
| `dev.txt` | Validation set (UIE format) |
| `test.txt` | Test set (UIE format) |

`train.txt`, `dev.txt`, and `test.txt` are split from `CitrusNER.jsonl` at roughly **7 : 1 : 2** and converted to [UIE](https://github.com/heiheiyoyo/uie_pytorch) training format. One sentence may span multiple lines (expanded by entity-type `prompt`).

## Entity Types (8)

`compound` · `citrus` · `by-product` · `aroma` · `health` · `location` · `taste` · `mouth_feel`

## CitrusNER.jsonl Format

One JSON object per line:

```json
{
  "id": 1449,
  "text": "In the high-end fragrance and flavor industry, bergamot ...",
  "entities": [
    {"id": 27538, "label": "location", "start_offset": 93, "end_offset": 108},
    {"id": 27539, "label": "by-product", "start_offset": 47, "end_offset": 87}
  ],
  "relations": []
}
```

- `start_offset` / `end_offset`: character offsets in `text` (left-closed, right-open)
- `relations` is empty for NER

## train / dev / test Format

One JSON object per line (UIE training format):

```json
{
  "content": "Potassium sorbate ... in lemon juice ...",
  "result_list": [{"text": "lemon", "start": 101, "end": 106}],
  "prompt": "citrus"
}
```

- `prompt`: entity type for the current line
- `result_list`: entity spans of that type; empty if none
