#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""消融实验核心推理逻辑（SciERC / CoNLL04 / citrus）。"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from ablation_config import AblationConfig, LTONG_DIR, PROJECT_ROOT, UIE_DIR

for p in (UIE_DIR, PROJECT_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from utils import dbc2sbc  # noqa: E402
from uie1_predictor import UIEPredictor  # noqa: E402

try:
    from scierc_uie_infer import align_predictor_prompt_style_to_scierc_training
except ImportError:
    align_predictor_prompt_style_to_scierc_training = None  # type: ignore

CITRUS_PREDEFINED_TRIPLES = [
    ("citrus", "category", "citrus"),
    ("citrus", "contain", "compound"),
    ("citrus", "has_aroma", "aroma"),
    ("citrus", "has_mouth_feel", "mouth_feel"),
    ("citrus", "has_taste", "taste"),
    ("citrus", "impact_on", "health"),
    ("citrus", "produce", "by-product"),
    ("citrus", "produced_in", "location"),
    ("compound", "category", "compound"),
    ("compound", "has_aroma", "aroma"),
    ("compound", "has_taste", "taste"),
    ("compound", "impact_on", "health"),
    ("by-product", "category", "by-product"),
    ("by-product", "contain", "compound"),
    ("by-product", "has_aroma", "aroma"),
    ("by-product", "has_mouth_feel", "mouth_feel"),
    ("by-product", "has_taste", "taste"),
    ("by-product", "impact_on", "health"),
    ("by-product", "produced_in", "location"),
    ("aroma", "category", "aroma"),
    ("taste", "category", "taste"),
    ("mouth_feel", "category", "mouth_feel"),
    ("location", "category", "location"),
]


def normalize_text(text: str) -> str:
    text = (text or "").strip()
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    return text


def normalize_label(label: str) -> str:
    alias = {"by-product": "by_product", "by_product": "by-product"}
    return alias.get((label or "").strip(), (label or "").strip())


def find_first_span(content: str, mention: str) -> Tuple[int, int]:
    if not mention or not content:
        return -1, -1
    i = content.find(mention)
    if i < 0:
        return -1, -1
    return i, i + len(mention)


def span_probability(item: dict) -> float:
    """UIE span 置信度（start/end 概率的 min，与 position_prob 过滤口径一致）。"""
    prob = item.get("probability", item.get("score"))
    if prob is None:
        return 1.0
    try:
        return float(prob)
    except (TypeError, ValueError):
        return 1.0


def span_record(item: dict, *, extra: Optional[dict] = None) -> dict:
    rec = {
        "text": item.get("text", ""),
        "start": item["start"],
        "end": item["end"],
        "probability": span_probability(item),
    }
    if extra:
        rec.update(extra)
    return rec


def to_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def write_jsonl(path: str, rows: List[dict]) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(to_serializable(row), ensure_ascii=False) + "\n")


def load_sentences(path: str) -> List[str]:
    out: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if t:
                out.append(t)
    return out


def resolve_task_path(model_root: str) -> str:
    model_root = os.path.abspath(model_root)
    model_best = os.path.join(model_root, "model_best")
    if os.path.isdir(model_best):
        return model_best
    return model_root


def _unwrap_ner_output(output: Any) -> Dict[str, Any]:
    if isinstance(output, list):
        merged: Dict[str, list] = defaultdict(list)
        for block in output:
            if not isinstance(block, dict):
                continue
            for ent_type, items in block.items():
                if isinstance(items, list):
                    merged[ent_type].extend(items)
        return dict(merged)
    if isinstance(output, dict):
        return output
    return {}


def export_oracle_ner_from_test(cfg: AblationConfig, output: str) -> None:
    by_content: Dict[str, Dict[str, List[dict]]] = defaultdict(lambda: defaultdict(list))
    seen: Dict[str, Dict[str, Set[tuple]]] = defaultdict(lambda: defaultdict(set))

    with open(cfg.test_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            content = normalize_text(row.get("content", ""))
            prompt = normalize_text(row.get("prompt", ""))
            if not content or prompt not in cfg.ner_types:
                continue
            for item in row.get("result_list", []):
                if not isinstance(item, dict):
                    continue
                text = normalize_text(item.get("text", ""))
                start, end = item.get("start"), item.get("end")
                if not text or not isinstance(start, int) or not isinstance(end, int):
                    continue
                key = (start, end, text)
                if key in seen[content][prompt]:
                    continue
                seen[content][prompt].add(key)
                by_content[content][prompt].append(
                    {"text": text, "start": start, "end": end, "probability": 1.0}
                )

    rows = []
    for text in load_sentences(cfg.input_txt):
        content = normalize_text(text)
        block = dict(by_content.get(content, {}))
        if cfg.name == "citrus":
            rows.append({"input": text, "output": [block] if block else []})
        else:
            rows.append({"input": text, "output": block})

    write_jsonl(output, rows)
    n_ent = sum(len(v) for r in rows for v in _unwrap_ner_output(r["output"]).values())
    print(f"oracle NER: {output} ({len(rows)} sentences, {n_ent} entities)")


def export_oracle_ner_from_conll_json(cfg: AblationConfig, output: str) -> None:
    with open(cfg.conll04_gold_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    by_content: Dict[str, Dict[str, List[dict]]] = defaultdict(lambda: defaultdict(list))
    seen: Dict[str, Dict[str, Set[tuple]]] = defaultdict(lambda: defaultdict(set))

    for item in data:
        tokens = [str(x) for x in item.get("tokens", [])]
        content = " ".join(tokens)
        spans = []
        pos = 0
        for idx, tok in enumerate(tokens):
            s, e = pos, pos + len(tok)
            spans.append((s, e))
            pos = e + (1 if idx < len(tokens) - 1 else 0)

        for ent in item.get("entities", []):
            ent_type = normalize_label(str(ent.get("type", "")))
            if ent_type not in cfg.ner_types:
                continue
            ts, te = ent.get("start"), ent.get("end")
            if not isinstance(ts, int) or not isinstance(te, int):
                continue
            if ts < 0 or te > len(spans) or te <= ts:
                continue
            cs = spans[ts][0]
            ce = spans[te - 1][1]
            text = content[cs:ce]
            key = (cs, ce, text)
            if key in seen[content][ent_type]:
                continue
            seen[content][ent_type].add(key)
            by_content[content][ent_type].append(
                {"text": text, "start": cs, "end": ce, "probability": 1.0}
            )

    rows = []
    for text in load_sentences(cfg.input_txt):
        content = normalize_text(text)
        block = dict(by_content.get(content, {}))
        rows.append({"input": text, "output": block})

    write_jsonl(output, rows)
    print(f"oracle NER (conll json): {output} ({len(rows)} sentences)")


def export_e2e_ner_types(cfg: AblationConfig, e2e_pred_file: str, output: str) -> None:
    pred_by_content: Dict[str, dict] = {}
    with open(e2e_pred_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            content = normalize_text(row.get("content", ""))
            if content:
                pred_by_content[content] = row.get("pred", {})

    rows = []
    for text in load_sentences(cfg.input_txt):
        content = normalize_text(text)
        pred = pred_by_content.get(content, {})
        block: Dict[str, List[dict]] = defaultdict(list)
        seen: Dict[str, Set[tuple]] = defaultdict(set)
        if isinstance(pred, dict):
            for ent_type, items in pred.items():
                norm_type = normalize_label(ent_type)
                if norm_type not in cfg.ner_types or not isinstance(items, list):
                    continue
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    t = normalize_text(item.get("text", ""))
                    s, e = item.get("start"), item.get("end")
                    if not t or not isinstance(s, int) or not isinstance(e, int):
                        continue
                    key = (s, e, t)
                    if key in seen[norm_type]:
                        continue
                    seen[norm_type].add(key)
                    block[norm_type].append(
                        {
                            "text": t,
                            "start": s,
                            "end": e,
                            "probability": float(
                                item.get("probability", item.get("score", 1.0)) or 1.0
                            ),
                        }
                    )
        if cfg.name == "citrus":
            rows.append({"input": text, "output": [dict(block)] if block else []})
        else:
            rows.append({"input": text, "output": dict(block)})

    write_jsonl(output, rows)
    print(f"e2e NER types: {output} ({len(rows)} sentences)")


def load_ner_jsonl(path: str) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            content = normalize_text(row.get("input", row.get("content", "")))
            if content:
                out[content] = _unwrap_ner_output(row.get("output", {}))
    return out


def load_e2e_pred_map(path: str) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            content = normalize_text(row.get("content", ""))
            if content:
                out[content] = row.get("pred", {})
    return out


def extract_heads_from_e2e_pred(pred: dict) -> List[str]:
    heads: Set[str] = set()
    if not isinstance(pred, dict):
        return []
    for _bucket, items in pred.items():
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, dict):
                t = normalize_text(item.get("text", ""))
                if t:
                    heads.add(t)
    return sorted(heads)


def pool_from_ner_block(block: Dict[str, Any], ner_types: List[str]) -> Dict[str, Set[str]]:
    pool: Dict[str, Set[str]] = {t: set() for t in ner_types}
    for label, ents in block.items():
        norm = normalize_label(label)
        if norm not in pool or not isinstance(ents, list):
            continue
        for ent in ents:
            if isinstance(ent, dict):
                name = normalize_text(ent.get("text", ""))
                if name:
                    pool[norm].add(name)
    return pool


def heads_from_pool(pool: Dict[str, Set[str]], ner_types: List[str]) -> List[str]:
    s: Set[str] = set()
    for t in ner_types:
        s.update(pool.get(t) or ())
    return sorted(s)


def make_uie_re(cfg: AblationConfig, infer_batch: int = 32) -> UIEPredictor:
    device = os.environ.get("UIE_DEVICE", "gpu").strip().lower()
    if device not in ("cpu", "gpu"):
        device = "gpu"
    uie = UIEPredictor(
        model="uie-base-en",
        schema=[cfg.pred_bucket],
        task_path=resolve_task_path(cfg.re_model_dir),
        schema_lang=cfg.schema_lang,
        engine="pytorch",
        device=device,
        position_prob=cfg.position_prob,
        max_seq_len=512,
        batch_size=infer_batch,
        split_sentence=False,
        use_fp16=False,
    )
    if cfg.use_scierc_re_align and align_predictor_prompt_style_to_scierc_training:
        align_predictor_prompt_style_to_scierc_training(uie)
    return uie


def predict_one_sentence_zh(
    uie_re: UIEPredictor,
    content: str,
    heads: List[str],
    relations: List[str],
    pred_bucket: str,
    infer_batch: int,
) -> dict:
    examples = []
    meta: List[Tuple[str, str, int, int]] = []
    text_norm = normalize_text(content)
    text_enc = dbc2sbc(text_norm)

    for head in heads:
        hs, he = find_first_span(text_norm, head)
        if hs < 0:
            continue
        for rel in relations:
            prompt = f"{head}的{rel}"
            examples.append({"text": text_enc, "prompt": dbc2sbc(prompt)})
            meta.append((head, rel, hs, he))

    if not examples:
        return {}

    merged: Dict[Tuple[int, int, str], Dict[str, List[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for start in range(0, len(examples), infer_batch):
        chunk_ex = examples[start : start + infer_batch]
        chunk_meta = meta[start : start + infer_batch]
        chunk_res = uie_re._single_stage_predict(chunk_ex)
        for res, (head, rel, hs, he) in zip(chunk_res, chunk_meta):
            key = (hs, he, head)
            if not res:
                continue
            seen_tail: Set[tuple] = set()
            for t in res:
                if not isinstance(t, dict):
                    continue
                ts, te = t.get("start"), t.get("end")
                if not isinstance(ts, int) or not isinstance(te, int):
                    continue
                tt = (ts, te, normalize_text(t.get("text", "")))
                if tt in seen_tail:
                    continue
                seen_tail.add(tt)
                merged[key][rel].append(span_record(t))

    if not merged:
        return {}

    bucket_list = []
    for (hs, he, head), relmap in sorted(merged.items()):
        bucket_list.append(
            {
                "text": head,
                "start": hs,
                "end": he,
                "relations": {rel: tails for rel, tails in relmap.items()},
            }
        )
    return {pred_bucket: bucket_list}


def _label_for_head(head: str, pool: Dict[str, Set[str]], ner_types: List[str]) -> str:
    for t in ner_types:
        if head in (pool.get(t) or ()):
            return t
    return ner_types[0] if ner_types else "Generic"


def _norm_key(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def predict_citrus_from_pool(
    uie_re: UIEPredictor,
    content: str,
    pool: Dict[str, Set[str]],
    ner_types: List[str],
) -> dict:
    text_norm = normalize_text(content)
    pred: Dict[str, Dict[tuple, dict]] = defaultdict(dict)

    for head_type, relation, tail_type in CITRUS_PREDEFINED_TRIPLES:
        heads = pool.get(head_type) or set()
        if not heads:
            continue
        tail_pool = {_norm_key(x) for x in (pool.get(tail_type) or set())}
        for head in sorted(heads):
            hs, he = find_first_span(text_norm, head)
            if hs < 0:
                continue
            prompt = f"{head}的{relation}"
            try:
                res = uie_re._single_stage_predict(
                    [{"text": text_norm, "prompt": prompt}]
                )
            except Exception:
                res = []
            if not res or not res[0]:
                continue
            tails = []
            seen = set()
            for t in res[0]:
                if not isinstance(t, dict):
                    continue
                ts, te = t.get("start"), t.get("end")
                if not isinstance(ts, int) or not isinstance(te, int):
                    continue
                ttext = normalize_text(t.get("text", ""))
                if not ttext or (ts, te, ttext) in seen:
                    continue
                seen.add((ts, te, ttext))
                tlabel = tail_type
                if _norm_key(ttext) not in tail_pool and tail_pool:
                    for cand_type in ner_types:
                        if _norm_key(ttext) in {_norm_key(x) for x in pool.get(cand_type, ())}:
                            tlabel = cand_type
                            break
                tails.append(span_record(t, extra={"label": tlabel}))
            if not tails:
                continue

            hspan = (hs, he)
            existing = pred[head_type].get(hspan)
            if existing is None:
                pred[head_type][hspan] = {
                    "text": head,
                    "start": hs,
                    "end": he,
                    "relations": {relation: tails},
                }
            else:
                rels = existing.setdefault("relations", {})
                rels.setdefault(relation, []).extend(tails)

    out: Dict[str, list] = defaultdict(list)
    for htype, span_map in pred.items():
        for item in span_map.values():
            out[htype].append(item)
    return dict(out)


def pred_rows_to_simplified_triples(rows: List[dict]) -> List[dict]:
    """将 predict JSONL（content + pred）展平为 input + triples，保留尾实体 probability。"""
    simplified: List[dict] = []
    for row in rows:
        content = row.get("content", row.get("input", ""))
        pred = row.get("pred", {})
        if not isinstance(pred, dict):
            continue
        triples: List[dict] = []
        seen: Set[tuple] = set()
        for head_label, entities in pred.items():
            if not isinstance(entities, list):
                continue
            for ent in entities:
                if not isinstance(ent, dict):
                    continue
                head_text = ent.get("text", "")
                relations = ent.get("relations") or {}
                if not isinstance(relations, dict):
                    continue
                for relation, tails in relations.items():
                    if not isinstance(tails, list):
                        continue
                    for tail in tails:
                        if not isinstance(tail, dict):
                            continue
                        attr = tail.get("text", "")
                        attr_label = tail.get("label", "")
                        key = (head_text, head_label, relation, attr, attr_label)
                        if key in seen:
                            continue
                        seen.add(key)
                        triple: Dict[str, Any] = {
                            "entity": head_text,
                            "entity_label": head_label,
                            "relation": relation,
                            "attribute": attr,
                            "attribute_label": attr_label,
                            "probability": span_probability(tail),
                        }
                        triples.append(triple)
        if triples:
            simplified.append({"input": content, "triples": triples})
    return simplified


def export_simplified_triples(rows: List[dict], output_jsonl: str) -> None:
    simplified = pred_rows_to_simplified_triples(rows)
    write_jsonl(output_jsonl, simplified)
    print(f"simplified triples -> {output_jsonl} ({len(simplified)} sentences)")


def run_prompt_re(
    cfg: AblationConfig,
    output_jsonl: str,
    head_source: str,
    ner_jsonl: str = "",
    e2e_pred_file: str = "",
    simplified_jsonl: str = "",
) -> None:
    sentences = load_sentences(cfg.input_txt)
    ner_map = load_ner_jsonl(ner_jsonl) if ner_jsonl else {}
    e2e_map = load_e2e_pred_map(e2e_pred_file) if e2e_pred_file else {}
    uie_re = make_uie_re(cfg)
    pred_rows: List[dict] = []
    empty_heads = 0

    print(f"dataset={cfg.name} head_source={head_source} sentences={len(sentences)}")
    for idx, text in enumerate(sentences, 1):
        key = normalize_text(text)
        pool = {t: set() for t in cfg.ner_types}

        if head_source in ("oracle_ner", "pred_ner"):
            pool = pool_from_ner_block(ner_map.get(key, {}), cfg.ner_types)
        elif head_source == "e2e_pred":
            for _b, items in (e2e_map.get(key, {}) or {}).items():
                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, dict):
                            t = normalize_text(item.get("text", ""))
                            bt = normalize_label(_b)
                            if t and bt in pool:
                                pool[bt].add(t)

        if cfg.name == "citrus":
            if not any(pool.values()):
                empty_heads += 1
                pred = {}
            else:
                pred = predict_citrus_from_pool(uie_re, key, pool, cfg.ner_types)
        else:
            heads = heads_from_pool(pool, cfg.ner_types)
            if not heads:
                empty_heads += 1
                pred = {}
            else:
                pred = predict_one_sentence_zh(
                    uie_re, key, heads, cfg.relations, cfg.pred_bucket, 32
                )

        pred_rows.append({"content": text, "pred": pred})
        if idx % 16 == 0 or idx == len(sentences):
            print(f"  RE {idx}/{len(sentences)}")

    write_jsonl(output_jsonl, pred_rows)
    print(f"prompt RE -> {output_jsonl} (empty_heads={empty_heads})")

    if simplified_jsonl or cfg.name == "citrus":
        out_simplified = simplified_jsonl or os.path.join(
            os.path.dirname(os.path.abspath(output_jsonl)),
            "coupled_RE_simplified.jsonl",
        )
        export_simplified_triples(pred_rows, out_simplified)


def run_oracle_prompt(cfg: AblationConfig, output_jsonl: str) -> None:
    script = cfg.oracle_prompt_script
    if not os.path.isfile(script):
        raise FileNotFoundError(script)
    cmd = [
        sys.executable,
        script,
        "--input_txt",
        cfg.input_txt,
        "--model_dir",
        cfg.re_model_dir,
        "--output_jsonl",
        output_jsonl,
        "--gold_prompts_file",
        cfg.test_file,
        "--schema_lang",
        cfg.schema_lang,
        "--device",
        os.environ.get("UIE_DEVICE", "gpu"),
        "--position_prob",
        str(cfg.position_prob),
    ]
    subprocess.run(cmd, check=True)
