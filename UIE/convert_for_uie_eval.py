#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 PyTorch UIE 的金标 test.txt 与预测 JSONL 转为官方 UIE (eval_extraction.py) 所需格式。

金标 test.json：每行含 text / tokens / relation（token 级 offset）。
预测 test_preds_record.txt：每行含 relation.offset / relation.string（RelationScorer 格式）。
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

# CoNLL04 关系参数类型约束（与 conll04_test.json 一致）
RELATION_ARG_TYPES: Dict[str, Tuple[str, str]] = {
    "Kill": ("Peop", "Peop"),
    "Live_In": ("Peop", "Loc"),
    "Located_In": ("Loc", "Loc"),
    "OrgBased_In": ("Org", "Loc"),
    "Work_For": ("Peop", "Org"),
}

# citrus / NER+RE 预定义三元组模板 (头类型, 关系, 尾类型)
CITRUS_PREDEFINED_TRIPLES: List[Tuple[str, str, str]] = [
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

CITRUS_RELATION_TO_TYPES: Dict[str, List[Tuple[str, str]]] = {}
for _h, _r, _t in CITRUS_PREDEFINED_TRIPLES:
    CITRUS_RELATION_TO_TYPES.setdefault(_r, []).append((_h, _t))


def normalize_text(text: str) -> str:
    text = (text or "").strip()
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    return text


def token_char_spans(tokens: List[str]) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    pos = 0
    for i, tok in enumerate(tokens):
        start = pos
        end = start + len(tok)
        spans.append((start, end))
        if i != len(tokens) - 1:
            pos = end + 1
    return spans


def char_span_to_token_offset(
    tokens: List[str], char_start: int, char_end: int
) -> Optional[List[int]]:
    """字符 span -> 官方 token offset 列表（含起止 token 下标）。"""
    if char_start < 0 or char_end <= char_start:
        return None
    tok_spans = token_char_spans(tokens)
    indices: List[int] = []
    for i, (ts, te) in enumerate(tok_spans):
        if te <= char_start:
            continue
        if ts >= char_end:
            break
        indices.append(i)
    if not indices:
        return None
    if len(indices) == 1:
        return [indices[0]]
    return [indices[0], indices[-1]]


def span_text(tokens: List[str], offset: List[int]) -> str:
    if not offset:
        return ""
    if len(offset) == 1:
        i = offset[0]
        return tokens[i] if 0 <= i < len(tokens) else ""
    s, e = offset[0], offset[-1]
    if s < 0 or e >= len(tokens) or s > e:
        return ""
    return " ".join(tokens[s : e + 1])


def parse_prompt(prompt: str) -> Tuple[str, str]:
    prompt = normalize_text(prompt)
    if "的" not in prompt:
        return "", ""
    head, rel = prompt.rsplit("的", 1)
    return normalize_text(head), normalize_text(rel)


def build_entity_indexes(test_file: str) -> Dict[str, Dict[tuple, Set[str]]]:
    """content -> (char_start, char_end) -> entity types from NER prompt lines."""
    span_types: Dict[str, Dict[tuple, Set[str]]] = defaultdict(lambda: defaultdict(set))
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            content = normalize_text(row.get("content", ""))
            prompt = normalize_text(row.get("prompt", ""))
            if not content or not prompt or "的" in prompt:
                continue
            ent_type = prompt
            for item in row.get("result_list", []):
                if not isinstance(item, dict):
                    continue
                start, end = item.get("start"), item.get("end")
                if isinstance(start, int) and isinstance(end, int):
                    span_types[content][(start, end)].add(ent_type)
    return span_types


def choose_type(type_set: Set[str]) -> str:
    if not type_set:
        return "__UNK__"
    return normalize_entity_type(sorted(type_set)[0])


def normalize_entity_type(label: str) -> str:
    label = (label or "").strip()
    alias = {
        "by_product": "by-product",
        "by product": "by-product",
    }
    return alias.get(label, label)


def infer_tail_type_from_schema(rel_type: str, head_type: str) -> str:
    head_type = normalize_entity_type(head_type)
    for h, t in CITRUS_RELATION_TO_TYPES.get(rel_type, []):
        if normalize_entity_type(h) == head_type:
            return normalize_entity_type(t)
    return ""


def infer_head_type_from_schema(rel_type: str, tail_type: str) -> str:
    tail_type = normalize_entity_type(tail_type)
    for h, t in CITRUS_RELATION_TO_TYPES.get(rel_type, []):
        if normalize_entity_type(t) == tail_type:
            return normalize_entity_type(h)
    return ""


def resolve_entity_type(
    rel_type: str,
    arg_index: int,
    *,
    ner_types: Optional[Set[str]] = None,
    explicit_label: str = "",
    bucket_type: str = "",
    span_types: Optional[Set[str]] = None,
    partner_type: str = "",
) -> str:
    """按优先级解析实体类型：NER > 显式 label > 类型桶 > span 索引 > schema。"""
    if ner_types:
        chosen = choose_type(ner_types)
        if chosen != "__UNK__":
            return chosen
    if explicit_label:
        return normalize_entity_type(explicit_label)
    if bucket_type:
        return normalize_entity_type(bucket_type)
    if span_types:
        chosen = choose_type(span_types)
        if chosen != "__UNK__":
            return chosen
    if arg_index == 1 and partner_type:
        inferred = infer_tail_type_from_schema(rel_type, partner_type)
        if inferred:
            return inferred
    if arg_index == 0 and partner_type:
        inferred = infer_head_type_from_schema(rel_type, partner_type)
        if inferred:
            return inferred
    schema = RELATION_ARG_TYPES.get(rel_type)
    if schema and 0 <= arg_index < len(schema):
        return schema[arg_index]
    return "__UNK__"


def infer_entity_type(
    rel_type: str,
    arg_index: int,
    span_types: Dict[str, Dict[tuple, Set[str]]],
    content: str,
    char_start: int,
    char_end: int,
) -> str:
    """优先 NER 标注类型，否则按 CoNLL04 关系 schema 推断。"""
    ent_type = choose_type(span_types.get(content, {}).get((char_start, char_end), set()))
    if ent_type != "__UNK__":
        return ent_type
    schema = RELATION_ARG_TYPES.get(rel_type)
    if schema and 0 <= arg_index < len(schema):
        return schema[arg_index]
    return "__UNK__"


def token_entity_offset(start: int, end: int) -> List[int]:
    if start == end:
        return [start]
    return [start, end]


def build_gold_from_conll04_json(
    json_path: str,
    sentences: Optional[List[str]] = None,
) -> Dict[str, dict]:
    """从 conll04_test.json（tokens/entities/relations）构建官方金标记录。"""
    with open(json_path, encoding="utf-8") as f:
        records = json.load(f)

    by_text: Dict[str, dict] = {}
    for rec in records:
        tokens = rec.get("tokens") or []
        if not tokens:
            continue
        text = normalize_text(" ".join(tokens))
        entities = rec.get("entities") or []
        relations: List[dict] = []
        for rel in rec.get("relations") or []:
            head_i, tail_i = rel.get("head"), rel.get("tail")
            if head_i is None or tail_i is None:
                continue
            if head_i >= len(entities) or tail_i >= len(entities):
                continue
            head_ent, tail_ent = entities[head_i], entities[tail_i]
            h_off = token_entity_offset(head_ent["start"], head_ent["end"])
            t_off = token_entity_offset(tail_ent["start"], tail_ent["end"])
            relations.append(
                {
                    "type": rel["type"],
                    "args": [
                        {
                            "type": head_ent["type"],
                            "offset": h_off,
                            "text": span_text(tokens, h_off),
                        },
                        {
                            "type": tail_ent["type"],
                            "offset": t_off,
                            "text": span_text(tokens, t_off),
                        },
                    ],
                }
            )
        by_text[text] = {
            "text": text,
            "tokens": tokens,
            "entity": [],
            "relation": relations,
            "event": [],
        }

    if sentences is None:
        return by_text
    return {
        s: by_text.get(
            s,
            {"text": s, "tokens": s.split(), "entity": [], "relation": [], "event": []},
        )
        for s in sentences
    }


def load_sentence_order(input_txt: str) -> List[str]:
    sentences: List[str] = []
    with open(input_txt, "r", encoding="utf-8") as f:
        for line in f:
            text = normalize_text(line)
            if text:
                sentences.append(text)
    return sentences


def build_gold_relations(
    test_file: str, sentences: List[str]
) -> Dict[str, List[dict]]:
    """从 UIE test.txt 构建每句 gold relation 列表（官方 args 格式）。"""
    span_types = build_entity_indexes(test_file)
    by_content: Dict[str, List[dict]] = defaultdict(list)
    seen: Dict[str, Set[tuple]] = defaultdict(set)

    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            content = normalize_text(row.get("content", ""))
            prompt = normalize_text(row.get("prompt", ""))
            if not content or "的" not in prompt:
                continue
            head_text, rel_type = parse_prompt(prompt)
            if not head_text or not rel_type:
                continue
            tokens = content.split(" ")
            head_char_spans = [
                (content.find(head_text), content.find(head_text) + len(head_text))
            ] if head_text in content else []
            if not head_char_spans or head_char_spans[0][0] < 0:
                continue
            hs, he = head_char_spans[0]
            h_off = char_span_to_token_offset(tokens, hs, he)
            if not h_off:
                continue
            h_type = infer_entity_type(rel_type, 0, span_types, content, hs, he)

            for item in row.get("result_list", []):
                if not isinstance(item, dict):
                    continue
                ts, te = item.get("start"), item.get("end")
                if not isinstance(ts, int) or not isinstance(te, int):
                    continue
                t_off = char_span_to_token_offset(tokens, ts, te)
                if not t_off:
                    continue
                t_type = infer_entity_type(rel_type, 1, span_types, content, ts, te)
                key = (rel_type, tuple(h_off), tuple(t_off))
                if key in seen[content]:
                    continue
                seen[content].add(key)
                by_content[content].append(
                    {
                        "type": rel_type,
                        "args": [
                            {
                                "type": h_type,
                                "offset": h_off,
                                "text": span_text(tokens, h_off),
                            },
                            {
                                "type": t_type,
                                "offset": t_off,
                                "text": span_text(tokens, t_off),
                            },
                        ],
                    }
                )

    return {c: by_content.get(c, []) for c in sentences}


def build_gold_entities(
    test_file: str, sentences: List[str]
) -> Dict[str, List[dict]]:
    """从 UIE test.txt 的 NER 行构建每句 gold entity 列表（官方 args 格式）。"""
    by_content: Dict[str, List[dict]] = defaultdict(list)
    seen: Dict[str, Set[tuple]] = defaultdict(set)

    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            content = normalize_text(row.get("content", ""))
            prompt = normalize_text(row.get("prompt", ""))
            if not content or not prompt or "的" in prompt:
                continue
            ent_type = normalize_entity_type(prompt)
            tokens = content.split(" ")
            for item in row.get("result_list", []):
                if not isinstance(item, dict):
                    continue
                ts, te = item.get("start"), item.get("end")
                if not isinstance(ts, int) or not isinstance(te, int):
                    continue
                t_off = char_span_to_token_offset(tokens, ts, te)
                if not t_off:
                    continue
                key = (ent_type, tuple(t_off))
                if key in seen[content]:
                    continue
                seen[content].add(key)
                by_content[content].append(
                    {
                        "type": ent_type,
                        "offset": t_off,
                        "text": span_text(tokens, t_off),
                    }
                )

    return {c: by_content.get(c, []) for c in sentences}


def find_entity_char_span(
    content: str, text: str, used: Optional[Set[tuple]] = None
) -> Optional[Tuple[int, int]]:
    if not text:
        return None
    start = 0
    while start <= len(content):
        idx = content.find(text, start)
        if idx < 0:
            return None
        span = (idx, idx + len(text))
        if used and span in used:
            start = idx + 1
            continue
        return span
    return None


def extract_ner_pred_entities(content: str, output: Any) -> List[dict]:
    """从 NER 预测 output 块提取带 token offset 的实体列表。"""
    entities: List[dict] = []
    seen: Set[tuple] = set()
    used_char_spans: Set[tuple] = set()
    output_dict = _unwrap_ner_output(output)
    tokens = content.split(" ")

    for ent_type, items in output_dict.items():
        if not isinstance(items, list):
            continue
        norm_type = normalize_entity_type(ent_type)
        for item in items:
            if isinstance(item, str):
                text = normalize_text(item)
                char_span = find_entity_char_span(content, text, used_char_spans)
            elif isinstance(item, dict):
                text = normalize_text(item.get("text", ""))
                cs, ce = item.get("start"), item.get("end")
                if isinstance(cs, int) and isinstance(ce, int):
                    char_span = (cs, ce)
                else:
                    char_span = find_entity_char_span(content, text, used_char_spans)
            else:
                continue
            if not text or not char_span:
                continue
            used_char_spans.add(char_span)
            t_off = char_span_to_token_offset(tokens, char_span[0], char_span[1])
            if not t_off:
                continue
            key = (norm_type, tuple(t_off))
            if key in seen:
                continue
            seen.add(key)
            entities.append(
                {
                    "type": norm_type,
                    "offset": t_off,
                    "text": span_text(tokens, t_off),
                }
            )
    return entities


def entities_to_pred_record(entities: List[dict]) -> Tuple[List, List]:
    off: List = []
    string: List = []
    for ent in entities:
        ent_type = ent["type"]
        off.append([ent_type, ent["offset"]])
        string.append([ent_type, ent["text"]])
    return off, string


def build_pred_span_types(pred: dict) -> Dict[tuple, Set[str]]:
    span_types: Dict[tuple, Set[str]] = defaultdict(set)
    if not isinstance(pred, dict):
        return span_types
    for head_type, items in pred.items():
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            s, e = item.get("start"), item.get("end")
            if isinstance(s, int) and isinstance(e, int):
                span_types[(s, e)].add(head_type)
    return span_types


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


def load_ner_pred_types(ner_path: str) -> Dict[str, Dict[tuple, Set[str]]]:
    by_content: Dict[str, Dict[tuple, Set[str]]] = defaultdict(lambda: defaultdict(set))
    if not ner_path or not os.path.isfile(ner_path):
        return {}
    with open(ner_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            content = normalize_text(row.get("input", row.get("content", "")))
            output = _unwrap_ner_output(row.get("output", {}))
            if not content or not output:
                continue
            for ent_type, items in output.items():
                if not isinstance(items, list):
                    continue
                norm_type = normalize_entity_type(ent_type)
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    s, e = item.get("start"), item.get("end")
                    if isinstance(s, int) and isinstance(e, int):
                        by_content[content][(s, e)].add(norm_type)
    return dict(by_content)


def _merge_relation_dicts(a: dict, b: dict) -> dict:
    merged: Dict[str, list] = defaultdict(list)
    for src in (a, b):
        if not isinstance(src, dict):
            continue
        for rel, items in src.items():
            if isinstance(items, list):
                merged[rel].extend(items)
    return {k: v for k, v in merged.items()}


def _dedupe_tail_items(items: List[dict]) -> List[dict]:
    seen: Set[tuple] = set()
    out: List[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        s, e = item.get("start"), item.get("end")
        text = normalize_text(item.get("text", ""))
        label = normalize_entity_type(str(item.get("label", "")))
        key = (text, s, e, label)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def fill_pred_entity_types_from_schema(pred: dict) -> dict:
    """
    按 citrus schema 补全 two_stage pred 中的实体类型：
    - 规范化类型名（by_product -> by-product）
    - 为 relation 尾实体填充 label（显式 label 优先，否则 schema 推断）
    - 将尾实体补入对应类型桶，便于后续 span->type 索引
    """
    if not isinstance(pred, dict):
        return {}

    merged: Dict[str, Dict[tuple, dict]] = defaultdict(dict)

    for bucket_type, items in pred.items():
        if not isinstance(items, list):
            continue
        norm_bucket = normalize_entity_type(bucket_type)
        for item in items:
            if not isinstance(item, dict):
                continue
            hs, he = item.get("start"), item.get("end")
            if not isinstance(hs, int) or not isinstance(he, int):
                continue
            span = (hs, he)
            existing = merged[norm_bucket].get(span)
            if existing is None:
                merged[norm_bucket][span] = {
                    "text": item.get("text", ""),
                    "start": hs,
                    "end": he,
                    "relations": dict(item.get("relations") or {}),
                }
            else:
                existing["relations"] = _merge_relation_dicts(
                    existing.get("relations", {}),
                    item.get("relations", {}),
                )

    extra_heads: Dict[str, Dict[tuple, dict]] = defaultdict(dict)
    for head_type, span_map in merged.items():
        for head_item in span_map.values():
            relations = head_item.get("relations", {})
            if not isinstance(relations, dict):
                continue
            for rel_type, tails in relations.items():
                if not isinstance(tails, list):
                    continue
                filled_tails: List[dict] = []
                for tail in tails:
                    if not isinstance(tail, dict):
                        continue
                    ts, te = tail.get("start"), tail.get("end")
                    if not isinstance(ts, int) or not isinstance(te, int):
                        continue
                    explicit = normalize_entity_type(str(tail.get("label", "")))
                    schema_tail = infer_tail_type_from_schema(rel_type, head_type)
                    tail_type = explicit if explicit and explicit != "__UNK__" else schema_tail
                    if tail_type:
                        tail = dict(tail)
                        tail["label"] = tail_type
                        filled_tails.append(tail)
                        t_span = (ts, te)
                        if (
                            t_span not in merged.get(tail_type, {})
                            and t_span not in extra_heads[tail_type]
                        ):
                            extra_heads[tail_type][t_span] = {
                                "text": tail.get("text", ""),
                                "start": ts,
                                "end": te,
                                "relations": {},
                            }
                    else:
                        filled_tails.append(tail)
                relations[rel_type] = _dedupe_tail_items(filled_tails)

    for ent_type, span_map in extra_heads.items():
        merged[ent_type].update(span_map)

    result: Dict[str, list] = {}
    for ent_type, span_map in merged.items():
        result[ent_type] = list(span_map.values())
    return result


def extract_pred_relations(
    content: str,
    pred: dict,
    ner_span_types: Optional[Dict[tuple, Set[str]]] = None,
) -> Tuple[List[list], List[list]]:
    """返回 (offset_list, string_list) 供 RelationScorer.load_pred_list。"""
    tokens = content.split(" ")
    pred_span_types = build_pred_span_types(pred)
    use_ner = bool(ner_span_types)
    offset_rows: List[list] = []
    string_rows: List[list] = []
    seen: Set[tuple] = set()

    if not isinstance(pred, dict):
        return offset_rows, string_rows

    for head_type, head_items in pred.items():
        if not isinstance(head_items, list):
            continue
        for head_item in head_items:
            if not isinstance(head_item, dict):
                continue
            hs, he = head_item.get("start"), head_item.get("end")
            if not isinstance(hs, int) or not isinstance(he, int):
                continue
            h_off = char_span_to_token_offset(tokens, hs, he)
            if not h_off:
                continue
            h_type = resolve_entity_type(
                rel_type="",
                arg_index=0,
                ner_types=ner_span_types.get((hs, he)) if use_ner else None,
                explicit_label="",
                bucket_type=head_type,
                span_types=pred_span_types.get((hs, he)),
            )
            h_text = span_text(tokens, h_off)

            relations = head_item.get("relations", {})
            if not isinstance(relations, dict):
                continue
            for rel_type, tails in relations.items():
                if not isinstance(tails, list):
                    continue
                for tail in tails:
                    if not isinstance(tail, dict):
                        continue
                    ts, te = tail.get("start"), tail.get("end")
                    if not isinstance(ts, int) or not isinstance(te, int):
                        continue
                    t_off = char_span_to_token_offset(tokens, ts, te)
                    if not t_off:
                        continue
                    t_type = resolve_entity_type(
                        rel_type=rel_type,
                        arg_index=1,
                        ner_types=ner_span_types.get((ts, te)) if use_ner else None,
                        explicit_label=str(tail.get("label", "")),
                        span_types=pred_span_types.get((ts, te)),
                        partner_type=h_type,
                    )
                    if h_type == "__UNK__":
                        h_type = resolve_entity_type(
                            rel_type=rel_type,
                            arg_index=0,
                            ner_types=ner_span_types.get((hs, he)) if use_ner else None,
                            bucket_type=head_type,
                            span_types=pred_span_types.get((hs, he)),
                            partner_type=t_type,
                        )
                    t_text = span_text(tokens, t_off)
                    key = (rel_type, h_type, tuple(h_off), t_type, tuple(t_off))
                    if key in seen:
                        continue
                    seen.add(key)
                    offset_rows.append([rel_type, h_type, h_off, t_type, t_off])
                    string_rows.append([rel_type, h_type, h_text, t_type, t_text])

    return offset_rows, string_rows


def load_sentence_order_from_conll04(json_path: str) -> List[str]:
    with open(json_path, encoding="utf-8") as f:
        records = json.load(f)
    sentences: List[str] = []
    for rec in records:
        tokens = rec.get("tokens") or []
        if tokens:
            sentences.append(normalize_text(" ".join(tokens)))
    return sentences


def cmd_export_sentences(args: argparse.Namespace) -> None:
    sentences = load_sentence_order_from_conll04(args.conll04_json)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as out:
        for text in sentences:
            out.write(text + "\n")
    print(f"sentences written: {args.output} ({len(sentences)} lines)")


def load_sentence_order_from_pred(pred_file: str) -> List[str]:
    sentences: List[str] = []
    with open(pred_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            content = normalize_text(row.get("content", row.get("input", "")))
            if content:
                sentences.append(content)
    return sentences


def cmd_build_gold(args: argparse.Namespace) -> None:
    if getattr(args, "conll04_json", ""):
        sentences = load_sentence_order_from_conll04(args.conll04_json)
    elif getattr(args, "pred_file", ""):
        sentences = load_sentence_order_from_pred(args.pred_file)
    elif args.input_txt:
        sentences = load_sentence_order(args.input_txt)
    else:
        sentences = []
    if not sentences and not getattr(args, "conll04_json", ""):
        raise ValueError("empty sentence list: provide --input_txt, --pred_file, or --conll04_json")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as out:
        if getattr(args, "conll04_json", ""):
            gold_by_text = build_gold_from_conll04_json(args.conll04_json)
            if sentences:
                ordered = sentences
            else:
                ordered = list(gold_by_text.keys())
            matched = sum(1 for s in ordered if s in gold_by_text)
            for content in ordered:
                record = gold_by_text.get(
                    content,
                    {
                        "text": content,
                        "tokens": content.split(),
                        "entity": [],
                        "relation": [],
                        "event": [],
                    },
                )
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(
                f"gold written: {args.output} ({len(ordered)} sentences, "
                f"{matched} matched in conll04 json, "
                f"{sum(len(gold_by_text[s]['relation']) for s in ordered if s in gold_by_text)} relations)"
            )
            return

        gold_relations = build_gold_relations(args.test_file, sentences)
        for content in sentences:
            tokens = content.split(" ")
            record = {
                "text": content,
                "tokens": tokens,
                "entity": [],
                "relation": gold_relations.get(content, []),
                "event": [],
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"gold written: {args.output} ({len(sentences)} sentences)")


def cmd_build_gold_ner(args: argparse.Namespace) -> None:
    if args.input_txt:
        sentences = load_sentence_order(args.input_txt)
    elif getattr(args, "pred_file", ""):
        sentences = load_sentence_order_from_pred(args.pred_file)
    else:
        sentences = []
    if not sentences:
        raise ValueError("build-gold-ner 需要 --input_txt 或 --pred_file")

    gold_entities = build_gold_entities(args.test_file, sentences)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    ent_total = 0
    with open(args.output, "w", encoding="utf-8") as out:
        for content in sentences:
            tokens = content.split(" ")
            entities = gold_entities.get(content, [])
            ent_total += len(entities)
            record = {
                "text": content,
                "tokens": tokens,
                "entity": entities,
                "relation": [],
                "event": [],
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(
        f"gold (NER) written: {args.output} "
        f"({len(sentences)} sentences, {ent_total} entities)"
    )


def cmd_fill_pred_types(args: argparse.Namespace) -> None:
    rows: List[dict] = []
    with open(args.pred_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    filled = 0
    with open(args.output, "w", encoding="utf-8") as out:
        for row in rows:
            content = normalize_text(row.get("content", ""))
            pred = row.get("pred", {})
            new_pred = fill_pred_entity_types_from_schema(pred)
            if new_pred != pred:
                filled += 1
            out.write(
                json.dumps(
                    {"content": content, "pred": new_pred},
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"filled pred types: {args.output} ({filled}/{len(rows)} sentences updated)")


def cmd_build_pred(args: argparse.Namespace) -> None:
    gold_lines = [json.loads(l) for l in open(args.gold_file, encoding="utf-8") if l.strip()]
    gold_by_text = {normalize_text(r["text"]): r for r in gold_lines}
    ner_types = load_ner_pred_types(args.ner_pred_file or "")

    pred_map: Dict[str, dict] = {}
    with open(args.pred_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            content = normalize_text(row.get("content", ""))
            if content:
                pred_map[content] = row.get("pred", {})

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    missing = 0
    with open(args.output, "w", encoding="utf-8") as out:
        for gold in gold_lines:
            content = normalize_text(gold["text"])
            tokens = gold.get("tokens") or content.split(" ")
            off, string = extract_pred_relations(
                content,
                pred_map.get(content, {}),
                ner_span_types=ner_types.get(content),
            )
            record = {
                "text": content,
                "tokens": tokens,
                "entity": {"offset": [], "string": []},
                "relation": {"offset": off, "string": string},
                "event": {"offset": [], "string": []},
            }
            if content not in pred_map:
                missing += 1
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"pred record written: {args.output} (missing pred for {missing} sentences)")


def cmd_build_pred_ner(args: argparse.Namespace) -> None:
    gold_lines = [json.loads(l) for l in open(args.gold_file, encoding="utf-8") if l.strip()]
    pred_map: Dict[str, Any] = {}
    with open(args.pred_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            content = normalize_text(row.get("input", row.get("content", "")))
            if content:
                pred_map[content] = row.get("output", {})

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    missing = 0
    with open(args.output, "w", encoding="utf-8") as out:
        for gold in gold_lines:
            content = normalize_text(gold["text"])
            tokens = gold.get("tokens") or content.split(" ")
            entities = extract_ner_pred_entities(content, pred_map.get(content, {}))
            ent_off, ent_str = entities_to_pred_record(entities)
            record = {
                "text": content,
                "tokens": tokens,
                "entity": {"offset": ent_off, "string": ent_str},
                "relation": {"offset": [], "string": []},
                "event": {"offset": [], "string": []},
            }
            if content not in pred_map:
                missing += 1
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"pred record (NER) written: {args.output} (missing pred for {missing} sentences)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert UIE PyTorch IO for official eval_extraction.py")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_gold = sub.add_parser("build-gold", help="UIE test.txt / conll04_test.json -> official test.json")
    p_gold.add_argument("--test_file", default="")
    p_gold.add_argument("--input_txt", default="")
    p_gold.add_argument("--pred_file", default="", help="从预测 JSONL 读取评测句子顺序")
    p_gold.add_argument(
        "--conll04_json",
        default="",
        help="原始 CoNLL04 金标 JSON（tokens/entities/relations）",
    )
    p_gold.add_argument("--output", required=True)

    p_pred = sub.add_parser("build-pred", help="predict JSONL -> test_preds_record.txt")
    p_pred.add_argument("--pred_file", required=True)
    p_pred.add_argument("--gold_file", required=True)
    p_pred.add_argument("--output", required=True)
    p_pred.add_argument("--ner_pred_file", default="")

    p_gold_ner = sub.add_parser("build-gold-ner", help="UIE test.txt -> official NER test.json")
    p_gold_ner.add_argument("--test_file", required=True)
    p_gold_ner.add_argument("--input_txt", default="")
    p_gold_ner.add_argument("--pred_file", default="")
    p_gold_ner.add_argument("--output", required=True)

    p_pred_ner = sub.add_parser("build-pred-ner", help="NER predict JSONL -> test_preds_record.txt")
    p_pred_ner.add_argument("--pred_file", required=True)
    p_pred_ner.add_argument("--gold_file", required=True)
    p_pred_ner.add_argument("--output", required=True)

    p_fill = sub.add_parser(
        "fill-pred-types",
        help="按 citrus schema 补全 two_stage predict JSONL 的实体类型",
    )
    p_fill.add_argument("--pred_file", required=True)
    p_fill.add_argument("--output", default="")

    p_export = sub.add_parser("export-sentences", help="conll04_test.json -> sentences.txt")
    p_export.add_argument("--conll04_json", required=True)
    p_export.add_argument("--output", required=True)

    args = parser.parse_args()
    if args.cmd == "build-gold":
        if not args.conll04_json and not args.test_file:
            parser.error("build-gold 需要 --conll04_json 或 --test_file")
        cmd_build_gold(args)
    elif args.cmd == "build-pred":
        cmd_build_pred(args)
    elif args.cmd == "build-gold-ner":
        cmd_build_gold_ner(args)
    elif args.cmd == "build-pred-ner":
        cmd_build_pred_ner(args)
    elif args.cmd == "fill-pred-types":
        if not args.output:
            args.output = args.pred_file
        cmd_fill_pred_types(args)
    elif args.cmd == "export-sentences":
        cmd_export_sentences(args)


if __name__ == "__main__":
    main()
