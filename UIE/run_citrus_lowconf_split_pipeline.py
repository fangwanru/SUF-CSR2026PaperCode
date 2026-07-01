#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
低置信度句子 Qwen 复述拆分 + 两阶段 RE 重抽取 + UIE 官方评测（v5）。

流程:
  1. 筛选源句：含 probability < threshold 的低置信三元组（默认不含空 RE 句）
  2. 将源句 + 高置信三元组一并送入 Qwen 做复述拆分（API 阶段关键名词逐字校验）
  3. 拆分句 RE：使用原句 NER 池（实体须在拆分句中出现）
  4. 映射：优先原句 NER 对齐（v2 严格）；仅 RE 为空时启用 v3 宽松映射
  5. 合并：prob<0.5 丢弃；0.5~threshold 仅在与拆分结果冲突时丢弃；叠加 split_min_prob 过滤后的拆分 RE
  6. UIE offset-rel-strict 评测
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from openai import OpenAI

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SUF_ROOT = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from paths import (  # noqa: E402
    CITRUS_OUT,
    GOLD_UIE,
    NER_PRED,
    PRED_TWO_STAGE,
    QWEN_CONFIG,
    SIMPLIFIED_RE,
    SUF_NER_MODEL,
    SUF_RE_MODEL,
)

from ablation_config import get_config  # noqa: E402
from ablation_engine import (  # noqa: E402
    CITRUS_PREDEFINED_TRIPLES,
    export_simplified_triples,
    find_first_span,
    normalize_text,
    predict_citrus_from_pool,
    write_jsonl,
)
from uie1_predictor import UIEPredictor  # noqa: E402

DEFAULT_OUT = os.path.join(CITRUS_OUT, "suf_csr")
DEFAULT_SPLIT_MIN_PROB = 0.7
HARD_DROP_PROB = 0.5
UIE_ROOT = os.path.join(SCRIPT_DIR, "UIE")
CONVERT_SCRIPT = os.path.join(SCRIPT_DIR, "convert_for_uie_eval.py")
BASELINE_PRED = PRED_TWO_STAGE
BASELINE_SIMPLIFIED = SIMPLIFIED_RE
BASELINE_NER = NER_PRED
NER_MODEL = SUF_NER_MODEL
RE_MODEL = SUF_RE_MODEL

NER_SCHEMA = [
    "citrus",
    "compound",
    "aroma",
    "taste",
    "mouth_feel",
    "location",
    "by-product",
    "health",
]

SPLIT_SYSTEM = """You are a scientific sentence simplification assistant for citrus-domain relation extraction.

Task: Paraphrase and SPLIT a complex sentence into several simpler sentences.

STRICT rules:
1. Preserve ALL key nouns and named entities EXACTLY as they appear in the original (same spelling/casing).
2. Do NOT replace words with synonyms or hypernyms/hyponyms (e.g. do NOT change "nucleu" to "nucleus").
3. Only rephrase structure: split compound/clause-heavy sentences into simple SVO-style sentences.
4. Each output sentence should express ONE main fact when possible.
5. Do not add new facts or entities not supported by the original.
6. If verified relations are provided, your splits should preserve those facts and not contradict them.
7. Every key noun / named entity in each split MUST be copied verbatim from the original (identical spelling and casing).
8. Output ONLY the split sentences, one per line, each prefixed with "→ " (arrow + space)."""

# (head_type, tail_type) -> canonical relation
CANONICAL_REL_BY_PAIR: Dict[Tuple[str, str], str] = {
    (h, t): r for h, r, t in CITRUS_PREDEFINED_TRIPLES
}

# 拆分句谓语/语义线索 -> 预定义关系（用于替换 RE 误预测的 relation）
RELATION_SEMANTIC_PATTERNS: Dict[str, List[str]] = {
    "contain": [
        r"\bcontain",
        r"\bpresent in\b",
        r"\bfound in\b",
        r"\brich in\b",
        r"\babundant",
        r"\bdominates?\b",
        r"\bconcentrated in\b",
        r"\bidentified as\b",
        r"\bprimary\b.*\bcompound\b",
    ],
    "produce": [
        r"\bproduc",
        r"\byield",
        r"\bgenerat",
        r"\bderived from\b",
        r"\bessential oil\b",
        r"\bby-product\b",
        r"\bpeel\b",
        r"\bjuice\b",
    ],
    "produced_in": [
        r"\bgrown in\b",
        r"\bproduced in\b",
        r"\bnative to\b",
        r"\bregions of\b",
        r"\bsourced (?:mainly )?from\b",
        r"\borigin",
    ],
    "has_aroma": [
        r"\baroma\b",
        r"\bfragrant\b",
        r"\bscent\b",
        r"\bvolatile\b",
        r"\bodor\b",
        r"\bsmell\b",
    ],
    "has_taste": [
        r"\btaste\b",
        r"\bsour",
        r"\bbitter\b",
        r"\bsweet\b",
        r"\bflavou?r\b",
        r"\bsourness\b",
    ],
    "has_mouth_feel": [
        r"\btexture\b",
        r"\bmouth[- ]?feel\b",
        r"\bchewing\b",
        r"\bjuicy\b",
        r"\bastringent\b",
        r"\btender\b",
    ],
    "impact_on": [
        r"\beffect",
        r"\bactivity\b",
        r"\bantioxidant\b",
        r"\banti-",
        r"\bhealth\b",
        r"\bsedative\b",
        r"\bhypnotic\b",
        r"\bdecreased\b",
        r"\bincreased\b",
        r"\bstimulation\b",
    ],
    "category": [
        r"\bcategory\b",
        r"\bspecies of\b",
        r"\bhybrid",
        r"\bdeveloped from\b",
        r"\bparent",
        r"\bcultivar\b",
        r"\bCitrus [A-Za-z]+\b",
        r"\bis one of\b",
        r"\bbelongs to\b",
    ],
}


@dataclass
class SentenceBundle:
    source_text: str
    low_triples: List[dict] = field(default_factory=list)
    high_triples: List[dict] = field(default_factory=list)
    reason: str = "low_conf"  # low_conf | empty_pred


@dataclass
class SplitRecord:
    source_id: int
    source_text: str
    split_index: int
    split_text: str


def load_qwen_client(api_key: str, base_url: str, model: str) -> Tuple[OpenAI, str]:
    if not api_key:
        raise ValueError("缺少 API key，请设置 DASHSCOPE_API_KEY 或 --api-key")
    return OpenAI(api_key=api_key, base_url=base_url), model


def load_qwen_defaults(config_path: str) -> Tuple[str, str, str]:
    text = Path(config_path).read_text(encoding="utf-8")

    def ext(key: str) -> str:
        m = re.search(rf'{key}\s*=\s*"([^"]+)"', text)
        if not m:
            raise ValueError(f"config 缺少 {key}: {config_path}")
        return m.group(1)

    return ext("BASE_URL"), ext("MODEL_NAME"), ext("DASHSCOPE_API_KEY")


def baseline_pred_is_empty(pred: dict) -> bool:
    """baseline 两阶段 RE 无任何 relation tail。"""
    if not pred:
        return True
    for items in pred.values():
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, dict) and item.get("relations"):
                return False
    return True


def load_simplified_map(simplified_path: str) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {}
    if not os.path.isfile(simplified_path):
        return out
    with open(simplified_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = row.get("input", "")
            if text:
                out[text] = row.get("triples") or []
    return out


def select_sentence_bundles(
    simplified_path: str,
    baseline_pred_path: str,
    threshold: float,
    include_empty_pred: bool = True,
) -> List[SentenceBundle]:
    """选出需 Qwen 拆分的源句：低置信三元组 和/或 baseline RE 为空。"""
    triples_map = load_simplified_map(simplified_path)
    bundles: List[SentenceBundle] = []
    seen: Set[str] = set()

    with open(baseline_pred_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = row.get("content", "")
            if not text or text in seen:
                continue
            pred = row.get("pred") or {}
            triples = triples_map.get(text, [])
            low = [
                t
                for t in triples
                if float(t.get("probability", 1.0) or 1.0) < threshold
            ]
            high = [
                t
                for t in triples
                if float(t.get("probability", 1.0) or 1.0) >= threshold
            ]
            if low:
                bundles.append(
                    SentenceBundle(
                        source_text=text,
                        low_triples=low,
                        high_triples=high,
                        reason="low_conf",
                    )
                )
                seen.add(text)
            elif include_empty_pred and baseline_pred_is_empty(pred):
                bundles.append(
                    SentenceBundle(
                        source_text=text,
                        low_triples=[],
                        high_triples=[],
                        reason="empty_pred",
                    )
                )
                seen.add(text)
    return bundles


def format_triples_for_prompt(triples: List[dict]) -> str:
    if not triples:
        return "(none)"
    lines = []
    for t in triples:
        prob = float(t.get("probability", 1.0) or 1.0)
        lines.append(
            f"- [{t.get('entity_label')}] {t.get('entity')} "
            f"--{t.get('relation')}--> [{t.get('attribute_label')}] {t.get('attribute')} "
            f"(confidence={prob:.3f})"
        )
    return "\n".join(lines)


def split_tokens(text: str) -> List[str]:
    return [t for t in re.findall(r"[\w\-]+(?:'[\w]+)?", text or "") if len(t) >= 3]


def collect_key_nouns(
    bundle: SentenceBundle,
    parent_ner: Optional[dict] = None,
) -> Set[str]:
    """从三元组与源句 NER 收集关键名词（须在拆分句中原样出现）。"""
    nouns: Set[str] = set()
    for triple in bundle.low_triples + bundle.high_triples:
        for field in ("entity", "attribute"):
            value = (triple.get(field) or "").strip()
            if value:
                nouns.add(value)
    if parent_ner:
        for label, ents in parent_ner.items():
            if label not in NER_SCHEMA or not isinstance(ents, list):
                continue
            for ent in ents:
                if isinstance(ent, dict):
                    value = (ent.get("text") or "").strip()
                    if value:
                        nouns.add(value)
    return nouns


def validate_split_in_parent(
    split_text: str,
    parent_text: str,
    key_nouns: Optional[Set[str]] = None,
) -> bool:
    """拆分句中出现的每个关键名词须与原句逐字一致（一模一样）。"""
    if not split_text or len(split_text.strip()) < 5:
        return False
    nouns = key_nouns or set()
    if not nouns:
        return True
    for noun in sorted(nouns, key=len, reverse=True):
        if len(noun) < 2:
            continue
        pos = 0
        while pos < len(split_text):
            idx = split_text.find(noun, pos)
            if idx < 0:
                lower_idx = split_text.lower().find(noun.lower(), pos)
                if lower_idx >= 0:
                    fragment = split_text[lower_idx : lower_idx + len(noun)]
                    if fragment.lower() == noun.lower() and fragment != noun:
                        return False
                break
            if parent_text.find(noun) < 0:
                return False
            pos = idx + len(noun)
    return True


def parse_split_response(content: str, source_text: str) -> List[str]:
    lines: List[str] = []
    for raw in content.splitlines():
        s = raw.strip()
        if not s:
            continue
        s = re.sub(r"^[\d]+[\.\)、]\s*", "", s)
        s = re.sub(r"^[→\-–—*]\s*", "", s)
        s = s.strip().strip('"')
        if len(s) < 5:
            continue
        if s.lower() == source_text.strip().lower():
            continue
        lines.append(s)
    seen: Set[str] = set()
    out: List[str] = []
    for s in lines:
        key = normalize_text(s)
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def call_qwen_split(
    client: OpenAI,
    model: str,
    sentence: str,
    high_triples: List[dict],
    key_nouns: Set[str],
    max_retries: int = 3,
) -> List[str]:
    verified = format_triples_for_prompt(high_triples)
    user_msg = (
        f"Original sentence:\n{sentence}\n\n"
        f"Verified relations to preserve (do not contradict):\n{verified}\n\n"
        "Split the original sentence following all rules."
    )
    messages = [
        {"role": "system", "content": SPLIT_SYSTEM},
        {"role": "user", "content": user_msg},
    ]
    last_err: Optional[Exception] = None
    last_content = ""
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2,
            )
            content = resp.choices[0].message.content or ""
            last_content = content
            splits = parse_split_response(content, sentence)
            validated = [
                s for s in splits if validate_split_in_parent(s, sentence, key_nouns)
            ]
            if validated:
                return validated
            if splits and attempt == max_retries - 1:
                print(f"  警告: 拆分句均未通过子串校验，丢弃 {len(splits)} 条")
                return []
        except Exception as exc:
            last_err = exc
            time.sleep(1.5 * (attempt + 1))
    if last_err is not None:
        raise RuntimeError(f"Qwen split API 失败: {last_err}")
    preview = (last_content or "").replace("\n", "\\n")[:200]
    print(f"  警告: Qwen 未返回有效拆分句 (raw={preview!r})")
    return []


def run_qwen_splits(
    client: OpenAI,
    model: str,
    bundles: List[SentenceBundle],
    out_path: str,
    cache_path: str,
    refresh_cache: bool,
    parent_ner_map: Dict[str, dict],
) -> List[SplitRecord]:
    cache: Dict[str, List[str]] = {}
    if not refresh_cache and os.path.isfile(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    src = row["source_text"]
                    bundle = next((b for b in bundles if b.source_text == src), None)
                    if bundle is None:
                        continue
                    key_nouns = collect_key_nouns(bundle, parent_ner_map.get(src))
                    splits = row.get("splits") or []
                    cache[src] = [
                        s
                        for s in splits
                        if validate_split_in_parent(s, src, key_nouns)
                    ]

    records: List[SplitRecord] = []
    rejected = 0
    for sid, bundle in enumerate(bundles):
        src = bundle.source_text
        key_nouns = collect_key_nouns(bundle, parent_ner_map.get(src))
        splits = cache.get(src)
        if splits is None:
            print(f"  Qwen split [{sid + 1}/{len(bundles)}]: {src[:80]}...")
            if bundle.high_triples:
                print(f"    保留 {len(bundle.high_triples)} 条高置信三元组作为输入提示")
            raw_splits = call_qwen_split(
                client, model, src, bundle.high_triples, key_nouns
            )
            splits = [
                s for s in raw_splits if validate_split_in_parent(s, src, key_nouns)
            ]
            rejected += max(0, len(raw_splits) - len(splits))
            cache[src] = splits
            with open(cache_path, "a", encoding="utf-8") as cf:
                cf.write(
                    json.dumps(
                        {
                            "source_text": src,
                            "high_triples": bundle.high_triples,
                            "splits": splits,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            time.sleep(0.3)
        for idx, st in enumerate(splits):
            records.append(SplitRecord(sid, src, idx, st))

    write_jsonl(
        out_path,
        [
            {
                "source_id": r.source_id,
                "source_text": r.source_text,
                "split_index": r.split_index,
                "split_text": r.split_text,
                "substring_validated": True,
            }
            for r in records
        ],
    )
    print(
        f"拆分句共 {len(records)} 条（来自 {len(bundles)} 源句，子串校验拒绝 {rejected} 条）-> {out_path}"
    )
    return records


def run_ner_on_texts(texts: List[str], model_dir: str, device: str) -> Dict[str, dict]:
    uie = UIEPredictor(
        model="uie-base-en",
        task_path=model_dir,
        schema=NER_SCHEMA,
        schema_lang="en",
        max_seq_len=512,
        device=device,
        position_prob=0.5,
    )
    out: Dict[str, dict] = {}
    for i, text in enumerate(texts, 1):
        ner = uie(text)
        block: Dict[str, list] = defaultdict(list)
        if ner and isinstance(ner[0], dict):
            for label, ents in ner[0].items():
                if isinstance(ents, list):
                    block[label].extend(ents)
        out[text] = dict(block)
        if i % 20 == 0 or i == len(texts):
            print(f"    NER {i}/{len(texts)}")
    return out


def pool_from_ner_block(block: dict, ner_types: List[str]) -> Dict[str, Set[str]]:
    pool = {t: set() for t in ner_types}
    for label, ents in (block or {}).items():
        if label not in pool or not isinstance(ents, list):
            continue
        for ent in ents:
            if isinstance(ent, dict):
                t = (ent.get("text") or "").strip()
                if t:
                    pool[label].add(t)
    return pool


def _token_overlap(a: str, b: str) -> bool:
    ta = {t.lower() for t in split_tokens(a)}
    tb = {t.lower() for t in split_tokens(b)}
    return bool(ta & tb)


FILLER_WORDS = frozenset(
    {"flesh", "its", "the", "a", "an", "is", "has", "was", "were", "with", "or"}
)


def _attribute_span_candidates(text: str) -> List[str]:
    """从 tail 文本生成候选 span，优先非 filler 片段。"""
    text = (text or "").strip()
    if not text:
        return []
    tokens = re.findall(r"[\w\-]+(?:'[\w]+)?", text)
    content = [t for t in tokens if t.lower() not in FILLER_WORDS]
    candidates: List[str] = []
    seen: Set[str] = set()

    def add(s: str) -> None:
        s = s.strip()
        if len(s) >= 2 and s not in seen:
            seen.add(s)
            candidates.append(s)

    if content:
        add(" ".join(content))
        for tok in sorted(content, key=len, reverse=True):
            add(tok)
    add(text)
    for length in range(len(text), 2, -1):
        for start in range(0, len(text) - length + 1):
            add(text[start : start + length])
    return candidates


def pool_for_split_from_parent(
    parent_ner: dict,
    split_text: str,
    ner_types: List[str],
) -> Dict[str, Set[str]]:
    """原句 NER 池中，仅保留在拆分句中出现的实体。"""
    pool = pool_from_ner_block(parent_ner, ner_types)
    st = normalize_text(split_text)
    out: Dict[str, Set[str]] = {t: set() for t in ner_types}
    for label, ents in pool.items():
        for ent in ents:
            if ent in st:
                out[label].add(ent)
    return out


def pred_has_relations(pred: dict) -> bool:
    for items in (pred or {}).values():
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, dict) and item.get("relations"):
                return True
    return False


def map_span_strict(parent: str, text: str) -> Tuple[int, int]:
    if not text or not parent:
        return -1, -1
    return find_first_span(parent, text)


def map_tail_via_parent_ner(
    parent_text: str,
    split_tail_text: str,
    tail_label: str,
    parent_ner: Optional[dict],
    split_text: str,
) -> Tuple[int, int]:
    """尾实体优先对齐原句 baseline NER span。"""
    if parent_ner and tail_label:
        best: Optional[Tuple[int, int, int]] = None
        for ent in parent_ner.get(tail_label, []) or []:
            if not isinstance(ent, dict):
                continue
            et = (ent.get("text") or "").strip()
            hs, he = ent.get("start"), ent.get("end")
            if not et or not isinstance(hs, int) or not isinstance(he, int):
                continue
            if et not in split_text:
                continue
            if (
                et in split_tail_text
                or split_tail_text in et
                or _token_overlap(et, split_tail_text)
            ):
                score = len(et) + (200 if et in split_tail_text else 0)
                if best is None or score > best[0]:
                    best = (score, hs, he)
        if best:
            return best[1], best[2]
    return map_span_strict(parent_text, split_tail_text)


def map_head_via_parent_ner(
    parent_text: str,
    head_text: str,
    head_label: str,
    parent_ner: Optional[dict],
    split_text: str,
) -> Tuple[int, int]:
    s, e = map_span_strict(parent_text, head_text)
    if s >= 0:
        return s, e
    if parent_ner and head_label:
        for ent in parent_ner.get(head_label, []) or []:
            if not isinstance(ent, dict):
                continue
            et = (ent.get("text") or "").strip()
            hs, he = ent.get("start"), ent.get("end")
            if et and et in split_text and isinstance(hs, int) and isinstance(he, int):
                return hs, he
    return -1, -1


def merge_entity_item(
    parent_text: str,
    item: dict,
    parent_ner: Optional[dict] = None,
    split_text: str = "",
    relaxed: bool = False,
) -> Optional[dict]:
    head_label = item.get("_bucket", "")
    hs, he = map_head_via_parent_ner(
        parent_text, item.get("text", ""), head_label, parent_ner, split_text
    )
    if hs < 0:
        return None
    merged_rels: Dict[str, list] = {}
    for rel, tails in (item.get("relations") or {}).items():
        if not isinstance(tails, list):
            continue
        kept = []
        for tail in tails:
            if not isinstance(tail, dict):
                continue
            tail_label = tail.get("label") or ""
            if relaxed:
                ts, te = map_span_to_parent_relaxed(
                    parent_text,
                    tail.get("text", ""),
                    parent_ner,
                    tail_label,
                    split_text,
                )
            else:
                ts, te = map_tail_via_parent_ner(
                    parent_text,
                    tail.get("text", ""),
                    tail_label,
                    parent_ner,
                    split_text,
                )
            if ts < 0:
                continue
            nt = dict(tail)
            nt["start"], nt["end"] = ts, te
            nt["text"] = parent_text[ts:te]
            kept.append(nt)
        if kept:
            merged_rels[rel] = kept
    if not merged_rels:
        return None
    out = dict(item)
    out["text"] = parent_text[hs:he]
    out["start"], out["end"] = hs, he
    out.pop("_bucket", None)
    out["relations"] = merged_rels
    return out


def map_pred_to_parent(
    pred: dict,
    parent_text: str,
    parent_ner: Optional[dict],
    split_text: str,
    relaxed: bool = False,
) -> dict:
    out: Dict[str, list] = {}
    for bucket, items in (pred or {}).items():
        if not isinstance(items, list):
            continue
        kept_items = []
        for item in items:
            if not isinstance(item, dict):
                continue
            tagged = dict(item)
            tagged["_bucket"] = bucket
            mapped = merge_entity_item(
                parent_text,
                tagged,
                parent_ner,
                split_text,
                relaxed=relaxed,
            )
            if mapped and mapped.get("relations"):
                kept_items.append(mapped)
        if kept_items:
            out[bucket] = kept_items
    return out


def filter_pred_spans_in_parent(
    pred: dict,
    parent_text: str,
    parent_ner: Optional[dict] = None,
    split_text: str = "",
    relaxed: bool = False,
) -> dict:
    return map_pred_to_parent(pred, parent_text, parent_ner, split_text, relaxed=relaxed)


def run_re_on_splits(
    split_records: List[SplitRecord],
    re_model_dir: str,
    device: str,
    parent_ner_map: Dict[str, dict],
) -> List[dict]:
    cfg = get_config("citrus")
    uie_re = UIEPredictor(
        model="uie-base-en",
        schema=[cfg.pred_bucket],
        task_path=re_model_dir,
        schema_lang=cfg.schema_lang,
        device=device,
        position_prob=cfg.position_prob,
        max_seq_len=512,
        batch_size=1,
        split_sentence=False,
    )
    rows: List[dict] = []
    strict_hits = relaxed_hits = 0
    for i, rec in enumerate(split_records, 1):
        parent_ner = parent_ner_map.get(rec.source_text, {})
        pool = pool_for_split_from_parent(
            parent_ner, rec.split_text, cfg.ner_types
        )
        if not any(pool.values()):
            pred = {}
        else:
            raw = predict_citrus_from_pool(
                uie_re, normalize_text(rec.split_text), pool, cfg.ner_types
            )
            raw = canonicalize_pred_relations(raw)
            pred = filter_pred_spans_in_parent(
                raw,
                rec.source_text,
                parent_ner,
                rec.split_text,
                relaxed=False,
            )
            if pred_has_relations(pred):
                strict_hits += 1
            else:
                pred = filter_pred_spans_in_parent(
                    raw,
                    rec.source_text,
                    parent_ner,
                    rec.split_text,
                    relaxed=True,
                )
                if pred_has_relations(pred):
                    relaxed_hits += 1
        rows.append(
            {
                "source_id": rec.source_id,
                "source_text": rec.source_text,
                "split_index": rec.split_index,
                "split_text": rec.split_text,
                "pred": pred,
            }
        )
        if i % 20 == 0 or i == len(split_records):
            print(f"    RE {i}/{len(split_records)}")
    print(f"    映射: v2严格={strict_hits} v3宽松fallback={relaxed_hits}")
    return rows


def map_span_to_parent_relaxed(
    parent: str,
    text: str,
    parent_ner: Optional[dict] = None,
    prefer_label: str = "",
    split_text: str = "",
) -> Tuple[int, int]:
    """v3 宽松 fallback：原句 NER 对齐 → 属性词子串。"""
    ts, te = map_tail_via_parent_ner(
        parent, text, prefer_label, parent_ner, split_text or text
    )
    if ts >= 0:
        return ts, te
    if not text or not parent:
        return -1, -1
    for candidate in _attribute_span_candidates(text):
        s, e = find_first_span(parent, candidate)
        if s >= 0:
            return s, e
        j = parent.lower().find(candidate.lower())
        if j >= 0:
            return j, j + len(candidate)
    return -1, -1


def map_span_to_parent(
    parent: str,
    text: str,
    parent_ner: Optional[dict] = None,
    prefer_label: str = "",
) -> Tuple[int, int]:
    return map_span_strict(parent, text)


def sentence_supports_relation(text: str, relation: str) -> bool:
    patterns = RELATION_SEMANTIC_PATTERNS.get(relation, [])
    if not patterns:
        return False
    lower = text.lower()
    return any(re.search(p, lower) for p in patterns)


def _tail_key(tail: dict) -> tuple:
    return (tail.get("start"), tail.get("end"), tail.get("text", ""))


def canonicalize_pred_relations(pred: dict) -> dict:
    """同一 (head_type, tail_type) 的预测归并到预定义 canonical relation。"""
    out: Dict[str, list] = {}
    for head_type, items in (pred or {}).items():
        if not isinstance(items, list):
            continue
        new_items: List[dict] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            rels: Dict[str, list] = {}
            for rel, tails in (item.get("relations") or {}).items():
                if not isinstance(tails, list):
                    continue
                for tail in tails:
                    if not isinstance(tail, dict):
                        continue
                    tail_type = tail.get("label") or ""
                    canon = CANONICAL_REL_BY_PAIR.get((head_type, tail_type))
                    target_rel = canon or rel
                    rels.setdefault(target_rel, [])
                    seen = {_tail_key(t) for t in rels[target_rel]}
                    tk = _tail_key(tail)
                    if tk not in seen:
                        rels[target_rel].append(dict(tail))
            if rels:
                ni = dict(item)
                ni["relations"] = rels
                new_items.append(ni)
        if new_items:
            out[head_type] = new_items
    return out


def apply_semantic_relation_replacement(pred: dict, split_text: str) -> dict:
    """将 RE 预测的关系归一化为预定义三元组（不做 span 过滤）。"""
    return canonicalize_pred_relations(pred)


def _relation_slot_key(item: dict, rel: str) -> tuple:
    return (item.get("start"), item.get("end"), item.get("text", ""), rel)


def _parent_tail_key(tail: dict) -> tuple:
    return (tail.get("start"), tail.get("end"), tail.get("text", ""))


def _index_split_relations(split_part: dict) -> Dict[tuple, Set[tuple]]:
    slots: Dict[tuple, Set[tuple]] = defaultdict(set)
    for items in (split_part or {}).values():
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            for rel, tails in (item.get("relations") or {}).items():
                slot = _relation_slot_key(item, rel)
                for tail in tails or []:
                    if isinstance(tail, dict):
                        slots[slot].add(_parent_tail_key(tail))
    return slots


def filter_baseline_for_merge(
    baseline_pred: dict,
    split_part: dict,
    prob_threshold: float,
    hard_drop: float = HARD_DROP_PROB,
) -> dict:
    """保留 baseline；prob<hard_drop 丢弃；prob<threshold 仅在与拆分冲突时丢弃。"""
    split_slots = _index_split_relations(split_part)
    out: Dict[str, list] = {}
    for bucket, items in (baseline_pred or {}).items():
        if not isinstance(items, list):
            continue
        kept_items = []
        for item in items:
            if not isinstance(item, dict):
                continue
            rels_out: Dict[str, list] = {}
            for rel, tails in (item.get("relations") or {}).items():
                if not isinstance(tails, list):
                    continue
                slot = _relation_slot_key(item, rel)
                kept_tails = []
                for tail in tails:
                    if not isinstance(tail, dict):
                        continue
                    prob = float(tail.get("probability", 1.0) or 1.0)
                    if prob < hard_drop:
                        continue
                    if prob < prob_threshold and slot in split_slots and split_slots[slot]:
                        if _parent_tail_key(tail) not in split_slots[slot]:
                            continue
                    kept_tails.append(tail)
                if kept_tails:
                    rels_out[rel] = kept_tails
            if rels_out:
                ni = dict(item)
                ni["relations"] = rels_out
                kept_items.append(ni)
        if kept_items:
            out[bucket] = kept_items
    return out


def merge_preds_to_parent(split_rows: List[dict]) -> dict:
    """合并已映射到原句坐标的拆分 RE 预测。"""
    merged: Dict[str, Dict[tuple, dict]] = defaultdict(dict)
    for row in split_rows:
        pred = row.get("pred") or {}
        if not isinstance(pred, dict):
            continue
        for bucket, items in pred.items():
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                hs, he = item.get("start"), item.get("end")
                if not isinstance(hs, int) or not isinstance(he, int):
                    continue
                key = (hs, he, item.get("text", ""))
                existing = merged[bucket].get(key)
                if existing is None:
                    merged[bucket][key] = dict(item)
                else:
                    for rel, tails in (item.get("relations") or {}).items():
                        existing.setdefault("relations", {}).setdefault(rel, [])
                        seen = {
                            _parent_tail_key(t)
                            for t in existing["relations"][rel]
                            if isinstance(t, dict)
                        }
                        for t in tails or []:
                            if isinstance(t, dict):
                                tk = _parent_tail_key(t)
                                if tk not in seen:
                                    existing["relations"][rel].append(dict(t))
                                    seen.add(tk)
    return {k: list(v.values()) for k, v in merged.items()}


def filter_pred_keep_confidence(pred: dict, min_prob: float) -> dict:
    """仅保留 relation tail probability >= min_prob 的部分。"""
    out: Dict[str, list] = {}
    for bucket, items in (pred or {}).items():
        if not isinstance(items, list):
            continue
        kept_items = []
        for item in items:
            if not isinstance(item, dict):
                continue
            rels_out: Dict[str, list] = {}
            for rel, tails in (item.get("relations") or {}).items():
                if not isinstance(tails, list):
                    continue
                kept_tails = []
                for tail in tails:
                    if not isinstance(tail, dict):
                        continue
                    prob = float(tail.get("probability", 1.0) or 1.0)
                    if prob >= min_prob:
                        kept_tails.append(tail)
                if kept_tails:
                    rels_out[rel] = kept_tails
            if rels_out:
                ni = dict(item)
                ni["relations"] = rels_out
                kept_items.append(ni)
        if kept_items:
            out[bucket] = kept_items
    return out


def merge_pred_dicts(base: dict, extra: dict) -> dict:
    """合并两个 pred 结构，按 span+relation+tail 去重。"""
    merged: Dict[str, Dict[tuple, dict]] = defaultdict(dict)
    for pred in (base, extra):
        for bucket, items in (pred or {}).items():
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                hs, he = item.get("start"), item.get("end")
                if not isinstance(hs, int) or not isinstance(he, int):
                    continue
                key = (hs, he, item.get("text", ""))
                existing = merged[bucket].get(key)
                if existing is None:
                    merged[bucket][key] = {
                        "text": item.get("text", ""),
                        "start": hs,
                        "end": he,
                        "relations": {},
                    }
                    existing = merged[bucket][key]
                for rel, tails in (item.get("relations") or {}).items():
                    if not isinstance(tails, list):
                        continue
                    existing.setdefault("relations", {}).setdefault(rel, [])
                    seen = {
                        (t.get("start"), t.get("end"), t.get("text"))
                        for t in existing["relations"][rel]
                        if isinstance(t, dict)
                    }
                    for t in tails:
                        if isinstance(t, dict):
                            tk = (t.get("start"), t.get("end"), t.get("text"))
                            if tk not in seen:
                                existing["relations"][rel].append(dict(t))
                                seen.add(tk)
    return {k: list(v.values()) for k, v in merged.items()}


def load_baseline_pred_map(path: str) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                content = row.get("content", "")
                if content:
                    out[content] = row.get("pred") or {}
    return out


def load_baseline_ner_map(path: str) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            text = row.get("input", "")
            output = row.get("output") or []
            if text and output and isinstance(output[0], dict):
                out[text] = output[0]
    return out


def build_merged_corpus(
    baseline_pred_path: str,
    low_conf_sources: Set[str],
    parent_preds: Dict[str, dict],
    out_path: str,
) -> None:
    rows: List[dict] = []
    with open(baseline_pred_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            content = row.get("content", "")
            if content in low_conf_sources and content in parent_preds:
                rows.append({"content": content, "pred": parent_preds[content]})
            else:
                rows.append(row)
    write_jsonl(out_path, rows)
    print(f"合并预测 -> {out_path} ({len(rows)} 句)")


def run_uie_eval(
    pred_jsonl: str,
    ner_jsonl: str,
    out_dir: str,
    gold_file: str = GOLD_UIE,
) -> Dict[str, float]:
    os.makedirs(out_dir, exist_ok=True)
    pred_record = os.path.join(out_dir, "test_preds_record.txt")
    results_file = os.path.join(out_dir, "test_results.txt")

    subprocess.run(
        [
            sys.executable,
            CONVERT_SCRIPT,
            "build-pred",
            "--pred_file",
            pred_jsonl,
            "--gold_file",
            gold_file,
            "--output",
            pred_record,
            "--ner_pred_file",
            ner_jsonl,
        ],
        check=True,
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{UIE_ROOT}:{env.get('PYTHONPATH', '')}"
    gold_dir = os.path.dirname(gold_file)
    subprocess.run(
        [
            sys.executable,
            "scripts/eval_extraction.py",
            "-g",
            gold_dir,
            "-p",
            out_dir,
            "-w",
            "-m",
            "normal",
        ],
        check=True,
        cwd=UIE_ROOT,
        env=env,
    )

    metrics: Dict[str, float] = {}
    if os.path.isfile(results_file):
        for line in open(results_file, encoding="utf-8"):
            line = line.strip()
            if line.startswith("test_offset-rel-strict-"):
                k, _, v = line.partition("=")
                try:
                    metrics[k] = float(v)
                except ValueError:
                    pass
    return metrics


def _normalize_rel_key(item) -> tuple:
    return tuple(tuple(p) if isinstance(p, list) else p for p in item)


def _relation_offset_set(record: dict) -> Set[tuple]:
    rel = record.get("relation", {})
    if isinstance(rel, dict):
        return {_normalize_rel_key(x) for x in rel.get("offset", [])}
    if isinstance(rel, list):
        out: Set[tuple] = set()
        for item in rel:
            if not isinstance(item, dict):
                continue
            rel_type = item.get("type", "")
            args = item.get("args") or []
            if len(args) < 2:
                continue
            h, t = args[0], args[1]
            if not isinstance(h, dict) or not isinstance(t, dict):
                continue
            out.add(
                (
                    rel_type,
                    h.get("type", ""),
                    tuple(h.get("offset") or []),
                    t.get("type", ""),
                    tuple(t.get("offset") or []),
                )
            )
        return out
    return set()


def eval_subset_report(
    pred_record_path: str,
    gold_file: str,
    subset_texts: Set[str],
) -> Dict[str, float]:
    from convert_for_uie_eval import normalize_text as nt

    gold_lines = [json.loads(l) for l in open(gold_file, encoding="utf-8") if l.strip()]
    pred_lines = [json.loads(l) for l in open(pred_record_path, encoding="utf-8") if l.strip()]
    subset_norm = {nt(t) for t in subset_texts}

    tp = fp = fn = 0
    for gold, pred in zip(gold_lines, pred_lines):
        content = nt(gold["text"])
        if content not in subset_norm:
            continue
        g_rel = _relation_offset_set(gold)
        p_rel = _relation_offset_set(pred)
        tp += len(g_rel & p_rel)
        fp += len(p_rel - g_rel)
        fn += len(g_rel - p_rel)

    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"P": p, "R": r, "F1": f1, "TP": tp, "FP": fp, "FN": fn}


def main() -> None:
    p = argparse.ArgumentParser(description="低置信度句子 Qwen 拆分 + 两阶段 RE + 评测 (v5)")
    p.add_argument("--out_dir", default=DEFAULT_OUT)
    p.add_argument("--prob_threshold", type=float, default=0.8)
    p.add_argument("--hard-drop-prob", type=float, default=HARD_DROP_PROB)
    p.add_argument("--api-key", default=os.environ.get("DASHSCOPE_API_KEY", ""))
    p.add_argument("--base-url", default="")
    p.add_argument("--model", default="")
    p.add_argument("--config", default=QWEN_CONFIG)
    p.add_argument("--device", default=os.environ.get("UIE_DEVICE", "gpu"))
    p.add_argument("--refresh-split-cache", action="store_true")
    p.add_argument("--skip-split", action="store_true")
    p.add_argument("--skip-predict", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument(
        "--include-empty-pred",
        action="store_true",
        help="同时将 baseline RE 为空的句子纳入拆分（默认仅低置信触发）",
    )
    p.add_argument(
        "--no-empty-pred",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--seed-split-cache",
        default="",
        help="从已有 qwen_split_cache.jsonl 复制作为初始缓存",
    )
    p.add_argument(
        "--split-min-prob",
        type=float,
        default=DEFAULT_SPLIT_MIN_PROB,
        help=f"拆分映射回源句后，仅保留 tail probability >= 该值的三元组（默认 {DEFAULT_SPLIT_MIN_PROB}）",
    )
    args = p.parse_args()
    args.out_dir = os.path.abspath(args.out_dir)
    if args.no_empty_pred:
        args.include_empty_pred = False

    os.makedirs(args.out_dir, exist_ok=True)
    base_url, model, config_key = load_qwen_defaults(args.config)
    base_url = args.base_url or base_url
    model = args.model or model
    api_key = args.api_key or config_key

    selected_path = os.path.join(args.out_dir, "selected_low_conf.json")
    splits_path = os.path.join(args.out_dir, "split_sentences.jsonl")
    split_cache = os.path.join(args.out_dir, "qwen_split_cache.jsonl")
    split_ner_path = os.path.join(args.out_dir, "split_NER.jsonl")
    split_pred_path = os.path.join(args.out_dir, "predict_split_two_stage.jsonl")
    merged_pred_path = os.path.join(args.out_dir, "predict_merged_two_stage.jsonl")
    merged_ner_path = os.path.join(args.out_dir, "coupled_NER_merged.jsonl")
    merged_simplified_path = os.path.join(args.out_dir, "coupled_RE_simplified.jsonl")
    report_path = os.path.join(args.out_dir, "evaluate_report.txt")

    bundles = select_sentence_bundles(
        BASELINE_SIMPLIFIED,
        BASELINE_PRED,
        args.prob_threshold,
        include_empty_pred=args.include_empty_pred,
    )
    selected = [b.source_text for b in bundles]
    total_high = sum(len(b.high_triples) for b in bundles)
    n_low = sum(1 for b in bundles if b.reason == "low_conf")
    n_empty = sum(1 for b in bundles if b.reason == "empty_pred")

    with open(selected_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "threshold": args.prob_threshold,
                "include_empty_pred": args.include_empty_pred,
                "count": len(bundles),
                "low_conf_count": n_low,
                "empty_pred_count": n_empty,
                "high_conf_triple_count": total_high,
                "sentences": [
                    {
                        "source_text": b.source_text,
                        "reason": b.reason,
                        "low_triples": b.low_triples,
                        "high_triples": b.high_triples,
                    }
                    for b in bundles
                ],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(
        f"待拆分源句: {len(bundles)} "
        f"(低置信={n_low}, 空RE={n_empty}, threshold={args.prob_threshold}), "
        f"待保留高置信三元组: {total_high}"
    )

    if args.refresh_split_cache and os.path.isfile(split_cache):
        os.remove(split_cache)
    elif args.seed_split_cache and not os.path.isfile(split_cache):
        import shutil

        shutil.copy2(args.seed_split_cache, split_cache)
        print(f"种子缓存: {args.seed_split_cache} -> {split_cache}")

    parent_ner_map = load_baseline_ner_map(BASELINE_NER)
    bundle_by_src = {b.source_text: b for b in bundles}

    if not args.skip_split:
        client, model_name = load_qwen_client(api_key, base_url, model)
        split_records = run_qwen_splits(
            client,
            model_name,
            bundles,
            splits_path,
            split_cache,
            args.refresh_split_cache,
            parent_ner_map,
        )
    else:
        split_records = []
        with open(splits_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    bundle = bundle_by_src.get(row["source_text"])
                    key_nouns = (
                        collect_key_nouns(bundle, parent_ner_map.get(row["source_text"]))
                        if bundle
                        else set()
                    )
                    if validate_split_in_parent(
                        row["split_text"], row["source_text"], key_nouns
                    ):
                        split_records.append(
                            SplitRecord(
                                row["source_id"],
                                row["source_text"],
                                row["split_index"],
                                row["split_text"],
                            )
                        )
        print(f"复用拆分句: {len(split_records)} 条（关键名词逐字校验通过）")

    baseline_pred_map = load_baseline_pred_map(BASELINE_PRED)

    if not args.skip_predict:
        if not split_records:
            print("警告: 无有效拆分句，将仅保留高置信 baseline 三元组")

        split_texts = [r.split_text for r in split_records]
        if split_texts:
            print(">>> 拆分句 RE（原句 NER 池 + v2严格/v3宽松 fallback 映射）")
            split_pred_rows = run_re_on_splits(
                split_records,
                RE_MODEL,
                args.device,
                parent_ner_map,
            )
            # 记录原句 NER 池中用于各拆分句的实体（非重跑 NER）
            pool_debug = []
            cfg = get_config("citrus")
            for rec in split_records:
                pool = pool_for_split_from_parent(
                    parent_ner_map.get(rec.source_text, {}),
                    rec.split_text,
                    cfg.ner_types,
                )
                pool_debug.append(
                    {
                        "input": rec.split_text,
                        "parent_pool": {k: sorted(v) for k, v in pool.items() if v},
                    }
                )
            write_jsonl(split_ner_path, pool_debug)
        else:
            split_pred_rows = []

        write_jsonl(split_pred_path, split_pred_rows)
        if split_pred_rows:
            export_simplified_triples(
                [{"content": r["split_text"], "pred": r["pred"]} for r in split_pred_rows],
                merged_simplified_path,
            )

        by_source: Dict[str, List[dict]] = defaultdict(list)
        for row in split_pred_rows:
            by_source[row["source_text"]].append(row)

        parent_preds: Dict[str, dict] = {}
        for bundle in bundles:
            src = bundle.source_text
            baseline = baseline_pred_map.get(src, {})
            split_part = merge_preds_to_parent(by_source.get(src, []))
            if args.split_min_prob > 0:
                split_part = filter_pred_keep_confidence(split_part, args.split_min_prob)
            filtered_base = filter_baseline_for_merge(
                baseline,
                split_part,
                args.prob_threshold,
                args.hard_drop_prob,
            )
            parent_preds[src] = merge_pred_dicts(filtered_base, split_part)
            bc = sum(
                len(tails)
                for items in filtered_base.values()
                for it in items
                for tails in (it.get("relations") or {}).values()
            )
            sc = sum(
                len(tails)
                for items in split_part.values()
                for it in items
                for tails in (it.get("relations") or {}).values()
            )
            print(f"  合并 [{src[:60]}...] baseline保留={bc} 拆分新增={sc}")

        build_merged_corpus(
            BASELINE_PRED,
            set(selected),
            parent_preds,
            merged_pred_path,
        )

        # NER：低置信句保留 baseline NER（高置信实体已在其中）
        merged_ner_rows = []
        with open(BASELINE_NER, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    merged_ner_rows.append(json.loads(line))
        write_jsonl(merged_ner_path, merged_ner_rows)
    else:
        merged_pred_path = os.path.join(args.out_dir, "predict_merged_two_stage.jsonl")
        merged_ner_path = os.path.join(args.out_dir, "coupled_NER_merged.jsonl")

    if not args.skip_eval:
        print(">>> 评测 baseline (全量)")
        base_metrics = run_uie_eval(
            BASELINE_PRED,
            BASELINE_NER,
            os.path.join(args.out_dir, "uie_baseline"),
        )
        print(">>> 评测 v5 split-merged (全量)")
        split_metrics = run_uie_eval(
            merged_pred_path,
            merged_ner_path,
            os.path.join(args.out_dir, "uie_split_merged"),
        )

        base_rec = os.path.join(args.out_dir, "uie_baseline", "test_preds_record.txt")
        split_rec = os.path.join(args.out_dir, "uie_split_merged", "test_preds_record.txt")
        subset_base = eval_subset_report(base_rec, GOLD_UIE, set(selected))
        subset_split = eval_subset_report(split_rec, GOLD_UIE, set(selected))

        strategy = "低置信触发拆分"
        if args.include_empty_pred:
            strategy = "低置信或空RE触发拆分"
        strategy += " | 原句NER池 | v2映射+空则v3宽松 | 低置信仅冲突/prob<0.5丢弃"
        if args.split_min_prob > 0:
            strategy += f" | split_min_prob={args.split_min_prob}"

        lines = [
            "=" * 72,
            "citrus 低置信度拆分复述 v5 + 两阶段 RE 评测",
            "=" * 72,
            f"策略: {strategy}",
            f"prob_threshold: {args.prob_threshold}, hard_drop: {args.hard_drop_prob}, "
            f"include_empty_pred: {args.include_empty_pred}, split_min_prob: {args.split_min_prob}",
            f"待拆分源句数: {len(bundles)} (低置信={n_low}, 空RE={n_empty})",
            f"保留高置信三元组数: {total_high}",
            f"有效拆分句数: {len(split_records)}",
            "",
            "--- 全量 test 集 (UIE offset-rel-strict) ---",
            f"Baseline  P={base_metrics.get('test_offset-rel-strict-P', 0):.4f}  "
            f"R={base_metrics.get('test_offset-rel-strict-R', 0):.4f}  "
            f"F1={base_metrics.get('test_offset-rel-strict-F1', 0):.4f}",
            f"v5 Merge  P={split_metrics.get('test_offset-rel-strict-P', 0):.4f}  "
            f"R={split_metrics.get('test_offset-rel-strict-R', 0):.4f}  "
            f"F1={split_metrics.get('test_offset-rel-strict-F1', 0):.4f}",
            "",
            f"--- 待拆分子集 ({len(bundles)} 句) ---",
            f"Baseline  P={subset_base['P']*100:.2f}% R={subset_base['R']*100:.2f}% F1={subset_base['F1']*100:.2f}% "
            f"(TP={subset_base['TP']} FP={subset_base['FP']} FN={subset_base['FN']})",
            f"v5 Merge  P={subset_split['P']*100:.2f}% R={subset_split['R']*100:.2f}% F1={subset_split['F1']*100:.2f}% "
            f"(TP={subset_split['TP']} FP={subset_split['FP']} FN={subset_split['FN']})",
            "",
            f"金标: {GOLD_UIE}",
            f"Baseline 预测: {BASELINE_PRED}",
            f"v5 合并预测: {merged_pred_path}",
            f"拆分句 RE: {split_pred_path}",
        ]
        text = "\n".join(lines)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(text)
        print(f"\n报告: {report_path}")


if __name__ == "__main__":
    main()
