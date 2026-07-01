#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
citrusNER 评估：总体与各实体类型的 P/R/F1。

默认使用 UIE 官方金标/预测（offset 匹配，与 EntityScorer 一致）。
也可通过 --test_file + --pred_file 对 JSONL 做类型+文本匹配。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
UIE_ROOT = os.environ.get("UIE_ROOT", os.path.join(SCRIPT_DIR, "UIE"))
if UIE_ROOT not in sys.path:
    sys.path.insert(0, UIE_ROOT)

from uie.extraction.scorer import Metric, tuple_offset  # noqa: E402

ENTITY_TYPES = [
    "aroma",
    "by-product",
    "citrus",
    "compound",
    "health",
    "location",
    "mouth_feel",
    "taste",
]


def normalize_entity(entity: str) -> str:
    entity = entity.strip()
    entity = entity.replace("\u2018", "'").replace("\u2019", "'")
    entity = entity.replace("\u201C", '"').replace("\u201D", '"')
    return entity


def load_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def extract_entities_from_jsonl(data: dict) -> Dict[str, set]:
    entities: Dict[str, set] = defaultdict(set)
    for output_dict in data.get("output") or []:
        if not isinstance(output_dict, dict):
            continue
        for entity_type, entity_list in output_dict.items():
            if not isinstance(entity_list, list):
                continue
            for entity in entity_list:
                if isinstance(entity, dict) and "text" in entity:
                    text = normalize_entity(str(entity["text"]))
                else:
                    text = normalize_entity(str(entity))
                if text:
                    entities[entity_type].add(text)
    return dict(entities)


def load_gold_jsonl(test_file: str) -> List[dict]:
    return load_jsonl(test_file)


def eval_string_mode(gold_rows: List[dict], pred_rows: List[dict]) -> Tuple[dict, dict]:
    pred_by_input = {}
    for row in pred_rows:
        inp = (row.get("input") or "").strip()
        if inp:
            pred_by_input[inp] = row

    total_metric = Metric(match_mode="normal")
    type_metrics = {t: Metric(match_mode="normal") for t in ENTITY_TYPES}

    for gold in gold_rows:
        inp = (gold.get("input") or "").strip()
        pred = pred_by_input.get(inp, {"input": inp, "output": [{}]})
        gold_entities = extract_entities_from_jsonl(gold)
        pred_entities = extract_entities_from_jsonl(pred)
        all_types = set(gold_entities) | set(pred_entities) | set(ENTITY_TYPES)

        gold_all = []
        pred_all = []
        for entity_type in all_types:
            gold_set = gold_entities.get(entity_type, set())
            pred_set = pred_entities.get(entity_type, set())
            gold_list = [(entity_type, text) for text in sorted(gold_set)]
            pred_list = [(entity_type, text) for text in sorted(pred_set)]
            gold_all.extend(gold_list)
            pred_all.extend(pred_list)
            if entity_type in type_metrics:
                type_metrics[entity_type].count_instance(gold_list, pred_list)

        total_metric.count_instance(gold_all, pred_all)

    return total_metric, type_metrics


def load_uie_offset_instances(
    gold_file: str, pred_file: str
) -> Tuple[List[List[Tuple[str, tuple]]], List[List[Tuple[str, tuple]]]]:
    gold_instances: List[List[Tuple[str, tuple]]] = []
    pred_instances: List[List[Tuple[str, tuple]]] = []

    with open(gold_file, "r", encoding="utf-8") as gf, open(
        pred_file, "r", encoding="utf-8"
    ) as pf:
        for gline, pline in zip(gf, pf):
            gold = json.loads(gline)
            pred = json.loads(pline)
            gold_list = [
                (ent["type"], tuple_offset(ent["offset"]))
                for ent in gold.get("entity") or []
            ]
            pred_list = [
                (item[0], tuple_offset(item[1]))
                for item in pred.get("entity", {}).get("offset") or []
            ]
            gold_instances.append(gold_list)
            pred_instances.append(pred_list)

    return gold_instances, pred_instances


def eval_offset_mode(
    gold_instances: List[List[Tuple[str, tuple]]],
    pred_instances: List[List[Tuple[str, tuple]]],
) -> Tuple[Metric, Dict[str, Metric]]:
    total_metric = Metric(match_mode="normal")
    type_metrics = {t: Metric(match_mode="normal") for t in ENTITY_TYPES}

    for gold_list, pred_list in zip(gold_instances, pred_instances):
        total_metric.count_instance(gold_list, pred_list)
        for entity_type in ENTITY_TYPES:
            g = [item for item in gold_list if item[0] == entity_type]
            p = [item for item in pred_list if item[0] == entity_type]
            type_metrics[entity_type].count_instance(g, p)

    return total_metric, type_metrics


def metric_prf(metric: Metric) -> Tuple[float, float, float, int, int, int]:
    tp = int(metric.tp)
    gold = int(metric.gold_num)
    pred = int(metric.pred_num)
    fp = pred - tp
    fn = gold - tp
    p = metric.safe_div(tp, pred)
    r = metric.safe_div(tp, gold)
    f1 = metric.safe_div(2 * p * r, p + r)
    return p, r, f1, tp, fp, fn


def format_report(
    title: str,
    total_metric: Metric,
    type_metrics: Dict[str, Metric],
    match_desc: str,
) -> str:
    lines = []
    p, r, f1, tp, fp, fn = metric_prf(total_metric)
    lines.append("=" * 90)
    lines.append(title)
    lines.append("=" * 90)
    lines.append(f"匹配口径: {match_desc}")
    lines.append(f"TP={tp}, FP={fp}, FN={fn}")
    lines.append(f"Precision={p:.6f} ({p * 100:.2f}%)")
    lines.append(f"Recall   ={r:.6f} ({r * 100:.2f}%)")
    lines.append(f"F1       ={f1:.6f} ({f1 * 100:.2f}%)")
    lines.append("")
    lines.append("=" * 90)
    lines.append("各实体类型评估结果")
    lines.append("=" * 90)
    lines.append(
        f"{'实体类型':<22}{'Precision':>12}{'Recall':>12}{'F1':>12}{'TP':>8}{'FP':>8}{'FN':>8}"
    )
    lines.append("-" * 90)

    for entity_type in ENTITY_TYPES:
        tp_p, tp_r, tp_f1, tp, fp, fn = metric_prf(type_metrics[entity_type])
        if tp + fp + fn == 0:
            continue
        lines.append(
            f"{entity_type:<22}{tp_p:>12.6f}{tp_r:>12.6f}{tp_f1:>12.6f}"
            f"{tp:>8}{fp:>8}{fn:>8}"
        )

    lines.append("=" * 90)
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="citrusNER 各类型 P/R/F1 评估")
    parser.add_argument("--gold_file", default="", help="UIE 官方金标 test.json")
    parser.add_argument("--pred_record", default="", help="UIE 官方预测 test_preds_record.txt")
    parser.add_argument("--test_file", default="", help="金标 JSONL（build_citrus_ner_gold.py 输出）")
    parser.add_argument("--pred_file", default="", help="预测 JSONL")
    parser.add_argument("--output", default="", help="可选，写出报告文件")
    args = parser.parse_args()

    if args.gold_file and args.pred_record:
        gold_instances, pred_instances = load_uie_offset_instances(
            args.gold_file, args.pred_record
        )
        total_metric, type_metrics = eval_offset_mode(gold_instances, pred_instances)
        report = format_report(
            "citrusNER 评估（offset 匹配，与 UIE EntityScorer 一致）",
            total_metric,
            type_metrics,
            "类型 + token offset（normal 一对一匹配）",
        )
    elif args.test_file and args.pred_file:
        gold_rows = load_gold_jsonl(args.test_file)
        pred_rows = load_jsonl(args.pred_file)
        total_metric, type_metrics = eval_string_mode(gold_rows, pred_rows)
        report = format_report(
            "citrusNER 评估（类型 + 文本匹配）",
            total_metric,
            type_metrics,
            "类型 + 实体文本（normal 一对一匹配）",
        )
    else:
        parser.error("请提供 --gold_file + --pred_record，或 --test_file + --pred_file")

    print(report, end="")
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"评估结果已保存到: {args.output}")


if __name__ == "__main__":
    main()
