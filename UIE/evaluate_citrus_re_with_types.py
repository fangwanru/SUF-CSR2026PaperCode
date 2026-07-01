#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
citrusRE 评估：总体与各关系类型的 P/R/F1（strict / boundary）。

使用 UIE 官方金标/预测，offset 匹配与 RelationScorer 一致。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
UIE_ROOT = os.environ.get("UIE_ROOT", os.path.join(SCRIPT_DIR, "UIE"))
if UIE_ROOT not in sys.path:
    sys.path.insert(0, UIE_ROOT)

from uie.extraction.scorer import Metric, tuple_offset  # noqa: E402

RELATION_TYPES = [
    "category",
    "contain",
    "has_aroma",
    "has_mouth_feel",
    "has_taste",
    "impact_on",
    "produce",
    "produced_in",
]


def load_gold_relations(gold_file: str) -> List[List[Tuple]]:
    instances: List[List[Tuple]] = []
    with open(gold_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            gold = json.loads(line)
            strict_list = []
            for record in gold.get("relation") or []:
                args = record.get("args") or []
                if len(args) != 2:
                    continue
                strict_list.append(
                    (
                        record["type"],
                        args[0]["type"],
                        tuple_offset(args[0]["offset"]),
                        args[1]["type"],
                        tuple_offset(args[1]["offset"]),
                    )
                )
            instances.append(strict_list)
    return instances


def load_pred_relations(pred_file: str) -> List[List[Tuple]]:
    instances: List[List[Tuple]] = []
    with open(pred_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pred = json.loads(line)
            strict_list = []
            for item in pred.get("relation", {}).get("offset") or []:
                if len(item) != 5:
                    continue
                head_off = item[2]
                tail_off = item[4]
                if not isinstance(head_off, tuple):
                    head_off = tuple_offset(head_off)
                if not isinstance(tail_off, tuple):
                    tail_off = tuple_offset(tail_off)
                strict_list.append(
                    (
                        item[0],
                        item[1],
                        head_off,
                        item[3],
                        tail_off,
                    )
                )
            instances.append(strict_list)
    return instances


def to_boundary(rel_list: List[Tuple]) -> List[Tuple]:
    return [(r[0], r[2], r[4]) for r in rel_list]


def eval_mode(
    gold_instances: List[List[Tuple]],
    pred_instances: List[List[Tuple]],
    mode: str,
) -> Tuple[Metric, Dict[str, Metric]]:
    total_metric = Metric(match_mode="normal")
    type_metrics = {t: Metric(match_mode="normal") for t in RELATION_TYPES}

    for gold_strict, pred_strict in zip(gold_instances, pred_instances):
        if mode == "strict":
            gold_list = gold_strict
            pred_list = pred_strict
        else:
            gold_list = to_boundary(gold_strict)
            pred_list = to_boundary(pred_strict)

        total_metric.count_instance(gold_list, pred_list)
        for rel_type in RELATION_TYPES:
            g = [item for item in gold_list if item[0] == rel_type]
            p = [item for item in pred_list if item[0] == rel_type]
            type_metrics[rel_type].count_instance(g, p)

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


def format_mode_section(
    title: str,
    match_desc: str,
    total_metric: Metric,
    type_metrics: Dict[str, Metric],
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
    lines.append("各关系类型评估结果")
    lines.append("-" * 90)
    lines.append(
        f"{'关系类型':<22}{'Precision':>12}{'Recall':>12}{'F1':>12}{'TP':>8}{'FP':>8}{'FN':>8}"
    )
    lines.append("-" * 90)

    for rel_type in RELATION_TYPES:
        rp, rr, rf1, rtp, rfp, rfn = metric_prf(type_metrics[rel_type])
        if rtp + rfp + rfn == 0:
            continue
        lines.append(
            f"{rel_type:<22}{rp:>12.6f}{rr:>12.6f}{rf1:>12.6f}"
            f"{rtp:>8}{rfp:>8}{rfn:>8}"
        )

    lines.append("=" * 90)
    return "\n".join(lines) + "\n"


def format_report(gold_file: str, pred_file: str) -> str:
    gold_instances = load_gold_relations(gold_file)
    pred_instances = load_pred_relations(pred_file)

    strict_total, strict_by_type = eval_mode(gold_instances, pred_instances, "strict")
    boundary_total, boundary_by_type = eval_mode(
        gold_instances, pred_instances, "boundary"
    )

    sections = []
    sections.append(
        format_mode_section(
            "citrusRE Strict 评估（offset-rel-strict）",
            "关系类型 + 头实体类型 + 头 token span + 尾实体类型 + 尾 token span",
            strict_total,
            strict_by_type,
        )
    )
    sections.append(
        format_mode_section(
            "citrusRE Boundary 评估（offset-rel-boundary）",
            "关系类型 + 头 token span + 尾 token span（忽略实体类型）",
            boundary_total,
            boundary_by_type,
        )
    )
    return "\n".join(sections)


def main() -> None:
    parser = argparse.ArgumentParser(description="citrusRE 各关系类型 P/R/F1 评估")
    parser.add_argument("--gold_file", required=True)
    parser.add_argument("--pred_record", required=True)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    report = format_report(args.gold_file, args.pred_record)
    print(report, end="")
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"评估结果已保存到: {args.output}")


if __name__ == "__main__":
    main()
