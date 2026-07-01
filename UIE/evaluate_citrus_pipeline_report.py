#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
citrus 流水线评估报告：第一步 NER（各实体类型）+ 第二步 RE（各关系类型）。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from typing import Dict, List, Set

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from evaluate_citrus_ner_with_types import (  # noqa: E402
    eval_offset_mode,
    format_report as format_ner_report,
    load_uie_offset_instances,
)
from evaluate_citrus_re_with_types import format_report as format_re_report  # noqa: E402

NER_TYPES = [
    "aroma",
    "by-product",
    "citrus",
    "compound",
    "health",
    "location",
    "mouth_feel",
    "taste",
]


def normalize_text(text: str) -> str:
    text = (text or "").strip()
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    return text


def load_sentences(path: str) -> List[str]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(line)
    return rows


def export_e2e_ner_pred(e2e_pred_file: str, input_txt: str, output_jsonl: str) -> None:
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
    for text in load_sentences(input_txt):
        content = normalize_text(text)
        pred = pred_by_content.get(content, {})
        block: Dict[str, List[dict]] = defaultdict(list)
        seen: Dict[str, Set[tuple]] = defaultdict(set)
        if isinstance(pred, dict):
            for ent_type, items in pred.items():
                norm_type = normalize_text(ent_type)
                if norm_type not in NER_TYPES or not isinstance(items, list):
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
        rows.append({"input": text, "output": [dict(block)] if block else [{}]})

    os.makedirs(os.path.dirname(os.path.abspath(output_jsonl)) or ".", exist_ok=True)
    with open(output_jsonl, "w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_convert(convert_script: str, args: List[str]) -> None:
    cmd = [sys.executable, convert_script, *args]
    subprocess.run(cmd, check=True)


def build_ner_offset_report(
    gold_ner_file: str, pred_ner_record: str, step_title: str, pred_desc: str
) -> str:
    gold_instances, pred_instances = load_uie_offset_instances(
        gold_ner_file, pred_ner_record
    )
    total_metric, type_metrics = eval_offset_mode(gold_instances, pred_instances)
    header = [
        "",
        "#" * 90,
        step_title,
        f"金标: {gold_ner_file}",
        f"预测: {pred_desc}",
        "#" * 90,
        "",
    ]
    body = format_ner_report(
        "第一步 NER 评估（offset 匹配，与 UIE EntityScorer 一致）",
        total_metric,
        type_metrics,
        "类型 + token offset（normal 一对一匹配）",
    )
    return "\n".join(header) + body


def build_re_detail_report(gold_re_file: str, pred_re_record: str, pred_desc: str) -> str:
    header = [
        "",
        "#" * 90,
        "第二步 RE 评估",
        f"金标: {gold_re_file}",
        f"预测: {pred_desc}",
        "#" * 90,
        "",
    ]
    body = format_re_report(gold_re_file, pred_re_record)
    return "\n".join(header) + body


def main() -> None:
    parser = argparse.ArgumentParser(description="citrus 流水线 NER+RE 详细评估报告")
    parser.add_argument("--test_file", required=True)
    parser.add_argument("--input_txt", required=True)
    parser.add_argument("--pred_mode", choices=["e2e", "two_stage"], required=True)
    parser.add_argument("--ner_pred_file", default="")
    parser.add_argument("--e2e_pred_file", default="")
    parser.add_argument("--gold_re_file", required=True)
    parser.add_argument("--pred_re_record", required=True)
    parser.add_argument("--uie_pred_dir", required=True)
    parser.add_argument("--convert_script", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    os.makedirs(args.uie_pred_dir, exist_ok=True)
    gold_ner_file = os.path.join(args.uie_pred_dir, "gold_ner_test.json")
    pred_ner_record = os.path.join(args.uie_pred_dir, "ner_preds_record.txt")

    run_convert(
        args.convert_script,
        [
            "build-gold-ner",
            "--test_file",
            args.test_file,
            "--input_txt",
            args.input_txt,
            "--output",
            gold_ner_file,
        ],
    )

    ner_pred_jsonl = args.ner_pred_file
    ner_pred_desc = args.ner_pred_file
    if args.pred_mode == "e2e":
        if not args.e2e_pred_file:
            parser.error("e2e 模式需要 --e2e_pred_file")
        ner_pred_jsonl = os.path.join(args.uie_pred_dir, "e2e_ner_export.jsonl")
        export_e2e_ner_pred(args.e2e_pred_file, args.input_txt, ner_pred_jsonl)
        ner_pred_desc = f"{ner_pred_jsonl}（由 e2e 预测 {args.e2e_pred_file} 导出实体）"

    if not ner_pred_jsonl or not os.path.isfile(ner_pred_jsonl):
        raise FileNotFoundError(f"NER 预测文件不存在: {ner_pred_jsonl}")

    run_convert(
        args.convert_script,
        [
            "build-pred-ner",
            "--pred_file",
            ner_pred_jsonl,
            "--gold_file",
            gold_ner_file,
            "--output",
            pred_ner_record,
        ],
    )

    mode_label = "端到端 (e2e)" if args.pred_mode == "e2e" else "两阶段 (two_stage)"
    sections = [
        "=" * 90,
        f"citrus 流水线详细评估 — {mode_label}",
        "=" * 90,
    ]
    sections.append(
        build_ner_offset_report(
            gold_ner_file,
            pred_ner_record,
            "【第一步】NER — 各实体类型 P/R/F1",
            ner_pred_desc,
        )
    )
    sections.append(
        build_re_detail_report(
            args.gold_re_file,
            args.pred_re_record,
            args.pred_re_record,
        )
    )

    report = "\n".join(sections)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report)
    print(report, end="")
    print(f"流水线详细报告已保存到: {args.output}")


if __name__ == "__main__":
    main()
