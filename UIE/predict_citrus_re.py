#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对 citrusRE 测试句子做关系抽取预测。"""

import argparse
import json
import os
import sys
from typing import List

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from uie1_predictor import UIEPredictor

DEFAULT_SCHEMA = {
    "aroma": [
        "by-product",
        "category",
        "citrus",
        "compound",
        "contain",
        "has_aroma",
        "health",
        "impact_on",
        "location",
        "mouth_feel",
        "taste",
    ],
    "by-product": [
        "aroma",
        "category",
        "citrus",
        "compound",
        "contain",
        "has_aroma",
        "has_mouth_feel",
        "has_taste",
        "health",
        "impact_on",
        "location",
        "mouth_feel",
        "produce",
        "produced_in",
        "taste",
    ],
    "citrus": [
        "aroma",
        "by-product",
        "category",
        "compound",
        "contain",
        "has_aroma",
        "has_mouth_feel",
        "has_taste",
        "health",
        "impact_on",
        "location",
        "mouth_feel",
        "produce",
        "produced_in",
        "taste",
    ],
    "compound": [
        "aroma",
        "by-product",
        "category",
        "citrus",
        "contain",
        "has_aroma",
        "has_mouth_feel",
        "has_taste",
        "health",
        "impact_on",
        "location",
        "mouth_feel",
        "produce",
        "taste",
    ],
    "health": [
        "category",
        "contain",
        "impact_on",
        "produce",
    ],
    "location": [
        "category",
        "compound",
        "mouth_feel",
        "produced_in",
        "taste",
    ],
    "mouth_feel": [
        "aroma",
        "by-product",
        "category",
        "citrus",
        "has_mouth_feel",
        "has_taste",
        "health",
        "location",
        "taste",
    ],
    "taste": [
        "aroma",
        "by-product",
        "category",
        "citrus",
        "compound",
        "has_mouth_feel",
        "has_taste",
        "impact_on",
        "location",
        "mouth_feel",
    ],
}


def resolve_task_path(model_root: str) -> str:
    model_root = os.path.abspath(model_root)
    model_best = os.path.join(model_root, "model_best")
    if os.path.isdir(model_best):
        return model_best
    return model_root


def load_sentences(path: str) -> List[str]:
    sentences = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                sentences.append(text)
    return sentences


def write_jsonl(path: str, rows: List[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(to_jsonable(row), ensure_ascii=False) + "\n")


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def main():
    parser = argparse.ArgumentParser(description="citrus RE 关系抽取预测")
    parser.add_argument("--input_txt", required=True, help="每行一个句子的输入文件")
    parser.add_argument("--model_dir", required=True, help="PyTorch 模型目录")
    parser.add_argument("--output_jsonl", required=True, help="预测输出 JSONL")
    parser.add_argument("--model", default="uie-base-en")
    parser.add_argument("--schema_lang", default="en", choices=["zh", "en"])
    parser.add_argument("--device", default="gpu", choices=["cpu", "gpu"])
    parser.add_argument("--engine", default="pytorch", choices=["pytorch", "onnx"])
    parser.add_argument("--position_prob", type=float, default=0.5)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    task_path = resolve_task_path(args.model_dir)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_jsonl)), exist_ok=True)

    sentences = load_sentences(args.input_txt)
    if not sentences:
        raise ValueError(f"输入文件为空: {args.input_txt}")

    print(f"输入句子数: {len(sentences)}")
    print(f"使用模型目录: {task_path}")

    uie = UIEPredictor(
        model=args.model,
        schema=DEFAULT_SCHEMA,
        task_path=task_path,
        schema_lang=args.schema_lang,
        engine=args.engine,
        device=args.device,
        position_prob=args.position_prob,
        max_seq_len=args.max_seq_len,
        batch_size=1,
        split_sentence=False,
        use_fp16=False,
    )

    rows = []
    total = len(sentences)
    for idx, text in enumerate(sentences, start=1):
        pred = uie(text)[0]
        rows.append({"content": text, "pred": pred})
        if idx % args.batch_size == 0 or idx == total:
            print(f"已预测: {idx}/{total}")

    write_jsonl(args.output_jsonl, rows)
    print(f"预测完成，输出文件: {args.output_jsonl}")


if __name__ == "__main__":
    main()
