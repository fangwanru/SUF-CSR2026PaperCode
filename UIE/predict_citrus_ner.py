#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对 citrusNER 测试集做 NER 预测，输出评估格式 JSONL。"""

import argparse
import json
import os
import sys
from collections import OrderedDict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from uie1_predictor import UIEPredictor

SCHEMA = [
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


def load_unique_texts(path: str):
    texts = OrderedDict()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            text = item.get("content", "").strip()
            if text and text not in texts:
                texts[text] = None
    return list(texts.keys())


def convert_output(raw_output):
    converted = {}
    if isinstance(raw_output, list) and raw_output:
        first = raw_output[0]
        if isinstance(first, dict):
            for entity_type, values in first.items():
                if not isinstance(values, list):
                    continue
                entities = []
                for x in values:
                    if isinstance(x, dict) and "text" in x:
                        val = normalize_entity(x["text"])
                        if val and val not in entities:
                            entities.append(val)
                converted[entity_type] = entities
    return converted


def main():
    parser = argparse.ArgumentParser(description="citrus NER 预测")
    parser.add_argument("--test_file", required=True)
    parser.add_argument("--model_dir", required=True, help="PyTorch 模型目录")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="gpu", choices=["cpu", "gpu"])
    parser.add_argument("--schema_lang", default="en")
    parser.add_argument("--position_prob", type=float, default=0.5)
    args = parser.parse_args()

    texts = load_unique_texts(args.test_file)
    print(f"读取待预测文本: {len(texts)} 条")
    print(f"模型目录: {args.model_dir}")

    uie = UIEPredictor(
        model="uie-base-en",
        task_path=args.model_dir,
        schema=SCHEMA,
        schema_lang=args.schema_lang,
        max_seq_len=512,
        device=args.device,
        position_prob=args.position_prob,
    )

    with open(args.output, "w", encoding="utf-8") as f:
        for idx, text in enumerate(texts, 1):
            raw_output = uie(text)
            converted = convert_output(raw_output)
            item = {"input": text, "output": [converted]}
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            if idx % 50 == 0:
                print(f"已预测: {idx}/{len(texts)}")

    print(f"预测完成，输出文件: {args.output}")


if __name__ == "__main__":
    main()
