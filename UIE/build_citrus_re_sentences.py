#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 citrusRE test.txt 提取去重句子，生成预测输入 test_sentences.txt。"""

import argparse
import json
from collections import OrderedDict


def build_sentences(test_file: str, output_file: str) -> int:
    uniq_sentences = OrderedDict()
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            content = json.loads(line).get("content", "").strip()
            if content and content not in uniq_sentences:
                uniq_sentences[content] = None

    with open(output_file, "w", encoding="utf-8") as f:
        for sentence in uniq_sentences:
            f.write(sentence + "\n")

    return len(uniq_sentences)


def main():
    parser = argparse.ArgumentParser(description="构建 citrus RE 预测句子文件")
    parser.add_argument("--test_file", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    n = build_sentences(args.test_file, args.output)
    print(f"共写出 {n} 条句子到 {args.output}")


if __name__ == "__main__":
    main()
