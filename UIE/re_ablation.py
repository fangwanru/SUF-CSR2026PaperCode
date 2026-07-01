#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多数据集消融实验 CLI：SciERC / CoNLL04 / citrus

  oracle-ner | e2e-ner-types | prompt-re | oracle-prompt
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

LTONG_DIR = os.path.dirname(os.path.abspath(__file__))
if LTONG_DIR not in sys.path:
    sys.path.insert(0, LTONG_DIR)

from ablation_config import get_config  # noqa: E402
from ablation_engine import (  # noqa: E402
    export_e2e_ner_types,
    export_oracle_ner_from_conll_json,
    export_oracle_ner_from_test,
    export_simplified_triples,
    run_oracle_prompt,
    run_prompt_re,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RE 消融实验辅助脚本")
    p.add_argument("--dataset", required=True, choices=["citrus"])
    sub = p.add_subparsers(dest="cmd", required=True)

    p_oracle = sub.add_parser("oracle-ner")
    p_oracle.add_argument("--output", required=True)

    p_e2e = sub.add_parser("e2e-ner-types")
    p_e2e.add_argument("--e2e_pred_file", required=True)
    p_e2e.add_argument("--output", required=True)

    p_re = sub.add_parser("prompt-re")
    p_re.add_argument("--output_jsonl", required=True)
    p_re.add_argument(
        "--head_source",
        required=True,
        choices=["oracle_ner", "pred_ner", "e2e_pred"],
    )
    p_re.add_argument("--ner_jsonl", default="")
    p_re.add_argument("--e2e_pred_file", default="")
    p_re.add_argument(
        "--simplified_jsonl",
        default="",
        help="展平三元组输出路径；citrus 默认写同目录 coupled_RE_simplified.jsonl",
    )

    p_c1 = sub.add_parser("oracle-prompt")
    p_c1.add_argument("--output_jsonl", required=True)

    p_simp = sub.add_parser("export-simplified")
    p_simp.add_argument("--pred_jsonl", required=True, help="predict_*.jsonl（content+pred）")
    p_simp.add_argument("--output", required=True, help="coupled_RE_simplified.jsonl")

    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg = get_config(args.dataset)

    if args.cmd == "oracle-ner":
        if cfg.name == "conll04" and cfg.conll04_gold_json:
            export_oracle_ner_from_conll_json(cfg, args.output)
        else:
            export_oracle_ner_from_test(cfg, args.output)
    elif args.cmd == "e2e-ner-types":
        export_e2e_ner_types(cfg, args.e2e_pred_file, args.output)
    elif args.cmd == "prompt-re":
        run_prompt_re(
            cfg,
            args.output_jsonl,
            args.head_source,
            ner_jsonl=args.ner_jsonl,
            e2e_pred_file=args.e2e_pred_file,
            simplified_jsonl=args.simplified_jsonl,
        )
    elif args.cmd == "oracle-prompt":
        run_oracle_prompt(cfg, args.output_jsonl)
    elif args.cmd == "export-simplified":
        rows = []
        with open(args.pred_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        export_simplified_triples(rows, args.output)


if __name__ == "__main__":
    main()
