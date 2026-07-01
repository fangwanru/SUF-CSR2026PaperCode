#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SUF two-stage pipeline (NER -> RE) without CSR re-extraction.

Corresponds to paper ablation "-w/o CSR module" (Table 7).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from ablation_config import get_config  # noqa: E402
from ablation_engine import run_prompt_re, write_jsonl  # noqa: E402
from paths import (  # noqa: E402
    CITRUS_OUT,
    GOLD_UIE,
    NER_PRED,
    PRED_TWO_STAGE,
    RE_SENTENCES,
    RE_TEST,
    SIMPLIFIED_RE,
    SUF_NER_MODEL,
    SUF_RE_MODEL,
)
from predict_citrus_ner import SCHEMA, convert_output  # noqa: E402
from uie1_predictor import UIEPredictor  # noqa: E402


def run_ner(model_dir: str, output_jsonl: str, device: str) -> None:
    texts = []
    seen = set()
    with open(RE_TEST, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            text = row.get("content", "").strip()
            if text and text not in seen:
                seen.add(text)
                texts.append(text)

    uie = UIEPredictor(
        model="uie-base-en",
        task_path=model_dir,
        schema=SCHEMA,
        schema_lang="en",
        max_seq_len=512,
        device=device,
        position_prob=0.5,
    )
    rows = []
    for idx, text in enumerate(texts, 1):
        raw = uie(text)
        converted = convert_output(raw)
        rows.append({"input": text, "output": [converted]})
        if idx % 50 == 0:
            print(f"NER {idx}/{len(texts)}")
    write_jsonl(output_jsonl, rows)
    print(f"NER saved: {output_jsonl}")


def build_gold() -> None:
    convert = os.path.join(SCRIPT_DIR, "convert_for_uie_eval.py")
    os.makedirs(os.path.dirname(GOLD_UIE), exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            convert,
            "build-gold",
            "--test_file",
            RE_TEST,
            "--input_txt",
            RE_SENTENCES,
            "--output",
            GOLD_UIE,
        ],
        check=True,
    )


def evaluate(pred_file: str, ner_file: str, report_file: str) -> None:
    convert = os.path.join(SCRIPT_DIR, "convert_for_uie_eval.py")
    uie_root = os.path.join(SCRIPT_DIR, "UIE")
    pred_dir = os.path.join(CITRUS_OUT, "uie_official", "pred_two_stage")
    pred_record = os.path.join(pred_dir, "test_preds_record.txt")
    results_file = os.path.join(pred_dir, "test_results.txt")
    os.makedirs(pred_dir, exist_ok=True)

    build_gold()
    subprocess.run(
        [
            sys.executable,
            convert,
            "build-pred",
            "--pred_file",
            pred_file,
            "--gold_file",
            GOLD_UIE,
            "--output",
            pred_record,
            "--ner_pred_file",
            ner_file,
        ],
        check=True,
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{uie_root}:{SCRIPT_DIR}:{env.get('PYTHONPATH', '')}"
    subprocess.run(
        [
            sys.executable,
            "scripts/eval_extraction.py",
            "-g",
            os.path.dirname(GOLD_UIE),
            "-p",
            pred_dir,
            "-w",
            "-m",
            "normal",
        ],
        check=True,
        cwd=uie_root,
        env=env,
    )

    per_type = os.path.join(pred_dir, "per_type_report.txt")
    subprocess.run(
        [
            sys.executable,
            os.path.join(SCRIPT_DIR, "evaluate_citrus_re_with_types.py"),
            "--gold_file",
            GOLD_UIE,
            "--pred_record",
            pred_record,
            "--output",
            per_type,
        ],
        check=True,
    )

    with open(report_file, "w", encoding="utf-8") as out:
        out.write("SUF two-stage (w/o CSR) — UIE official eval\n")
        out.write(f"NER model: {SUF_NER_MODEL}\n")
        out.write(f"RE model:  {SUF_RE_MODEL}\n\n")
        if os.path.isfile(results_file):
            for line in open(results_file, encoding="utf-8"):
                if line.startswith("test_offset-rel-"):
                    out.write(line)
        if os.path.isfile(per_type):
            out.write("\n")
            out.write(open(per_type, encoding="utf-8").read())
    print(f"Report: {report_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="SUF two-stage NER+RE (no CSR)")
    parser.add_argument("--device", default=os.environ.get("UIE_DEVICE", "gpu"))
    parser.add_argument("--skip-ner", action="store_true")
    parser.add_argument("--skip-re", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    os.makedirs(CITRUS_OUT, exist_ok=True)
    cfg = get_config("citrus")

    if not args.skip_ner:
        run_ner(SUF_NER_MODEL, NER_PRED, args.device)

    if not args.skip_re:
        run_prompt_re(
            cfg,
            PRED_TWO_STAGE,
            "pred_ner",
            ner_jsonl=NER_PRED,
            simplified_jsonl=SIMPLIFIED_RE,
        )

    if not args.skip_eval:
        report = os.path.join(CITRUS_OUT, "evaluate_two_stage_no_csr.txt")
        evaluate(PRED_TWO_STAGE, NER_PRED, report)


if __name__ == "__main__":
    main()
