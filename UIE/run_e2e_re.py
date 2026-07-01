#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-end RE without NER stage (UIE-RE-tuned baseline).

Corresponds to paper ablation "-w/o NER module" (Table 7) and UIE-RE-tuned (Table 6).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from paths import CITRUS_OUT, GOLD_UIE, PRED_E2E, RE_SENTENCES, RE_TEST, SUF_RE_MODEL


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end RE (w/o NER stage)")
    parser.add_argument("--device", default=os.environ.get("UIE_DEVICE", "gpu"))
    parser.add_argument("--skip-predict", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    os.makedirs(CITRUS_OUT, exist_ok=True)
    convert = os.path.join(SCRIPT_DIR, "convert_for_uie_eval.py")
    predict = os.path.join(SCRIPT_DIR, "predict_citrus_re.py")
    uie_root = os.path.join(SCRIPT_DIR, "UIE")
    pred_dir = os.path.join(CITRUS_OUT, "uie_official", "pred_e2e")
    pred_record = os.path.join(pred_dir, "test_preds_record.txt")
    report = os.path.join(CITRUS_OUT, "evaluate_e2e_no_ner.txt")

    if not os.path.isfile(RE_SENTENCES):
        subprocess.run(
            [sys.executable, os.path.join(SCRIPT_DIR, "build_citrus_re_sentences.py"),
             "--test_file", RE_TEST, "--output", RE_SENTENCES],
            check=True,
        )

    if not args.skip_predict:
        subprocess.run(
            [
                sys.executable,
                predict,
                "--input_txt",
                RE_SENTENCES,
                "--model_dir",
                SUF_RE_MODEL,
                "--output_jsonl",
                PRED_E2E,
                "--device",
                args.device,
                "--schema_lang",
                "en",
                "--position_prob",
                "0.5",
            ],
            check=True,
        )

    if args.skip_eval:
        return

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
    os.makedirs(pred_dir, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            convert,
            "build-pred",
            "--pred_file",
            PRED_E2E,
            "--gold_file",
            GOLD_UIE,
            "--output",
            pred_record,
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

    results = os.path.join(pred_dir, "test_results.txt")
    with open(report, "w", encoding="utf-8") as out:
        out.write("End-to-end RE (w/o NER stage) — UIE official eval\n")
        out.write(f"RE model: {SUF_RE_MODEL}\n\n")
        if os.path.isfile(results):
            for line in open(results, encoding="utf-8"):
                if line.startswith("test_offset-rel-"):
                    out.write(line)
    print(f"Report: {report}")


if __name__ == "__main__":
    main()
