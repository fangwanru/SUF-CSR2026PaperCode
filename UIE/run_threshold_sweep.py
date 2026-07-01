#!/usr/bin/env python3
"""Run prob_threshold sweep (0.60–0.90, step 0.05) and collect P/R/F1 table."""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from paths import CITRUS_OUT, THRESHOLD_SWEEP_OUT  # noqa: E402

PIPELINE = os.path.join(SCRIPT_DIR, "run_citrus_lowconf_split_pipeline.py")
SWEEP_ROOT = THRESHOLD_SWEEP_OUT
T085_DIR = os.path.join(CITRUS_OUT, "suf_csr")
MASTER_CACHE = os.path.join(SWEEP_ROOT, "qwen_split_cache_master.jsonl")
SHARED_CACHE = os.path.join(SWEEP_ROOT, "qwen_split_cache_shared.jsonl")

THRESHOLDS = [0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60]


def ensure_shared_cache() -> None:
    os.makedirs(SWEEP_ROOT, exist_ok=True)
    if not os.path.isfile(SHARED_CACHE):
        src = MASTER_CACHE if os.path.isfile(MASTER_CACHE) else os.path.join(T085_DIR, "qwen_split_cache.jsonl")
        if os.path.isfile(src):
            shutil.copy2(src, SHARED_CACHE)
            print(f"Init shared cache from {src}")


def merge_cache_into_shared(out_dir: str) -> None:
    cache = os.path.join(out_dir, "qwen_split_cache.jsonl")
    if not os.path.isfile(cache):
        return
    existing = set()
    if os.path.isfile(SHARED_CACHE):
        with open(SHARED_CACHE, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    existing.add(json.loads(line)["source_text"])
    with open(SHARED_CACHE, "a", encoding="utf-8") as out, open(cache, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            src = row["source_text"]
            if src not in existing:
                out.write(line if line.endswith("\n") else line + "\n")
                existing.add(src)


def parse_metrics(report_path: str) -> dict:
    with open(report_path, encoding="utf-8") as f:
        text = f.read()
    m = re.search(
        r"v5 Merge\s+P=([\d.]+)\s+R=([\d.]+)\s+F1=([\d.]+)",
        text,
    )
    if not m:
        raise ValueError(f"Cannot parse metrics from {report_path}")
    return {"P": float(m.group(1)), "R": float(m.group(2)), "F1": float(m.group(3))}


def run_one(threshold: float, api_key: str) -> dict:
    tag = f"t{threshold:.2f}"
    out_dir = os.path.join(SWEEP_ROOT, tag)
    report = os.path.join(out_dir, "evaluate_report.txt")

    if threshold == 0.85 and os.path.isfile(os.path.join(T085_DIR, "evaluate_report.txt")):
        if not os.path.isfile(report):
            os.makedirs(out_dir, exist_ok=True)
            for name in os.listdir(T085_DIR):
                src = os.path.join(T085_DIR, name)
                dst = os.path.join(out_dir, name)
                if os.path.isfile(src) and not os.path.isfile(dst):
                    shutil.copy2(src, dst)
        metrics = parse_metrics(report if os.path.isfile(report) else os.path.join(T085_DIR, "evaluate_report.txt"))
        print(f"[{tag}] reuse t085 -> P={metrics['P']:.4f} R={metrics['R']:.4f} F1={metrics['F1']:.4f}")
        return {"threshold": threshold, **metrics}

    ensure_shared_cache()
    os.makedirs(out_dir, exist_ok=True)

    split_cache = os.path.join(out_dir, "qwen_split_cache.jsonl")
    if not os.path.isfile(split_cache) and os.path.isfile(SHARED_CACHE):
        shutil.copy2(SHARED_CACHE, split_cache)

    skip_split = os.path.isfile(os.path.join(out_dir, "split_sentences.jsonl")) and os.path.isfile(
        os.path.join(out_dir, "qwen_split_cache.jsonl")
    )

    cmd = [
        sys.executable,
        PIPELINE,
        "--out_dir",
        out_dir,
        "--prob_threshold",
        str(threshold),
        "--split-min-prob",
        "0.8",
        "--device",
        "cpu",
    ]
    if api_key:
        cmd.extend(["--api-key", api_key])
    if skip_split:
        cmd.append("--skip-split")

    print(f"\n>>> Running {tag}: {' '.join(cmd[:6])} ...")
    subprocess.run(cmd, check=True, cwd=SCRIPT_DIR)

    merge_cache_into_shared(out_dir)
    metrics = parse_metrics(report)
    print(f"[{tag}] P={metrics['P']:.4f} R={metrics['R']:.4f} F1={metrics['F1']:.4f}")
    return {"threshold": threshold, **metrics}


def print_table(rows: list[dict]) -> None:
    print("\n" + "=" * 60)
    print("Threshold Sweep (v5 Merge, 全量 test offset-rel-strict)")
    print("split_min_prob=0.8, include_empty_pred=False")
    print("=" * 60)
    rows_sorted = sorted(rows, key=lambda x: x["threshold"])
    print(f"{'Threshold':>10} | {'P':>8} | {'R':>8} | {'F1':>8}")
    print("-" * 42)
    for r in rows_sorted:
        print(f"{r['threshold']:>10.2f} | {r['P']:>8.4f} | {r['R']:>8.4f} | {r['F1']:>8.4f}")
    print("=" * 60)

    out_json = os.path.join(SWEEP_ROOT, "threshold_sweep_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(rows_sorted, f, ensure_ascii=False, indent=2)
    print(f"Saved: {out_json}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--api-key", default=os.environ.get("DASHSCOPE_API_KEY", ""))
    p.add_argument("--only", type=float, nargs="*", default=None)
    args = p.parse_args()
    if not args.api_key:
        # fall back to qwen.json via pipeline default
        pass

    thresholds = args.only if args.only else THRESHOLDS
    rows = []
    for t in thresholds:
        rows.append(run_one(t, args.api_key))
    print_table(rows)


if __name__ == "__main__":
    main()
