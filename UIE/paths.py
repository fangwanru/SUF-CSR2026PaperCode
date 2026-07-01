#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SUF-CSR project path constants."""

from __future__ import annotations

import os

UIE_DIR = os.path.dirname(os.path.abspath(__file__))
SUF_ROOT = os.path.dirname(UIE_DIR)

CITRUS_NER_DIR = os.path.join(SUF_ROOT, "CitrusNER")
CITRUS_RE_DIR = os.path.join(SUF_ROOT, "CitrusRE")
LLM_CONFIG_DIR = os.path.join(SUF_ROOT, "LLM", "config")

OUTPUT_ROOT = os.path.join(UIE_DIR, "outputs")
CHECKPOINT_ROOT = os.path.join(UIE_DIR, "checkpoints")

UIE_BASE_MODEL = os.path.join(CHECKPOINT_ROOT, "uie-base-en")
SUF_NER_MODEL = os.path.join(CHECKPOINT_ROOT, "SUF-NER", "model_best")
SUF_RE_MODEL = os.path.join(CHECKPOINT_ROOT, "SUF-RE", "model_best")

QWEN_CONFIG = os.path.join(LLM_CONFIG_DIR, "qwen.json")

# Dataset files (relative to SUF-CSR root)
NER_TRAIN = os.path.join(CITRUS_NER_DIR, "train.txt")
NER_DEV = os.path.join(CITRUS_NER_DIR, "dev.txt")
NER_TEST = os.path.join(CITRUS_NER_DIR, "test.txt")
NER_JSONL = os.path.join(CITRUS_NER_DIR, "CitrusNER.jsonl")

RE_TRAIN = os.path.join(CITRUS_RE_DIR, "train.txt")
RE_DEV = os.path.join(CITRUS_RE_DIR, "dev.txt")
RE_TEST = os.path.join(CITRUS_RE_DIR, "test.txt")
RE_SENTENCES = os.path.join(CITRUS_RE_DIR, "test_sentences.txt")
RE_JSONL = os.path.join(CITRUS_RE_DIR, "CitrusRE.jsonl")

# Experiment outputs
CITRUS_OUT = os.path.join(OUTPUT_ROOT, "citrus")
GOLD_UIE = os.path.join(CITRUS_OUT, "uie_official", "gold", "test.json")
PRED_E2E = os.path.join(CITRUS_OUT, "predict_e2e.jsonl")
PRED_TWO_STAGE = os.path.join(CITRUS_OUT, "predict_two_stage.jsonl")
NER_PRED = os.path.join(CITRUS_OUT, "coupled_NER.jsonl")
SIMPLIFIED_RE = os.path.join(CITRUS_OUT, "coupled_RE_simplified.jsonl")
CSR_OUT = os.path.join(CITRUS_OUT, "suf_csr")
THRESHOLD_SWEEP_OUT = os.path.join(CITRUS_OUT, "threshold_sweep")
