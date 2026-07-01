#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Citrus domain ablation / pipeline configuration for SUF-CSR."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

from paths import (
    CITRUS_OUT,
    GOLD_UIE,
    NER_PRED,
    PRED_E2E,
    PRED_TWO_STAGE,
    RE_SENTENCES,
    RE_TEST,
    SUF_NER_MODEL,
    SUF_RE_MODEL,
    UIE_DIR,
)

UIE_DIR = UIE_DIR
LTONG_DIR = UIE_DIR
PROJECT_ROOT = os.path.dirname(UIE_DIR)
OUTPUT_ROOT = os.path.join(UIE_DIR, "outputs")


@dataclass(frozen=True)
class AblationConfig:
    name: str
    input_txt: str
    test_file: str
    pred_e2e: str
    pred_two_stage: str
    ner_pred: str
    ablation_dir: str
    re_model_dir: str
    ner_model_dir: str
    ner_types: List[str]
    relations: List[str]
    pred_bucket: str
    schema_lang: str
    position_prob: float
    oracle_prompt_script: str
    eval_backend: str
    conll04_gold_json: str = ""
    citrus_gold_uie: str = ""
    use_scierc_re_align: bool = False


def get_config(dataset: str = "citrus") -> AblationConfig:
    ds = dataset.lower()
    if ds != "citrus":
        raise ValueError(f"Only 'citrus' is supported in SUF-CSR, got: {dataset}")

    return AblationConfig(
        name="citrus",
        input_txt=RE_SENTENCES,
        test_file=RE_TEST,
        pred_e2e=PRED_E2E,
        pred_two_stage=PRED_TWO_STAGE,
        ner_pred=NER_PRED,
        ablation_dir=os.path.join(CITRUS_OUT, "ablation"),
        re_model_dir=SUF_RE_MODEL,
        ner_model_dir=SUF_NER_MODEL,
        ner_types=[
            "citrus",
            "compound",
            "aroma",
            "taste",
            "mouth_feel",
            "location",
            "by-product",
            "health",
        ],
        relations=[
            "category",
            "contain",
            "has_aroma",
            "has_mouth_feel",
            "has_taste",
            "impact_on",
            "produce",
            "produced_in",
        ],
        pred_bucket="citrus",
        schema_lang="en",
        position_prob=0.5,
        oracle_prompt_script="",
        eval_backend="citrus_uie",
        citrus_gold_uie=GOLD_UIE,
    )
