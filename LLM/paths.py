"""Path constants for LLM experiments."""

import os

LLM_DIR = os.path.dirname(os.path.abspath(__file__))
SUF_ROOT = os.path.dirname(LLM_DIR)

CITRUS_NER = os.path.join(SUF_ROOT, "CitrusNER")
CITRUS_RE = os.path.join(SUF_ROOT, "CitrusRE")
CONFIG_DIR = os.path.join(LLM_DIR, "config")

QWEN_CONFIG = os.path.join(CONFIG_DIR, "qwen.json")
DEEPSEEK_CONFIG = os.path.join(CONFIG_DIR, "deepseek.json")

NER_TEST = os.path.join(CITRUS_NER, "test.txt")
NER_JSONL = os.path.join(CITRUS_NER, "CitrusNER.jsonl")

RE_TEST = os.path.join(CITRUS_RE, "test.txt")
RE_SENTENCES = os.path.join(CITRUS_RE, "test_sentences.txt")
RE_JSONL = os.path.join(CITRUS_RE, "CitrusRE.jsonl")

OUT_NER = os.path.join(LLM_DIR, "NER")
OUT_RE = os.path.join(LLM_DIR, "RE")
OUT_TWO = os.path.join(LLM_DIR, "two_stage")
