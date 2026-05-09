from __future__ import annotations

from .hf_tokenizer import HuggingFaceTokenizerAdapter


def build():
    return HuggingFaceTokenizerAdapter(
        name="gemma4_31b",
        family="gemma",
        model_id="google/gemma-4-31B",
    )
