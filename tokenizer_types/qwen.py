from __future__ import annotations

from .hf_tokenizer import HuggingFaceTokenizerAdapter


def build():
    return HuggingFaceTokenizerAdapter(
        name="qwen3_0_6b",
        family="qwen",
        model_id="Qwen/Qwen3-0.6B",
    )
