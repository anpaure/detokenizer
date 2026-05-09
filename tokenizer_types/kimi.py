from __future__ import annotations

from .hf_tokenizer import HuggingFaceTokenizerAdapter


def build():
    return HuggingFaceTokenizerAdapter(
        name="kimi_k2_instruct",
        family="kimi",
        model_id="moonshotai/Kimi-K2-Instruct",
    )
