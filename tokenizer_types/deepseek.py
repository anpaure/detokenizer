from __future__ import annotations

from .hf_tokenizer import HuggingFaceTokenizerAdapter


def build_v4_pro():
    return HuggingFaceTokenizerAdapter(
        name="deepseek_v4_pro",
        family="deepseek",
        model_id="deepseek-ai/DeepSeek-V4-Pro",
    )
