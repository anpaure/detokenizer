from __future__ import annotations

from pathlib import Path

from tokenizer_types.openai_tiktoken import OpenAITiktokenAdapter
from tokenizer_types.trained_bpe import TrainedByteLevelBPEAdapter


def build_tokenizer(name: str, *, train_file: Path | None = None, work_dir: Path | None = None, vocab_size: int | None = None):
    if name == "openai_o200k":
        return OpenAITiktokenAdapter("o200k_base")
    if name == "openai_cl100k":
        return OpenAITiktokenAdapter("cl100k_base")
    if name == "qwen3":
        from tokenizer_types.qwen import build

        return build()
    if name == "kimi_k2":
        from tokenizer_types.kimi import build

        return build()
    if name == "gemma4_31b":
        from tokenizer_types.gemma import build

        return build()
    if name == "deepseek_v4_pro":
        from tokenizer_types.deepseek import build_v4_pro

        return build_v4_pro()
    if name == "trained_bpe":
        if train_file is None or work_dir is None or vocab_size is None:
            raise ValueError("trained_bpe requires train_file, work_dir, and vocab_size")
        return TrainedByteLevelBPEAdapter("trained_bpe", train_file, vocab_size, work_dir / f"trained_bpe_v{vocab_size}")
    raise KeyError(f"unknown tokenizer: {name}")


DEFAULT_TOKENIZERS = ["openai_o200k", "openai_cl100k", "qwen3", "kimi_k2", "gemma4_31b", "deepseek_v4_pro"]
