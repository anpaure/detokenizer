from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class TokenizerSpec:
    name: str
    family: str
    vocab_size: int


class TokenizerAdapter(ABC):
    @property
    @abstractmethod
    def spec(self) -> TokenizerSpec:
        raise NotImplementedError

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [self.encode(text) for text in texts]

    @abstractmethod
    def decode(self, ids: Iterable[int]) -> str:
        raise NotImplementedError

    @abstractmethod
    def token_repr(self, token_id: int) -> bytes:
        """Canonical comparable token representation for evaluation.

        For byte-level tokenizers this should be raw token bytes. For HF
        tokenizers it is the serialized tokenizer token string.
        """

        raise NotImplementedError


def read_text_batches(path: Path, batch_chars: int = 1_000_000, batch_items: int = 8):
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        batch: list[str] = []
        while True:
            chunk = f.read(batch_chars)
            if not chunk:
                break
            batch.append(chunk)
            if len(batch) >= batch_items:
                yield batch
                batch = []
        if batch:
            yield batch
