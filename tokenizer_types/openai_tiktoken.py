from __future__ import annotations

from typing import Iterable

import tiktoken

from .base import TokenizerAdapter, TokenizerSpec


class OpenAITiktokenAdapter(TokenizerAdapter):
    def __init__(self, encoding_name: str = "o200k_base"):
        self.encoding_name = encoding_name
        self.enc = tiktoken.get_encoding(encoding_name)
        self._spec = TokenizerSpec(
            name=f"openai_{encoding_name}",
            family="openai_tiktoken",
            vocab_size=self.enc.n_vocab,
        )

    @property
    def spec(self) -> TokenizerSpec:
        return self._spec

    def encode(self, text: str) -> list[int]:
        return self.enc.encode(text, allowed_special="all")

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return self.enc.encode_batch(texts, allowed_special="all")

    def decode(self, ids: Iterable[int]) -> str:
        chunks: list[bytes] = []
        for token_id in ids:
            try:
                chunks.append(self.enc.decode_single_token_bytes(int(token_id)))
            except KeyError:
                chunks.append(b"\xef\xbf\xbd")
        return b"".join(chunks).decode("utf-8", errors="replace")

    def token_repr(self, token_id: int) -> bytes:
        try:
            return self.enc.decode_single_token_bytes(int(token_id))
        except KeyError:
            return f"<INVALID:{token_id}>".encode()
