from __future__ import annotations

from typing import Iterable

from transformers import AutoTokenizer

from .base import TokenizerAdapter, TokenizerSpec


class HuggingFaceTokenizerAdapter(TokenizerAdapter):
    def __init__(self, name: str, model_id: str, family: str):
        self.name = name
        self.model_id = model_id
        self.family = family
        self.tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=True)
        self._spec = TokenizerSpec(name=name, family=family, vocab_size=len(self.tok))

    @property
    def spec(self) -> TokenizerSpec:
        return self._spec

    def encode(self, text: str) -> list[int]:
        if hasattr(self.tok, "model") and hasattr(self.tok.model, "encode"):
            return self.tok.model.encode(text, allowed_special="all")
        return self.tok.encode(text, add_special_tokens=False)

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        if hasattr(self.tok, "model") and hasattr(self.tok.model, "encode_batch"):
            return self.tok.model.encode_batch(texts, allowed_special="all")
        return self.tok(texts, add_special_tokens=False, return_attention_mask=False)["input_ids"]

    def decode(self, ids: Iterable[int]) -> str:
        if hasattr(self.tok, "model") and hasattr(self.tok.model, "decode"):
            return self.tok.model.decode(list(map(int, ids)))
        return self.tok.decode(list(map(int, ids)), skip_special_tokens=False, clean_up_tokenization_spaces=False)

    def token_repr(self, token_id: int) -> bytes:
        token = self.tok.convert_ids_to_tokens(int(token_id))
        if token is None:
            token = f"<INVALID:{token_id}>"
        return str(token).encode("utf-8", errors="surrogatepass")
