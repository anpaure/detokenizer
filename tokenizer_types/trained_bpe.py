from __future__ import annotations

from pathlib import Path
from typing import Iterable

from tokenizers import ByteLevelBPETokenizer

from .base import TokenizerAdapter, TokenizerSpec


class TrainedByteLevelBPEAdapter(TokenizerAdapter):
    def __init__(self, name: str, train_file: Path, vocab_size: int, out_dir: Path):
        self.name = name
        self.out_dir = out_dir
        vocab_file = out_dir / "vocab.json"
        merges_file = out_dir / "merges.txt"
        if vocab_file.exists() and merges_file.exists():
            self.tok = ByteLevelBPETokenizer(str(vocab_file), str(merges_file))
        else:
            out_dir.mkdir(parents=True, exist_ok=True)
            tok = ByteLevelBPETokenizer()
            tok.train(files=[str(train_file)], vocab_size=vocab_size, min_frequency=2, special_tokens=[])
            tok.save_model(str(out_dir))
            self.tok = ByteLevelBPETokenizer(str(vocab_file), str(merges_file))
        self._vocab = self.tok.get_vocab()
        self._id_to_token = [""] * len(self._vocab)
        for token, idx in self._vocab.items():
            self._id_to_token[idx] = token
        self._spec = TokenizerSpec(name=name, family="trained_bytelevel_bpe", vocab_size=len(self._id_to_token))

    @property
    def spec(self) -> TokenizerSpec:
        return self._spec

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text).ids

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [enc.ids for enc in self.tok.encode_batch(texts)]

    def decode(self, ids: Iterable[int]) -> str:
        return self.tok.decode(list(map(int, ids)))

    def token_repr(self, token_id: int) -> bytes:
        return self._id_to_token[int(token_id)].encode("utf-8", errors="surrogatepass")
