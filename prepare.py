"""Fixed data prep and evaluation for detokenizer autoresearch.

The mutable experiment lives in train.py. This file owns:

- public reference/target text materialization from FineWeb shards
- tokenizer adapters and cached tokenized streams
- shuffled-ID fixture creation
- fixed oracle metrics for controlled experiments

Agents should not edit this file during a hillclimb run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow.parquet as pq
import tiktoken
from huggingface_hub import hf_hub_download
from rapidfuzz.distance import Levenshtein
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Fixed task defaults
# ---------------------------------------------------------------------------

CACHE_DIR = Path(os.path.expanduser("~")) / ".cache" / "detokenizer-autoresearch"
HF_CACHE_DIR = CACHE_DIR / "hf"
TEXT_DIR = CACHE_DIR / "text"
FIXTURE_DIR = CACHE_DIR / "fixtures"
REF_IDS_DIR = CACHE_DIR / "reference_ids"

FINEWEB_REPO = "HuggingFaceFW/fineweb"
TARGET_SHARD = "sample/10BT/014_00000.parquet"
REFERENCE_SHARD = "sample/10BT/013_00000.parquet"
DEFAULT_TEXT_TOKENS = 100_000_000
DEFAULT_TARGET_TOKENS = 1_000_000
DEFAULT_REFERENCE_TOKENS = 100_000_000
DEFAULT_SEED = 11
DEFAULT_SAMPLE_TOKENS = 500_000
DEFAULT_EVAL_CHARS = 50_000


@dataclass(frozen=True)
class TokenizerSpec:
    name: str
    family: str
    vocab_size: int


class TokenizerAdapter:
    spec: TokenizerSpec

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [self.encode(text) for text in texts]

    def decode(self, ids: Iterable[int]) -> str:
        raise NotImplementedError

    def token_bytes(self, token_id: int) -> bytes:
        raise NotImplementedError


class OpenAITiktokenAdapter(TokenizerAdapter):
    def __init__(self, encoding_name: str):
        self.encoding_name = encoding_name
        self.enc = tiktoken.get_encoding(encoding_name)
        self.spec = TokenizerSpec(
            name=f"openai_{encoding_name}",
            family="openai_tiktoken",
            vocab_size=self.enc.n_vocab,
        )

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

    def token_bytes(self, token_id: int) -> bytes:
        try:
            return self.enc.decode_single_token_bytes(int(token_id))
        except KeyError:
            return f"<INVALID:{token_id}>".encode()


class HuggingFaceTokenizerAdapter(TokenizerAdapter):
    def __init__(self, name: str, family: str, model_id: str):
        self.model_id = model_id
        self.tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=True)
        self.spec = TokenizerSpec(name=name, family=family, vocab_size=len(self.tok))

    def encode(self, text: str) -> list[int]:
        model = getattr(self.tok, "model", None)
        if hasattr(model, "encode"):
            return model.encode(text, allowed_special="all")
        return self.tok.encode(text, add_special_tokens=False)

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        model = getattr(self.tok, "model", None)
        if hasattr(model, "encode_batch"):
            return model.encode_batch(texts, allowed_special="all")
        return self.tok(texts, add_special_tokens=False, return_attention_mask=False)["input_ids"]

    def decode(self, ids: Iterable[int]) -> str:
        model = getattr(self.tok, "model", None)
        ids_list = list(map(int, ids))
        if hasattr(model, "decode"):
            return model.decode(ids_list)
        return self.tok.decode(ids_list, skip_special_tokens=False, clean_up_tokenization_spaces=False)

    def token_bytes(self, token_id: int) -> bytes:
        model = getattr(self.tok, "model", None)
        if hasattr(model, "decode_single_token_bytes"):
            try:
                return model.decode_single_token_bytes(int(token_id))
            except KeyError:
                return f"<INVALID:{token_id}>".encode()
        token = self.tok.convert_ids_to_tokens(int(token_id))
        if token is None:
            token = f"<INVALID:{token_id}>"
        return str(token).encode("utf-8", errors="surrogatepass")


def build_tokenizer(name: str) -> TokenizerAdapter:
    if name == "openai_o200k":
        return OpenAITiktokenAdapter("o200k_base")
    if name == "openai_cl100k":
        return OpenAITiktokenAdapter("cl100k_base")
    if name == "qwen3":
        return HuggingFaceTokenizerAdapter("qwen3_0_6b", "qwen", "Qwen/Qwen3-0.6B")
    if name == "kimi_k2":
        return HuggingFaceTokenizerAdapter("kimi_k2_instruct", "kimi", "moonshotai/Kimi-K2-Instruct")
    if name == "gemma4_31b":
        return HuggingFaceTokenizerAdapter("gemma4_31b", "gemma", "google/gemma-4-31B")
    if name == "deepseek_v4_pro":
        return HuggingFaceTokenizerAdapter("deepseek_v4_pro", "deepseek", "deepseek-ai/DeepSeek-V4-Pro")
    raise KeyError(f"unknown tokenizer: {name}")


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


def fineweb_text_path(kind: str, text_tokens: int = DEFAULT_TEXT_TOKENS) -> Path:
    return TEXT_DIR / f"fineweb_{kind}_{text_tokens}.txt"


def download_fineweb_text(kind: str, shard: str, target_tokens: int = DEFAULT_TEXT_TOKENS) -> Path:
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    out_path = fineweb_text_path(kind, target_tokens)
    meta_path = out_path.with_suffix(".meta.json")
    if out_path.exists() and meta_path.exists():
        return out_path
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    parquet_path = hf_hub_download(
        repo_id=FINEWEB_REPO,
        repo_type="dataset",
        filename=shard,
        cache_dir=str(HF_CACHE_DIR),
    )
    rows = 0
    tokens = 0
    bytes_written = 0
    with out_path.open("w", encoding="utf-8", errors="ignore") as out:
        parquet = pq.ParquetFile(parquet_path)
        for batch in parquet.iter_batches(batch_size=2048, columns=["text", "token_count", "language"]):
            texts = batch.column("text").to_pylist()
            token_counts = batch.column("token_count").to_pylist()
            languages = batch.column("language").to_pylist()
            for text, token_count, language in zip(texts, token_counts, languages):
                if language and language != "en":
                    continue
                if not text:
                    continue
                out.write(text)
                out.write("\n\n")
                rows += 1
                tokens += int(token_count or 0)
                bytes_written += len(text.encode("utf-8", errors="ignore")) + 2
                if tokens >= target_tokens:
                    break
            if tokens >= target_tokens:
                break
    meta = {
        "repo": FINEWEB_REPO,
        "shard": shard,
        "parquet_path": parquet_path,
        "rows": rows,
        "dataset_token_count_sum": tokens,
        "bytes_written": bytes_written,
        "target_tokens": target_tokens,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out_path


def encode_text(adapter: TokenizerAdapter, text_path: Path, token_limit: int | None = None) -> np.ndarray:
    chunks: list[np.ndarray] = []
    total = 0
    for batch in read_text_batches(text_path):
        for ids in adapter.encode_batch(batch):
            if not ids:
                continue
            if token_limit is not None:
                need = token_limit - total
                if need <= 0:
                    break
                ids = ids[:need]
            arr = np.asarray(ids, dtype=np.uint32)
            chunks.append(arr)
            total += len(arr)
        if token_limit is not None and total >= token_limit:
            break
    if not chunks:
        return np.empty(0, dtype=np.uint32)
    return np.concatenate(chunks)


def reference_cache_path(adapter: TokenizerAdapter, ref_path: Path, token_limit: int) -> Path:
    stat = ref_path.stat()
    key = {
        "cache_version": 1,
        "tokenizer": adapter.spec.name,
        "family": adapter.spec.family,
        "vocab_size": adapter.spec.vocab_size,
        "reference_path": str(ref_path.resolve()),
        "reference_size": stat.st_size,
        "reference_mtime_ns": stat.st_mtime_ns,
        "token_limit": token_limit,
    }
    digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode("utf-8")).hexdigest()[:24]
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", adapter.spec.name)
    return REF_IDS_DIR / f"{safe_name}_{token_limit}_{digest}.npy"


def encode_reference(adapter: TokenizerAdapter, ref_path: Path, token_limit: int) -> np.ndarray:
    cache_path = reference_cache_path(adapter, ref_path, token_limit)
    if cache_path.exists():
        print(f"loading cached reference ids {cache_path}", flush=True)
        return np.load(cache_path, mmap_mode="r")
    REF_IDS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ids = encode_text(adapter, ref_path, token_limit)
    tmp_path = cache_path.with_suffix(".tmp.npy")
    np.save(tmp_path, ids)
    tmp_path.replace(cache_path)
    cache_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "tokenizer": adapter.spec.name,
                "reference_text": str(ref_path),
                "token_limit": token_limit,
                "num_tokens": int(len(ids)),
                "elapsed_seconds": time.time() - t0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote cached reference ids {cache_path} tokens={len(ids):,}", flush=True)
    return np.load(cache_path, mmap_mode="r")


def fixture_paths(source_name: str, target_tokens: int, seed: int) -> dict[str, Path]:
    stem = f"{source_name}_{target_tokens}_seed{seed}"
    return {
        "secret": FIXTURE_DIR / f"{stem}.secret_ids.npy",
        "cipher": FIXTURE_DIR / f"{stem}.cipher_ids.npy",
        "perm": FIXTURE_DIR / f"{stem}.perm.npy",
        "meta": FIXTURE_DIR / f"{stem}.json",
    }


def make_fixture(
    source_adapter: TokenizerAdapter,
    target_text: Path,
    target_tokens: int,
    seed: int = DEFAULT_SEED,
) -> dict[str, np.ndarray | Path]:
    paths = fixture_paths(source_adapter.spec.name, target_tokens, seed)
    if paths["secret"].exists() and paths["cipher"].exists() and paths["perm"].exists():
        return {
            "secret": np.load(paths["secret"], mmap_mode="r"),
            "cipher": np.load(paths["cipher"], mmap_mode="r"),
            "perm": np.load(paths["perm"], mmap_mode="r"),
            **paths,
        }
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    secret = encode_text(source_adapter, target_text, target_tokens)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(source_adapter.spec.vocab_size).astype(np.uint32)
    cipher = perm[secret]
    np.save(paths["secret"], secret)
    np.save(paths["cipher"], cipher)
    np.save(paths["perm"], perm)
    paths["meta"].write_text(
        json.dumps(
            {
                "source_tokenizer": source_adapter.spec.name,
                "source_vocab_size": source_adapter.spec.vocab_size,
                "target_text": str(target_text),
                "target_tokens_requested": target_tokens,
                "target_tokens_observed": int(len(secret)),
                "seed": seed,
                "elapsed_seconds": time.time() - t0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"secret": secret, "cipher": cipher, "perm": perm, **paths}


class ByteNgramLM:
    def __init__(self, order: int = 4, alpha: float = 0.05):
        self.order = order
        self.alpha = alpha
        self.context_counts: list[dict[tuple[int, ...], int]] = [dict() for _ in range(order)]
        self.next_counts: list[dict[tuple[int, ...], dict[int, int]]] = [dict() for _ in range(order)]

    def train(self, data: bytes) -> None:
        padded = bytes([0]) * (self.order - 1) + data
        for pos in range(self.order - 1, len(padded)):
            nxt = padded[pos]
            for n in range(self.order):
                ctx = tuple(padded[pos - n : pos]) if n else ()
                self.context_counts[n][ctx] = self.context_counts[n].get(ctx, 0) + 1
                bucket = self.next_counts[n].setdefault(ctx, {})
                bucket[nxt] = bucket.get(nxt, 0) + 1

    def bits_per_byte(self, data: bytes) -> float:
        if not data:
            return 99.0
        padded = bytes([0]) * (self.order - 1) + data
        bits = 0.0
        for pos in range(self.order - 1, len(padded)):
            nxt = padded[pos]
            for n in range(self.order - 1, -1, -1):
                ctx = tuple(padded[pos - n : pos]) if n else ()
                total = self.context_counts[n].get(ctx, 0)
                if total:
                    count = self.next_counts[n].get(ctx, {}).get(nxt, 0)
                    bits -= math.log2((count + self.alpha) / (total + self.alpha * 256))
                    break
            else:
                bits += 8.0
        return bits / len(data)


def text_metrics(text: str, lm: ByteNgramLM) -> dict[str, float]:
    data = text.encode("utf-8", errors="replace")
    replacement = text.count("\ufffd") / max(1, len(text))
    printable = sum(1 for ch in text if ch.isprintable() or ch in "\n\r\t") / max(1, len(text))
    return {
        "byte_lm_bpb": lm.bits_per_byte(data),
        "replacement_rate": replacement,
        "printable_rate": printable,
    }


def char_error_rate(original: str, recovered: str, max_chars: int = DEFAULT_EVAL_CHARS) -> float:
    original = original[:max_chars]
    recovered = recovered[:max_chars]
    if not original:
        return 1.0 if recovered else 0.0
    return Levenshtein.distance(original, recovered) / len(original)


@dataclass
class Task:
    source_adapter: TokenizerAdapter
    target_adapter: TokenizerAdapter
    target_text: Path
    reference_text: Path
    secret_ids: np.ndarray
    cipher_ids: np.ndarray
    perm: np.ndarray
    ref_ids: np.ndarray
    byte_lm: ByteNgramLM


def load_task(
    source_tokenizer: str,
    target_tokenizer: str,
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    reference_tokens: int = DEFAULT_REFERENCE_TOKENS,
    seed: int = DEFAULT_SEED,
    lm_bytes: int = 16_000_000,
) -> Task:
    target_text = download_fineweb_text("target", TARGET_SHARD, DEFAULT_TEXT_TOKENS)
    reference_text = download_fineweb_text("reference", REFERENCE_SHARD, DEFAULT_TEXT_TOKENS)
    source_adapter = build_tokenizer(source_tokenizer)
    target_adapter = build_tokenizer(target_tokenizer)
    fixture = make_fixture(source_adapter, target_text, target_tokens, seed)
    ref_ids = encode_reference(target_adapter, reference_text, reference_tokens)
    lm = ByteNgramLM(order=4)
    lm.train(reference_text.read_bytes()[:lm_bytes])
    return Task(
        source_adapter=source_adapter,
        target_adapter=target_adapter,
        target_text=target_text,
        reference_text=reference_text,
        secret_ids=fixture["secret"],  # type: ignore[arg-type]
        cipher_ids=fixture["cipher"],  # type: ignore[arg-type]
        perm=fixture["perm"],  # type: ignore[arg-type]
        ref_ids=ref_ids,
        byte_lm=lm,
    )


def evaluate_recovery(task: Task, recovered_text: str, sample_tokens: int = DEFAULT_SAMPLE_TOKENS) -> dict[str, float]:
    original = task.source_adapter.decode(task.secret_ids[:sample_tokens].astype(int).tolist())
    metrics = text_metrics(recovered_text, task.byte_lm)
    metrics["cer50k"] = char_error_rate(original, recovered_text, DEFAULT_EVAL_CHARS)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare fixed detokenizer autoresearch fixtures")
    parser.add_argument("--source-tokenizer", default=os.environ.get("DETOK_SOURCE", "kimi_k2"))
    parser.add_argument("--target-tokenizer", default=os.environ.get("DETOK_TARGET", "openai_o200k"))
    parser.add_argument("--target-tokens", type=int, default=int(os.environ.get("DETOK_TARGET_TOKENS", DEFAULT_TARGET_TOKENS)))
    parser.add_argument("--reference-tokens", type=int, default=int(os.environ.get("DETOK_REFERENCE_TOKENS", DEFAULT_REFERENCE_TOKENS)))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    task = load_task(
        args.source_tokenizer,
        args.target_tokenizer,
        target_tokens=args.target_tokens,
        reference_tokens=args.reference_tokens,
        seed=args.seed,
    )
    print("ready")
    print(f"source_tokenizer:   {task.source_adapter.spec.name}")
    print(f"target_tokenizer:   {task.target_adapter.spec.name}")
    print(f"cipher_tokens:      {len(task.cipher_ids):,}")
    print(f"reference_tokens:   {len(task.ref_ids):,}")
    print(f"cache_dir:          {CACHE_DIR}")


if __name__ == "__main__":
    main()
