#!/usr/bin/env python3
"""100M-token unsupervised tokenizer-recovery experiment.

This is the scalable version of the prototype in detokenizer_attack.py. It uses
Hugging Face's Rust `tokenizers` package to train two disjoint ByteLevel BPE
tokenizers, encodes a 100M-token target stream with the secret tokenizer, shuffles
the secret token IDs, and then attacks the shuffled sequence with only a
surrogate tokenizer trained on public text.

The attack is still intentionally unsupervised: the secret vocabulary and merge
table are never used by the recovery algorithm. They are used only for metrics.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
from pathlib import Path

import numpy as np
from tokenizers import ByteLevelBPETokenizer


def make_slice_file(src: Path, dst: Path, start_byte: int, byte_budget: int) -> None:
    if dst.exists() and dst.stat().st_size >= byte_budget * 0.95:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("rb") as f, dst.open("wb") as out:
        f.seek(start_byte)
        remaining = byte_budget
        while remaining > 0:
            chunk = f.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            out.write(chunk)
            remaining -= len(chunk)


def train_tokenizer(path: Path, vocab_size: int, out_dir: Path) -> ByteLevelBPETokenizer:
    vocab_file = out_dir / "vocab.json"
    merges_file = out_dir / "merges.txt"
    tok = ByteLevelBPETokenizer()
    if vocab_file.exists() and merges_file.exists():
        return ByteLevelBPETokenizer(str(vocab_file), str(merges_file))
    out_dir.mkdir(parents=True, exist_ok=True)
    tok.train(files=[str(path)], vocab_size=vocab_size, min_frequency=2, special_tokens=[])
    tok.save_model(str(out_dir))
    return ByteLevelBPETokenizer(str(vocab_file), str(merges_file))


def id_to_token(tok: ByteLevelBPETokenizer) -> list[str]:
    vocab = tok.get_vocab()
    out = [""] * len(vocab)
    for token, idx in vocab.items():
        out[idx] = token
    return out


def encode_file_limit(tok: ByteLevelBPETokenizer, path: Path, token_limit: int, dtype: np.dtype) -> np.ndarray:
    chunks: list[np.ndarray] = []
    total = 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        batch: list[str] = []
        for line in f:
            batch.append(line)
            if len(batch) < 512:
                continue
            encoded = tok.encode_batch(batch)
            for enc in encoded:
                ids = enc.ids
                if not ids:
                    continue
                need = token_limit - total
                if need <= 0:
                    break
                arr = np.asarray(ids[:need], dtype=dtype)
                chunks.append(arr)
                total += len(arr)
            batch = []
            if total >= token_limit:
                break
        if total < token_limit and batch:
            encoded = tok.encode_batch(batch)
            for enc in encoded:
                need = token_limit - total
                if need <= 0:
                    break
                arr = np.asarray(enc.ids[:need], dtype=dtype)
                chunks.append(arr)
                total += len(arr)
    if not chunks:
        return np.asarray([], dtype=dtype)
    return np.concatenate(chunks)


def bigram_matrix(ids: np.ndarray, vocab_size: int, chunk_size: int) -> np.ndarray:
    mat = np.zeros((vocab_size, vocab_size), dtype=np.int64)
    if len(ids) < 2:
        return mat
    for start in range(0, len(ids) - 1, chunk_size):
        end = min(len(ids) - 1, start + chunk_size)
        a = ids[start:end].astype(np.int64, copy=False)
        b = ids[start + 1 : end + 1].astype(np.int64, copy=False)
        flat = np.bincount(a * vocab_size + b, minlength=vocab_size * vocab_size)
        mat += flat.reshape(vocab_size, vocab_size)
    return mat


def normalize_rows(x: np.ndarray, counts: np.ndarray) -> np.ndarray:
    y = x.astype(np.float32, copy=True)
    denom = np.sqrt(np.maximum(1.0, counts.astype(np.float32)))[:, None]
    y /= denom
    norms = np.linalg.norm(y, axis=1)
    nz = norms > 0
    y[nz] /= norms[nz, None]
    return y


def frequency_mapping(cipher_counts: np.ndarray, plain_counts: np.ndarray) -> np.ndarray:
    c_order = np.argsort(-cipher_counts)
    p_order = np.argsort(-plain_counts)
    mapping = np.zeros_like(c_order)
    mapping[c_order] = p_order
    return mapping


def context_refine(
    cipher_counts: np.ndarray,
    plain_counts: np.ndarray,
    cipher_bigram: np.ndarray,
    plain_bigram: np.ndarray,
    init_mapping: np.ndarray,
    top_tokens: int,
    anchors: int,
    candidate_window: int,
    rounds: int,
    freq_weight: float,
) -> np.ndarray:
    vocab_size = len(cipher_counts)
    c_order = np.argsort(-cipher_counts)[:top_tokens]
    p_order_all = np.argsort(-plain_counts)
    p_rank = np.empty(vocab_size, dtype=np.int32)
    p_rank[p_order_all] = np.arange(vocab_size, dtype=np.int32)
    mapping = init_mapping.copy()

    c_log = np.log(np.maximum(cipher_counts, 1) / max(1, int(cipher_counts.sum())))
    p_log = np.log(np.maximum(plain_counts, 1) / max(1, int(plain_counts.sum())))

    for _ in range(rounds):
        anchor_cipher = c_order[:anchors]
        anchor_plain = mapping[anchor_cipher]

        c_features = np.concatenate(
            [
                cipher_bigram[np.ix_(anchor_cipher, c_order)].T,
                cipher_bigram[np.ix_(c_order, anchor_cipher)],
            ],
            axis=1,
        )
        p_focus = p_order_all[:top_tokens]
        p_features = np.concatenate(
            [
                plain_bigram[np.ix_(anchor_plain, p_focus)].T,
                plain_bigram[np.ix_(p_focus, anchor_plain)],
            ],
            axis=1,
        )
        c_vec = normalize_rows(c_features, cipher_counts[c_order])
        p_vec = normalize_rows(p_features, plain_counts[p_focus])
        p_focus_index = {int(p): i for i, p in enumerate(p_focus)}

        edges: list[tuple[float, int, int]] = []
        for ci, c in enumerate(c_order):
            rank_center = int(p_rank[mapping[c]])
            lo = max(0, rank_center - candidate_window)
            hi = min(vocab_size, rank_center + candidate_window + 1)
            for p in p_order_all[lo:hi]:
                pi = p_focus_index.get(int(p))
                if pi is None:
                    continue
                score = float(np.dot(c_vec[ci], p_vec[pi])) - freq_weight * abs(float(c_log[c] - p_log[p]))
                edges.append((score, int(c), int(p)))
        edges.sort(reverse=True)

        new_mapping = mapping.copy()
        used_c: set[int] = set()
        used_p: set[int] = set()
        for _, c, p in edges:
            if c in used_c or p in used_p:
                continue
            new_mapping[c] = p
            used_c.add(c)
            used_p.add(p)
            if len(used_c) >= len(c_order):
                break
        mapping = new_mapping
    return mapping


def weighted_token_string_accuracy(
    cipher_counts: np.ndarray,
    mapping: np.ndarray,
    inv_perm: np.ndarray,
    secret_tokens: list[str],
    surrogate_tokens: list[str],
) -> float:
    total = int(cipher_counts.sum())
    ok = 0
    for c, n in enumerate(cipher_counts):
        if n == 0:
            continue
        if secret_tokens[int(inv_perm[c])] == surrogate_tokens[int(mapping[c])]:
            ok += int(n)
    return ok / max(1, total)


def oracle_overlap(
    cipher_counts: np.ndarray,
    inv_perm: np.ndarray,
    secret_tokens: list[str],
    surrogate_tokens: list[str],
) -> float:
    surrogate_set = set(surrogate_tokens)
    total = int(cipher_counts.sum())
    covered = 0
    for c, n in enumerate(cipher_counts):
        if n and secret_tokens[int(inv_perm[c])] in surrogate_set:
            covered += int(n)
    return covered / max(1, total)


def char_error_rate(a: str, b: str, max_chars: int) -> float:
    a = a[:max_chars]
    b = b[:max_chars]
    if not a:
        return 1.0 if b else 0.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] / len(a)


def gzip_ratio(text: str) -> float:
    data = text.encode("utf-8", errors="ignore")
    return len(gzip.compress(data, compresslevel=6)) / max(1, len(data))


def decode_prefix(tok: ByteLevelBPETokenizer, ids: np.ndarray, n: int) -> str:
    return tok.decode(ids[:n].astype(int).tolist())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-file", default=".cache/fineweb_100m.txt")
    parser.add_argument("--work-dir", default=".cache/bpe_100m")
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--train-bytes", type=int, default=96_000_000)
    parser.add_argument("--surrogate-start-byte", type=int, default=240_000_000)
    parser.add_argument("--reference-start-byte", type=int, default=320_000_000)
    parser.add_argument("--reference-bytes", type=int, default=112_000_000)
    parser.add_argument("--target-tokens", type=int, default=100_000_000)
    parser.add_argument("--reference-tokens", type=int, default=100_000_000)
    parser.add_argument("--top-tokens", type=int, default=900)
    parser.add_argument("--anchors", type=int, default=220)
    parser.add_argument("--candidate-window", type=int, default=120)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--freq-weight", type=float, default=0.20)
    parser.add_argument("--bigram-chunk", type=int, default=5_000_000)
    parser.add_argument("--preview-tokens", type=int, default=20_000)
    parser.add_argument("--eval-chars", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--save-arrays-dir", default="")
    parser.add_argument("--out", default="results_100m.json")
    args = parser.parse_args()

    text_file = Path(args.text_file)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    secret_train = work_dir / "secret_train.txt"
    surrogate_train = work_dir / "surrogate_train.txt"
    surrogate_reference = work_dir / "surrogate_reference.txt"
    make_slice_file(text_file, secret_train, 0, args.train_bytes)
    make_slice_file(text_file, surrogate_train, args.surrogate_start_byte, args.train_bytes)
    make_slice_file(text_file, surrogate_reference, args.reference_start_byte, args.reference_bytes)

    print(f"training secret tokenizer on {secret_train.stat().st_size:,} bytes")
    secret = train_tokenizer(secret_train, args.vocab_size, work_dir / f"secret_v{args.vocab_size}")
    print(f"training surrogate tokenizer on {surrogate_train.stat().st_size:,} bytes")
    surrogate = train_tokenizer(surrogate_train, args.vocab_size, work_dir / f"surrogate_v{args.vocab_size}")

    dtype = np.uint16 if args.vocab_size <= 65535 else np.uint32
    print(f"encoding target up to {args.target_tokens:,} secret tokens")
    secret_ids = encode_file_limit(secret, text_file, args.target_tokens, dtype)
    print(f"secret_target_tokens={len(secret_ids):,}")
    print(f"encoding reference up to {args.reference_tokens:,} surrogate tokens from {surrogate_reference}")
    reference_ids = encode_file_limit(surrogate, surrogate_reference, args.reference_tokens, dtype)
    print(f"surrogate_reference_tokens={len(reference_ids):,}")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(args.vocab_size).astype(dtype)
    inv_perm = np.empty(args.vocab_size, dtype=dtype)
    inv_perm[perm] = np.arange(args.vocab_size, dtype=dtype)
    cipher_ids = perm[secret_ids]

    cipher_counts = np.bincount(cipher_ids.astype(np.int64), minlength=args.vocab_size)
    plain_counts = np.bincount(reference_ids.astype(np.int64), minlength=args.vocab_size)
    print("building bigram matrices")
    cipher_bigram = bigram_matrix(cipher_ids, args.vocab_size, args.bigram_chunk)
    plain_bigram = bigram_matrix(reference_ids, args.vocab_size, args.bigram_chunk)

    freq_map = frequency_mapping(cipher_counts, plain_counts)
    refined_map = context_refine(
        cipher_counts,
        plain_counts,
        cipher_bigram,
        plain_bigram,
        freq_map,
        top_tokens=args.top_tokens,
        anchors=args.anchors,
        candidate_window=args.candidate_window,
        rounds=args.rounds,
        freq_weight=args.freq_weight,
    )

    if args.save_arrays_dir:
        arrays_dir = Path(args.save_arrays_dir)
        arrays_dir.mkdir(parents=True, exist_ok=True)
        np.save(arrays_dir / "secret_ids.npy", secret_ids)
        np.save(arrays_dir / "cipher_ids.npy", cipher_ids)
        np.save(arrays_dir / "reference_ids.npy", reference_ids)
        np.save(arrays_dir / "freq_map.npy", freq_map)
        np.save(arrays_dir / "refined_map.npy", refined_map)
        np.save(arrays_dir / "perm.npy", perm)
        np.save(arrays_dir / "inv_perm.npy", inv_perm)
        np.save(arrays_dir / "cipher_counts.npy", cipher_counts)
        np.save(arrays_dir / "plain_counts.npy", plain_counts)
        (arrays_dir / "secret_tokens.json").write_text(json.dumps(id_to_token(secret), ensure_ascii=False), encoding="utf-8")
        (arrays_dir / "surrogate_tokens.json").write_text(json.dumps(id_to_token(surrogate), ensure_ascii=False), encoding="utf-8")

    secret_tok = id_to_token(secret)
    surrogate_tok = id_to_token(surrogate)
    true_prefix = decode_prefix(secret, secret_ids, args.preview_tokens)
    freq_prefix = decode_prefix(surrogate, freq_map[cipher_ids[: args.preview_tokens]], args.preview_tokens)
    refined_prefix = decode_prefix(surrogate, refined_map[cipher_ids[: args.preview_tokens]], args.preview_tokens)

    metrics = {
        "text_file": str(text_file),
        "vocab_size": args.vocab_size,
        "secret_train_bytes": secret_train.stat().st_size,
        "surrogate_train_bytes": surrogate_train.stat().st_size,
        "surrogate_reference_bytes": surrogate_reference.stat().st_size,
        "target_tokens_observed": int(len(secret_ids)),
        "reference_tokens": int(len(reference_ids)),
        "observed_unique_cipher_ids": int(np.count_nonzero(cipher_counts)),
        "oracle_surrogate_string_coverage_weighted": oracle_overlap(cipher_counts, inv_perm, secret_tok, surrogate_tok),
        "frequency_weighted_exact_token_string_accuracy": weighted_token_string_accuracy(cipher_counts, freq_map, inv_perm, secret_tok, surrogate_tok),
        "refined_weighted_exact_token_string_accuracy": weighted_token_string_accuracy(cipher_counts, refined_map, inv_perm, secret_tok, surrogate_tok),
        "frequency_char_error_rate_prefix": char_error_rate(true_prefix, freq_prefix, args.eval_chars),
        "refined_char_error_rate_prefix": char_error_rate(true_prefix, refined_prefix, args.eval_chars),
        "true_gzip_ratio_prefix": gzip_ratio(true_prefix[: args.eval_chars]),
        "frequency_gzip_ratio_prefix": gzip_ratio(freq_prefix[: args.eval_chars]),
        "refined_gzip_ratio_prefix": gzip_ratio(refined_prefix[: args.eval_chars]),
        "true_prefix": true_prefix[:1000].replace("\n", "\\n"),
        "frequency_prefix": freq_prefix[:1000].replace("\n", "\\n"),
        "refined_prefix": refined_prefix[:1000].replace("\n", "\\n"),
    }
    Path(args.out).write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
