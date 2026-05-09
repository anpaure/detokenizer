#!/usr/bin/env python3
"""Plug-and-play recovery from token IDs to text.

Usage:
    uv run python recover_text.py --ids token_ids.txt --out recovered.txt

The script accepts whitespace/comma separated integer IDs, JSON arrays, or
`.npy` arrays. It tries two regimes:

1. Native IDs: decode directly with known tokenizer candidates and rank by a
   byte-level language-model score trained on public reference text.
2. Shuffled IDs: align the unknown token stream to each candidate codebook
   using unigram and left/right-context graph matching, then rank decoded
   samples by byte-LM score.

This cannot make arbitrary unknown tokenizers identifiable from integers alone.
It operationalizes the useful prior: the tokenizer is BPE-ish or close to a
frontier tokenizer, and public text is representative enough to build a target
language prior.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import random
import re
from pathlib import Path
from typing import Iterable

import numpy as np

from download_fineweb_100m import main as _unused_download_main  # noqa: F401
from tokenizer_registry import DEFAULT_TOKENIZERS, build_tokenizer
from tokenizer_types.base import TokenizerAdapter, read_text_batches
from tokenizer_types.trained_bpe import TrainedByteLevelBPEAdapter


ID_RE = re.compile(r"-?\d+")


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


def ensure_reference_text(path: Path, min_bytes: int) -> Path:
    if path.exists() and path.stat().st_size >= min_bytes:
        return path
    print(f"reference text missing or too small; downloading FineWeb slice to {path}", flush=True)
    from download_fineweb_100m import main as download_main
    import sys

    old_argv = sys.argv
    try:
        sys.argv = [
            "download_fineweb_100m.py",
            "--out",
            str(path),
            "--meta-out",
            str(path.with_suffix(".meta.json")),
            "--target-tokens",
            "100000000",
        ]
        download_main()
    finally:
        sys.argv = old_argv
    return path


def load_ids(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path).astype(np.int64, copy=False)
    text = path.read_text(encoding="utf-8", errors="ignore")
    if path.suffix == ".json":
        obj = json.loads(text)
        return np.asarray(obj, dtype=np.int64)
    return np.asarray([int(x) for x in ID_RE.findall(text)], dtype=np.int64)


def write_ids(path: Path, ids: np.ndarray) -> None:
    if path.suffix == ".npy":
        np.save(path, ids)
    else:
        path.write_text(" ".join(map(str, ids.tolist())) + "\n", encoding="utf-8")


def reference_cache_path(adapter: TokenizerAdapter, ref_path: Path, token_limit: int) -> Path:
    stat = ref_path.stat()
    key = {
        "cache_version": 2,
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
    return Path(".cache/reference_ids") / f"{safe_name}_{token_limit}_{digest}.npy"


def encode_reference(adapter: TokenizerAdapter, ref_path: Path, token_limit: int) -> np.ndarray:
    cache_path = reference_cache_path(adapter, ref_path, token_limit)
    if cache_path.exists():
        print(f"loading cached reference ids {cache_path}", flush=True)
        return np.load(cache_path, mmap_mode="r")

    chunks: list[np.ndarray] = []
    total = 0
    dtype = np.uint32 if adapter.spec.vocab_size <= np.iinfo(np.uint32).max else np.int64
    for batch in read_text_batches(ref_path):
        for ids in adapter.encode_batch(batch):
            if not ids:
                continue
            need = token_limit - total
            if need <= 0:
                break
            arr = np.asarray(ids[:need], dtype=dtype)
            chunks.append(arr)
            total += len(arr)
        if total >= token_limit:
            break
    if not chunks:
        return np.asarray([], dtype=np.int64)
    out = np.concatenate(chunks)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(".tmp.npy")
    np.save(tmp_path, out)
    tmp_path.replace(cache_path)
    meta = {
        "tokenizer": adapter.spec.name,
        "reference_text": str(ref_path),
        "token_limit": token_limit,
        "tokens_encoded": int(len(out)),
        "dtype": str(out.dtype),
        "cache": str(cache_path),
    }
    cache_path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote cached reference ids {cache_path} tokens={len(out):,}", flush=True)
    return out


def reference_prefix(ref_path: Path, work_dir: Path, byte_limit: int) -> Path:
    out = work_dir / f"reference_prefix_{byte_limit}.txt"
    if out.exists() and out.stat().st_size >= min(byte_limit, ref_path.stat().st_size):
        return out
    with ref_path.open("rb") as src, out.open("wb") as dst:
        remaining = byte_limit
        while remaining > 0:
            chunk = src.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            dst.write(chunk)
            remaining -= len(chunk)
    return out


def decode_ids(adapter: TokenizerAdapter, ids: np.ndarray, chunk_tokens: int = 200_000) -> str:
    parts: list[str] = []
    for start in range(0, len(ids), chunk_tokens):
        parts.append(adapter.decode(ids[start : start + chunk_tokens].tolist()))
    return "".join(parts)


def gzip_ratio(data: bytes) -> float:
    return len(gzip.compress(data, compresslevel=6)) / max(1, len(data))


def text_score(text: str, lm: ByteNgramLM) -> dict[str, float]:
    data = text.encode("utf-8", errors="replace")
    repl = text.count("\ufffd") / max(1, len(text))
    printable = sum((ch.isprintable() or ch.isspace()) for ch in text) / max(1, len(text))
    return {
        "byte_lm_bpb": lm.bits_per_byte(data),
        "replacement_rate": repl,
        "printable_rate": printable,
        "gzip_ratio": gzip_ratio(data),
    }


def composite_score(metrics: dict[str, float]) -> float:
    return (
        metrics["byte_lm_bpb"]
        + 20.0 * metrics["replacement_rate"]
        + 4.0 * max(0.0, 0.95 - metrics["printable_rate"])
    )


def counts(ids: np.ndarray, vocab_size: int) -> np.ndarray:
    return np.bincount(ids.astype(np.int64), minlength=vocab_size)


def sparse_context_maps(ids: np.ndarray, focus: np.ndarray, anchors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return [len(focus), len(anchors)] left/right anchor-count matrices."""

    max_id = int(max(int(ids.max(initial=0)), int(focus.max(initial=0)), int(anchors.max(initial=0)))) + 1
    focus_lookup = np.full(max_id, -1, dtype=np.int32)
    anchor_lookup = np.full(max_id, -1, dtype=np.int32)
    focus_lookup[focus.astype(np.int64)] = np.arange(len(focus), dtype=np.int32)
    anchor_lookup[anchors.astype(np.int64)] = np.arange(len(anchors), dtype=np.int32)
    left = np.zeros((len(focus), len(anchors)), dtype=np.float32)
    right = np.zeros((len(focus), len(anchors)), dtype=np.float32)
    prev = ids[:-1]
    nxt = ids[1:]
    prev_ok = prev < max_id
    nxt_ok = nxt < max_id

    fi_b = np.full(len(nxt), -1, dtype=np.int32)
    ai = np.full(len(prev), -1, dtype=np.int32)
    bi = np.full(len(nxt), -1, dtype=np.int32)
    fi_a = np.full(len(prev), -1, dtype=np.int32)
    fi_b[nxt_ok] = focus_lookup[nxt[nxt_ok]]
    ai[prev_ok] = anchor_lookup[prev[prev_ok]]
    bi[nxt_ok] = anchor_lookup[nxt[nxt_ok]]
    fi_a[prev_ok] = focus_lookup[prev[prev_ok]]

    mask = (fi_b >= 0) & (ai >= 0)
    np.add.at(left, (fi_b[mask], ai[mask]), 1.0)
    mask = (fi_a >= 0) & (bi >= 0)
    np.add.at(right, (fi_a[mask], bi[mask]), 1.0)
    return left, right


def normalize_features(x: np.ndarray, token_counts: np.ndarray) -> np.ndarray:
    y = x.astype(np.float32, copy=True)
    y /= np.sqrt(np.maximum(1.0, token_counts.astype(np.float32)))[:, None]
    norm = np.linalg.norm(y, axis=1)
    nz = norm > 0
    y[nz] /= norm[nz, None]
    return y


def align_shuffled(
    cipher_ids: np.ndarray,
    ref_ids: np.ndarray,
    vocab_size: int,
    top_tokens: int,
    anchors: int,
    candidate_window: int,
    rounds: int,
    freq_weight: float,
) -> np.ndarray:
    c_counts = counts(cipher_ids, int(max(vocab_size, int(cipher_ids.max()) + 1)))
    p_counts = counts(ref_ids, vocab_size)
    c_order_all = np.argsort(-c_counts)
    p_order_all = np.argsort(-p_counts)
    c_focus = c_order_all[: min(top_tokens, np.count_nonzero(c_counts))]
    p_focus = p_order_all[: min(top_tokens, np.count_nonzero(p_counts))]
    mapping = np.zeros(max(len(c_counts), vocab_size), dtype=np.int64)
    mapping[c_order_all[: len(p_order_all)]] = p_order_all[: len(c_order_all[: len(p_order_all)])]

    c_log = np.log(np.maximum(c_counts, 1) / max(1, int(c_counts.sum())))
    p_log = np.log(np.maximum(p_counts, 1) / max(1, int(p_counts.sum())))
    p_rank = np.empty(vocab_size, dtype=np.int64)
    p_rank[p_order_all] = np.arange(vocab_size)

    for _ in range(rounds):
        c_anchors = c_focus[: min(anchors, len(c_focus))]
        p_anchors = mapping[c_anchors]
        c_left, c_right = sparse_context_maps(cipher_ids, c_focus, c_anchors)
        p_left, p_right = sparse_context_maps(ref_ids, p_focus, p_anchors)
        c_vec = normalize_features(np.concatenate([c_left, c_right], axis=1), c_counts[c_focus])
        p_vec = normalize_features(np.concatenate([p_left, p_right], axis=1), p_counts[p_focus])
        p_focus_index = {int(p): i for i, p in enumerate(p_focus)}

        edges: list[tuple[float, int, int]] = []
        for i, c in enumerate(c_focus):
            center = int(p_rank[mapping[c]]) if mapping[c] < vocab_size else min(i, vocab_size - 1)
            lo = max(0, center - candidate_window)
            hi = min(vocab_size, center + candidate_window + 1)
            for p in p_order_all[lo:hi]:
                j = p_focus_index.get(int(p))
                if j is None:
                    continue
                score = float(np.dot(c_vec[i], p_vec[j])) - freq_weight * abs(float(c_log[c] - p_log[p]))
                edges.append((score, int(c), int(p)))
        edges.sort(reverse=True)
        used_c: set[int] = set()
        used_p: set[int] = set()
        next_mapping = mapping.copy()
        for _, c, p in edges:
            if c in used_c or p in used_p:
                continue
            next_mapping[c] = p
            used_c.add(c)
            used_p.add(p)
        mapping = next_mapping
    return mapping


def torch_context_maps(
    ids: np.ndarray,
    focus: np.ndarray,
    anchors: np.ndarray,
    device: str,
    chunk_tokens: int,
):
    import torch

    vocab_floor = int(max(int(ids.max(initial=0)), int(focus.max(initial=0)), int(anchors.max(initial=0)))) + 1
    focus_lookup = torch.full((vocab_floor,), -1, dtype=torch.int32, device=device)
    anchor_lookup = torch.full((vocab_floor,), -1, dtype=torch.int32, device=device)
    focus_t = torch.as_tensor(focus.astype(np.int64), dtype=torch.long, device=device)
    anchors_t = torch.as_tensor(anchors.astype(np.int64), dtype=torch.long, device=device)
    focus_lookup[focus_t] = torch.arange(len(focus), dtype=torch.int32, device=device)
    anchor_lookup[anchors_t] = torch.arange(len(anchors), dtype=torch.int32, device=device)

    left_flat = torch.zeros(len(focus) * len(anchors), dtype=torch.float32, device=device)
    right_flat = torch.zeros_like(left_flat)
    base = len(anchors)

    # Move chunks rather than the whole stream so large 100M-token runs fit
    # comfortably even when the GPU is also holding dense similarity matrices.
    for start in range(0, max(0, len(ids) - 1), chunk_tokens):
        stop = min(len(ids) - 1, start + chunk_tokens)
        prev = torch.as_tensor(ids[start:stop].astype(np.int64, copy=False), dtype=torch.long, device=device)
        nxt = torch.as_tensor(ids[start + 1 : stop + 1].astype(np.int64, copy=False), dtype=torch.long, device=device)

        prev_ok = prev < vocab_floor
        nxt_ok = nxt < vocab_floor

        fi_b = torch.full_like(nxt, -1, dtype=torch.int32)
        ai = torch.full_like(prev, -1, dtype=torch.int32)
        fi_a = torch.full_like(prev, -1, dtype=torch.int32)
        bi = torch.full_like(nxt, -1, dtype=torch.int32)
        fi_b[nxt_ok] = focus_lookup[nxt[nxt_ok]]
        ai[prev_ok] = anchor_lookup[prev[prev_ok]]
        fi_a[prev_ok] = focus_lookup[prev[prev_ok]]
        bi[nxt_ok] = anchor_lookup[nxt[nxt_ok]]

        mask = (fi_b >= 0) & (ai >= 0)
        if bool(mask.any()):
            flat = fi_b[mask].to(torch.long) * base + ai[mask].to(torch.long)
            left_flat += torch.bincount(flat, minlength=left_flat.numel()).to(torch.float32)

        mask = (fi_a >= 0) & (bi >= 0)
        if bool(mask.any()):
            flat = fi_a[mask].to(torch.long) * base + bi[mask].to(torch.long)
            right_flat += torch.bincount(flat, minlength=right_flat.numel()).to(torch.float32)

    left = left_flat.view(len(focus), len(anchors))
    right = right_flat.view(len(focus), len(anchors))
    return left, right


def torch_normalize_features(x, token_counts: np.ndarray, device: str):
    import torch

    counts_t = torch.as_tensor(np.maximum(1.0, token_counts.astype(np.float32)), dtype=torch.float32, device=device)
    y = x / torch.sqrt(counts_t)[:, None]
    norm = torch.linalg.vector_norm(y, dim=1).clamp_min(1e-12)
    return y / norm[:, None]


def torch_hashed_ngram_context_maps(
    ids: np.ndarray,
    focus: np.ndarray,
    anchors: np.ndarray,
    context_order: int,
    hash_bins: int,
    device: str,
    chunk_tokens: int,
):
    import torch

    if context_order <= 2 or hash_bins <= 0 or len(focus) == 0 or len(anchors) == 0:
        return torch.zeros((len(focus), 0), dtype=torch.float32, device=device)

    max_k = context_order - 1
    groups_per_k = 3
    num_groups = groups_per_k * (context_order - 2)
    feature_dim = num_groups * hash_bins
    vocab_floor = int(max(int(ids.max(initial=0)), int(focus.max(initial=0)), int(anchors.max(initial=0)))) + 1
    focus_lookup = torch.full((vocab_floor,), -1, dtype=torch.int32, device=device)
    anchor_lookup = torch.full((vocab_floor,), -1, dtype=torch.int32, device=device)
    focus_lookup[torch.as_tensor(focus.astype(np.int64), dtype=torch.long, device=device)] = torch.arange(
        len(focus), dtype=torch.int32, device=device
    )
    anchor_lookup[torch.as_tensor(anchors.astype(np.int64), dtype=torch.long, device=device)] = torch.arange(
        len(anchors), dtype=torch.int32, device=device
    )
    feature_flat = torch.zeros(len(focus) * feature_dim, dtype=torch.float32, device=device)

    def sequence_hash(parts: list):
        out = parts[0].to(torch.long) + 1
        for part in parts[1:]:
            out = (out * 1_000_003 + part.to(torch.long) + 1) % hash_bins
        return out % hash_bins

    if len(ids) <= 2 * max_k:
        return feature_flat.view(len(focus), feature_dim)

    for start in range(max_k, len(ids) - max_k, chunk_tokens):
        stop = min(len(ids) - max_k, start + chunk_tokens)
        center = torch.as_tensor(ids[start:stop].astype(np.int64, copy=False), dtype=torch.long, device=device)
        center_ok = center < vocab_floor
        focus_idx = torch.full_like(center, -1, dtype=torch.int32)
        focus_idx[center_ok] = focus_lookup[center[center_ok]]
        focus_mask = focus_idx >= 0
        if not bool(focus_mask.any()):
            continue

        prev_parts: list = []
        next_parts: list = []
        for offset in range(1, max_k + 1):
            prev = torch.as_tensor(ids[start - offset : stop - offset].astype(np.int64, copy=False), dtype=torch.long, device=device)
            nxt = torch.as_tensor(ids[start + offset : stop + offset].astype(np.int64, copy=False), dtype=torch.long, device=device)
            prev_idx = torch.full_like(prev, -1, dtype=torch.int32)
            next_idx = torch.full_like(nxt, -1, dtype=torch.int32)
            prev_ok = prev < vocab_floor
            next_ok = nxt < vocab_floor
            prev_idx[prev_ok] = anchor_lookup[prev[prev_ok]]
            next_idx[next_ok] = anchor_lookup[nxt[next_ok]]
            prev_parts.append(prev_idx)
            next_parts.append(next_idx)

        for k in range(2, max_k + 1):
            group_base = (k - 2) * groups_per_k * hash_bins

            left_seq = list(reversed(prev_parts[:k]))
            left_ok = focus_mask.clone()
            for part in left_seq:
                left_ok &= part >= 0
            if bool(left_ok.any()):
                h = sequence_hash([part[left_ok] for part in left_seq])
                flat = focus_idx[left_ok].to(torch.long) * feature_dim + group_base + h
                weights = torch.ones_like(flat, dtype=torch.float32)
                feature_flat += torch.bincount(flat, weights=weights, minlength=feature_flat.numel())

            right_seq = next_parts[:k]
            right_ok = focus_mask.clone()
            for part in right_seq:
                right_ok &= part >= 0
            if bool(right_ok.any()):
                h = sequence_hash([part[right_ok] for part in right_seq])
                flat = focus_idx[right_ok].to(torch.long) * feature_dim + group_base + hash_bins + h
                weights = torch.ones_like(flat, dtype=torch.float32)
                feature_flat += torch.bincount(flat, weights=weights, minlength=feature_flat.numel())

            left_count = k // 2
            right_count = k - left_count
            around_seq = list(reversed(prev_parts[:left_count])) + next_parts[:right_count]
            around_ok = focus_mask.clone()
            for part in around_seq:
                around_ok &= part >= 0
            if bool(around_ok.any()):
                h = sequence_hash([part[around_ok] for part in around_seq])
                flat = focus_idx[around_ok].to(torch.long) * feature_dim + group_base + 2 * hash_bins + h
                weights = torch.ones_like(flat, dtype=torch.float32)
                feature_flat += torch.bincount(flat, weights=weights, minlength=feature_flat.numel())

    return feature_flat.view(len(focus), feature_dim)


def torch_learned_context_embeddings(
    c_vec,
    p_vec,
    c_focus: np.ndarray,
    p_focus: np.ndarray,
    train_c_rows: np.ndarray,
    train_p_rows: np.ndarray,
    embed_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    temperature: float,
    max_pairs: int,
    device: str,
):
    import torch
    from torch import nn

    if epochs <= 0:
        return c_vec, p_vec, {"pairs": 0.0, "loss": 0.0}

    c_rows = train_c_rows.astype(np.int64, copy=False)
    p_rows = train_p_rows.astype(np.int64, copy=False)
    if len(c_rows) < max(128, batch_size):
        return c_vec, p_vec, {"pairs": float(len(c_rows)), "loss": 0.0}
    if max_pairs > 0 and len(c_rows) > max_pairs:
        c_rows = c_rows[:max_pairs]
        p_rows = p_rows[:max_pairs]

    input_dim = int(c_vec.shape[1])
    out_dim = min(embed_dim, input_dim)
    c_proj = nn.Linear(input_dim, out_dim, bias=False, device=device)
    p_proj = nn.Linear(input_dim, out_dim, bias=False, device=device)
    nn.init.eye_(c_proj.weight[: min(out_dim, input_dim), : min(out_dim, input_dim)])
    nn.init.eye_(p_proj.weight[: min(out_dim, input_dim), : min(out_dim, input_dim)])
    opt = torch.optim.AdamW(list(c_proj.parameters()) + list(p_proj.parameters()), lr=lr, weight_decay=0.01)
    c_train = torch.as_tensor(c_rows, dtype=torch.long, device=device)
    p_train = torch.as_tensor(p_rows, dtype=torch.long, device=device)
    labels_cache: dict[int, object] = {}
    last_loss = 0.0

    for _ in range(epochs):
        order = torch.randperm(len(c_train), device=device)
        for start in range(0, len(c_train), batch_size):
            idx = order[start : start + batch_size]
            if len(idx) < 2:
                continue
            ce = labels_cache.get(len(idx))
            if ce is None:
                ce = torch.arange(len(idx), dtype=torch.long, device=device)
                labels_cache[len(idx)] = ce
            c_emb = torch.nn.functional.normalize(c_proj(c_vec[c_train[idx]]), dim=1)
            p_emb = torch.nn.functional.normalize(p_proj(p_vec[p_train[idx]]), dim=1)
            logits = c_emb @ p_emb.T / max(temperature, 1e-4)
            loss = (
                torch.nn.functional.cross_entropy(logits, ce)
                + torch.nn.functional.cross_entropy(logits.T, ce)
            ) * 0.5
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(c_proj.parameters()) + list(p_proj.parameters()), 1.0)
            opt.step()
            last_loss = float(loss.detach().cpu())

    with torch.no_grad():
        c_out = torch.nn.functional.normalize(c_proj(c_vec), dim=1)
        p_out = torch.nn.functional.normalize(p_proj(p_vec), dim=1)
    return c_out, p_out, {"pairs": float(len(c_rows)), "loss": float(last_loss)}


def torch_confident_training_pairs(
    c_vec,
    p_vec,
    c_focus: np.ndarray,
    p_focus: np.ndarray,
    c_log: np.ndarray,
    p_log: np.ndarray,
    mapping: np.ndarray,
    p_rank: np.ndarray,
    candidate_window: int,
    freq_weight: float,
    max_pairs: int,
    min_margin: float,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    import torch

    if max_pairs <= 0 or len(c_focus) == 0 or len(p_focus) == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, {"pairs": 0.0, "mutual": 0.0, "mean_margin": 0.0}

    p_log_t = torch.as_tensor(p_log[p_focus].astype(np.float32), dtype=torch.float32, device=device)
    c_log_t_all = torch.as_tensor(c_log[c_focus].astype(np.float32), dtype=torch.float32, device=device)
    p_rank_t = torch.as_tensor(p_rank[p_focus].astype(np.int64), dtype=torch.long, device=device)
    c_center_rank = torch.as_tensor(p_rank[mapping[c_focus]].astype(np.int64), dtype=torch.long, device=device)

    best_p = torch.full((len(c_focus),), -1, dtype=torch.long, device=device)
    best_score = torch.full((len(c_focus),), -1.0e9, dtype=torch.float32, device=device)
    best_margin = torch.zeros((len(c_focus),), dtype=torch.float32, device=device)

    for start in range(0, len(c_focus), batch_size):
        stop = min(len(c_focus), start + batch_size)
        sim = c_vec[start:stop] @ p_vec.T
        sim -= freq_weight * torch.abs(c_log_t_all[start:stop, None] - p_log_t[None, :])
        if candidate_window > 0:
            mask = torch.abs(p_rank_t[None, :] - c_center_rank[start:stop, None]) <= candidate_window
            sim = sim.masked_fill(~mask, -1.0e9)
        values, indices = torch.topk(sim, k=min(2, len(p_focus)), dim=1)
        best_p[start:stop] = indices[:, 0]
        best_score[start:stop] = values[:, 0]
        if values.shape[1] > 1:
            best_margin[start:stop] = values[:, 0] - values[:, 1]

    best_c_for_p = torch.full((len(p_focus),), -1, dtype=torch.long, device=device)
    p_log_focus_t = p_log_t
    for start in range(0, len(p_focus), batch_size):
        stop = min(len(p_focus), start + batch_size)
        sim = p_vec[start:stop] @ c_vec.T
        sim -= freq_weight * torch.abs(p_log_focus_t[start:stop, None] - c_log_t_all[None, :])
        if candidate_window > 0:
            mask = torch.abs(p_rank_t[start:stop, None] - c_center_rank[None, :]) <= candidate_window
            sim = sim.masked_fill(~mask, -1.0e9)
        _, indices = torch.max(sim, dim=1)
        best_c_for_p[start:stop] = indices

    c_rows = torch.arange(len(c_focus), dtype=torch.long, device=device)
    mutual = best_c_for_p[best_p.clamp_min(0)] == c_rows
    mutual &= best_p >= 0
    if min_margin > 0:
        mutual &= best_margin >= min_margin
    rows = torch.nonzero(mutual, as_tuple=False).flatten()
    if len(rows) == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, {"pairs": 0.0, "mutual": 0.0, "mean_margin": 0.0}

    order_score = best_margin[rows] + 0.01 * best_score[rows]
    order = torch.argsort(order_score, descending=True)
    if len(order) > max_pairs:
        order = order[:max_pairs]
    rows = rows[order]
    p_rows = best_p[rows]
    margins = best_margin[rows]
    return (
        rows.detach().cpu().numpy().astype(np.int64),
        p_rows.detach().cpu().numpy().astype(np.int64),
        {
            "pairs": float(len(rows)),
            "mutual": float(int(mutual.sum().detach().cpu())),
            "mean_margin": float(margins.mean().detach().cpu()) if len(margins) else 0.0,
        },
    )


def torch_topk_edges(
    c_vec,
    p_vec,
    c_focus: np.ndarray,
    p_focus: np.ndarray,
    c_log: np.ndarray,
    p_log: np.ndarray,
    mapping: np.ndarray,
    p_rank: np.ndarray,
    candidate_window: int,
    freq_weight: float,
    topk: int,
    batch_size: int,
    device: str,
) -> list[tuple[float, int, int]]:
    import torch

    p_focus_t = torch.as_tensor(p_focus.astype(np.int64), dtype=torch.long, device=device)
    p_log_t = torch.as_tensor(p_log[p_focus].astype(np.float32), dtype=torch.float32, device=device)
    p_rank_t = torch.as_tensor(p_rank[p_focus].astype(np.int64), dtype=torch.long, device=device)
    edges: list[tuple[float, int, int]] = []
    k = min(topk, len(p_focus))

    for start in range(0, len(c_focus), batch_size):
        stop = min(len(c_focus), start + batch_size)
        c_batch = c_vec[start:stop]
        sim = c_batch @ p_vec.T

        c_ids = c_focus[start:stop]
        c_log_t = torch.as_tensor(c_log[c_ids].astype(np.float32), dtype=torch.float32, device=device)
        sim -= freq_weight * torch.abs(c_log_t[:, None] - p_log_t[None, :])

        if candidate_window > 0:
            centers = torch.as_tensor(p_rank[mapping[c_ids]].astype(np.int64), dtype=torch.long, device=device)
            mask = torch.abs(p_rank_t[None, :] - centers[:, None]) <= candidate_window
            sim = sim.masked_fill(~mask, -1.0e9)

        values, indices = torch.topk(sim, k=k, dim=1)
        values_cpu = values.detach().cpu().numpy()
        indices_cpu = indices.detach().cpu().numpy()
        for row, c in enumerate(c_ids):
            for col, score in zip(indices_cpu[row], values_cpu[row]):
                if score <= -1.0e8:
                    continue
                edges.append((float(score), int(c), int(p_focus[int(col)])))
    return edges


def sparse_sinkhorn_reweight_edges(
    edges: list[tuple[float, int, int]],
    c_focus: np.ndarray,
    p_focus: np.ndarray,
    iters: int,
    temperature: float,
    weight: float,
) -> list[tuple[float, int, int]]:
    if not edges or iters <= 0 or temperature <= 0 or weight <= 0:
        return edges

    c_pos = {int(c): i for i, c in enumerate(c_focus)}
    p_pos = {int(p): i for i, p in enumerate(p_focus)}
    scores = np.empty(len(edges), dtype=np.float32)
    rows = np.empty(len(edges), dtype=np.int64)
    cols = np.empty(len(edges), dtype=np.int64)
    c_ids = np.empty(len(edges), dtype=np.int64)
    p_ids = np.empty(len(edges), dtype=np.int64)
    kept = 0
    for score, c, p in edges:
        r = c_pos.get(int(c))
        col = p_pos.get(int(p))
        if r is None or col is None:
            continue
        scores[kept] = float(score)
        rows[kept] = r
        cols[kept] = col
        c_ids[kept] = int(c)
        p_ids[kept] = int(p)
        kept += 1
    if kept == 0:
        return edges
    scores = scores[:kept]
    rows = rows[:kept]
    cols = cols[:kept]
    c_ids = c_ids[:kept]
    p_ids = p_ids[:kept]

    log_w = (scores / float(temperature)).astype(np.float32)
    log_col_target = math.log(max(1.0, len(c_focus)) / max(1.0, len(p_focus)))

    def normalize(index: np.ndarray, size: int, target_log: float) -> None:
        max_per = np.full(size, -np.inf, dtype=np.float32)
        np.maximum.at(max_per, index, log_w)
        finite = np.isfinite(max_per[index])
        if not bool(finite.any()):
            return
        sums = np.zeros(size, dtype=np.float32)
        np.add.at(sums, index[finite], np.exp(log_w[finite] - max_per[index[finite]]))
        good = sums > 0
        lse = np.full(size, 0.0, dtype=np.float32)
        lse[good] = max_per[good] + np.log(sums[good])
        log_w[finite] += target_log - lse[index[finite]]

    for _ in range(iters):
        normalize(rows, len(c_focus), 0.0)
        normalize(cols, len(p_focus), log_col_target)

    ot_scores = (temperature * log_w).astype(np.float32)
    adjusted = (1.0 - weight) * scores + weight * ot_scores
    return [(float(s), int(c), int(p)) for s, c, p in zip(adjusted, c_ids, p_ids)]


def align_shuffled_torch(
    cipher_ids: np.ndarray,
    ref_ids: np.ndarray,
    vocab_size: int,
    top_tokens: int,
    anchors: int,
    candidate_window: int,
    rounds: int,
    freq_weight: float,
    torch_topk: int,
    torch_batch_size: int,
    torch_context_chunk: int,
    torch_device: str,
    context_order: int = 2,
    ngram_hash_bins: int = 4096,
    ngram_weight: float = 1.0,
    use_ml: bool = False,
    ml_embed_dim: int = 512,
    ml_epochs: int = 2,
    ml_batch_size: int = 512,
    ml_lr: float = 1e-3,
    ml_temperature: float = 0.05,
    ml_max_pairs: int = 16384,
    ml_min_margin: float = 0.01,
    use_ot: bool = False,
    ot_iters: int = 20,
    ot_temp: float = 0.05,
    ot_weight: float = 1.0,
) -> np.ndarray:
    import torch

    if torch_device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = torch_device
    print(f"torch_aligner device={device}", flush=True)

    c_counts = counts(cipher_ids, int(max(vocab_size, int(cipher_ids.max()) + 1)))
    p_counts = counts(ref_ids, vocab_size)
    c_order_all = np.argsort(-c_counts)
    p_order_all = np.argsort(-p_counts)
    c_focus = c_order_all[: min(top_tokens, np.count_nonzero(c_counts))].astype(np.int64)
    p_focus = p_order_all[: min(top_tokens, np.count_nonzero(p_counts))].astype(np.int64)

    mapping = np.zeros(max(len(c_counts), vocab_size), dtype=np.int64)
    mapping[c_order_all[: len(p_order_all)]] = p_order_all[: len(c_order_all[: len(p_order_all)])]

    c_log = np.log(np.maximum(c_counts, 1) / max(1, int(c_counts.sum())))
    p_log = np.log(np.maximum(p_counts, 1) / max(1, int(p_counts.sum())))
    p_rank = np.empty(vocab_size, dtype=np.int64)
    p_rank[p_order_all] = np.arange(vocab_size)

    for round_idx in range(rounds):
        c_anchors = c_focus[: min(anchors, len(c_focus))]
        p_anchors = mapping[c_anchors]
        print(
            f"torch_aligner round={round_idx + 1}/{rounds} focus={len(c_focus)} anchors={len(c_anchors)}",
            flush=True,
        )
        with torch.no_grad():
            c_left, c_right = torch_context_maps(cipher_ids, c_focus, c_anchors, device, torch_context_chunk)
            p_left, p_right = torch_context_maps(ref_ids, p_focus, p_anchors, device, torch_context_chunk)
            c_features = [c_left, c_right]
            p_features = [p_left, p_right]
            if context_order > 2:
                c_ngram = torch_hashed_ngram_context_maps(
                    cipher_ids,
                    c_focus,
                    c_anchors,
                    context_order,
                    ngram_hash_bins,
                    device,
                    torch_context_chunk,
                )
                p_ngram = torch_hashed_ngram_context_maps(
                    ref_ids,
                    p_focus,
                    p_anchors,
                    context_order,
                    ngram_hash_bins,
                    device,
                    torch_context_chunk,
                )
                if ngram_weight != 1.0:
                    c_ngram *= ngram_weight
                    p_ngram *= ngram_weight
                c_features.append(c_ngram)
                p_features.append(p_ngram)
                print(
                    f"torch_ngram_context order={context_order} hash_bins={ngram_hash_bins} dim={c_ngram.shape[1]}",
                    flush=True,
                )
            c_vec = torch_normalize_features(torch.cat(c_features, dim=1), c_counts[c_focus], device)
            p_vec = torch_normalize_features(torch.cat(p_features, dim=1), p_counts[p_focus], device)
            del c_left, c_right, p_left, p_right
            if context_order > 2:
                del c_ngram, p_ngram

        if use_ml and round_idx > 0:
            train_c_rows, train_p_rows, pair_stats = torch_confident_training_pairs(
                c_vec,
                p_vec,
                c_focus,
                p_focus,
                c_log,
                p_log,
                mapping,
                p_rank,
                candidate_window,
                freq_weight,
                ml_max_pairs,
                ml_min_margin,
                ml_batch_size,
                device,
            )
            print(
                "torch_ml_pairs "
                f"round={round_idx + 1}/{rounds} pairs={pair_stats['pairs']:.0f} "
                f"mutual={pair_stats['mutual']:.0f} mean_margin={pair_stats['mean_margin']:.4f}",
                flush=True,
            )
            c_vec, p_vec, ml_stats = torch_learned_context_embeddings(
                c_vec,
                p_vec,
                c_focus,
                p_focus,
                train_c_rows,
                train_p_rows,
                ml_embed_dim,
                ml_epochs,
                ml_batch_size,
                ml_lr,
                ml_temperature,
                ml_max_pairs,
                device,
            )
            print(
                f"torch_ml round={round_idx + 1}/{rounds} pairs={ml_stats['pairs']:.0f} loss={ml_stats['loss']:.4f}",
                flush=True,
            )

        with torch.no_grad():
            edges = torch_topk_edges(
                c_vec,
                p_vec,
                c_focus,
                p_focus,
                c_log,
                p_log,
                mapping,
                p_rank,
                candidate_window,
                freq_weight,
                torch_topk,
                torch_batch_size,
                device,
            )
            del c_vec, p_vec
            if device == "cuda":
                torch.cuda.empty_cache()

        if use_ot:
            edges = sparse_sinkhorn_reweight_edges(
                edges,
                c_focus,
                p_focus,
                iters=ot_iters,
                temperature=ot_temp,
                weight=ot_weight,
            )
            print(
                f"torch_ot round={round_idx + 1}/{rounds} edges={len(edges)} iters={ot_iters} temp={ot_temp}",
                flush=True,
            )

        edges.sort(reverse=True)
        used_c: set[int] = set()
        used_p: set[int] = set()
        next_mapping = mapping.copy()
        for _, c, p in edges:
            if c in used_c or p in used_p:
                continue
            next_mapping[c] = p
            used_c.add(c)
            used_p.add(p)
        mapping = next_mapping
        print(f"torch_aligner assigned={len(used_c)}", flush=True)
    return mapping


def build_pair_count_dict(prev: np.ndarray, nxt: np.ndarray, code_base: int) -> dict[int, int]:
    if len(prev) == 0:
        return {}
    codes = prev.astype(np.int64) * code_base + nxt.astype(np.int64)
    uniq, counts_arr = np.unique(codes, return_counts=True)
    return {int(code): int(count) for code, count in zip(uniq, counts_arr)}


class SparseBigramLM:
    def __init__(self, ref_ids: np.ndarray, vocab_size: int, alpha: float):
        self.vocab_size = vocab_size
        self.alpha = alpha
        self.prev_counts = np.bincount(ref_ids[:-1].astype(np.int64), minlength=vocab_size).astype(np.float64)
        unigram = np.bincount(ref_ids.astype(np.int64), minlength=vocab_size).astype(np.float64)
        self.base_prob = (unigram + 1.0) / (unigram.sum() + vocab_size)
        self.pair_counts = build_pair_count_dict(ref_ids[:-1], ref_ids[1:], vocab_size)
        self.cache: dict[int, float] = {}

    def logp(self, prev_plain: int, next_plain: int) -> float:
        if prev_plain < 0 or next_plain < 0 or prev_plain >= self.vocab_size or next_plain >= self.vocab_size:
            return -30.0
        code = int(prev_plain) * self.vocab_size + int(next_plain)
        cached = self.cache.get(code)
        if cached is not None:
            return cached
        count = self.pair_counts.get(code, 0)
        prob = (count + self.alpha * self.base_prob[next_plain]) / (self.prev_counts[prev_plain] + self.alpha)
        value = math.log(max(prob, 1e-300))
        self.cache[code] = value
        return value


def build_active_cipher_pairs(cipher_ids: np.ndarray, active: np.ndarray) -> tuple[dict[int, int], dict[int, list[int]], dict[int, list[int]], int]:
    code_base = int(cipher_ids.max()) + 1
    active_lookup = np.full(code_base, False, dtype=bool)
    active_lookup[active.astype(np.int64)] = True
    prev = cipher_ids[:-1].astype(np.int64)
    nxt = cipher_ids[1:].astype(np.int64)
    mask = active_lookup[prev] | active_lookup[nxt]
    pair_counts = build_pair_count_dict(prev[mask], nxt[mask], code_base)
    out_edges: dict[int, list[int]] = {int(c): [] for c in active}
    in_edges: dict[int, list[int]] = {int(c): [] for c in active}
    active_set = set(int(c) for c in active)
    for code in pair_counts:
        a = code // code_base
        b = code % code_base
        if a in active_set:
            out_edges[a].append(b)
        if b in active_set:
            in_edges[b].append(a)
    return pair_counts, out_edges, in_edges, code_base


def anneal_mapping(
    cipher_ids: np.ndarray,
    ref_ids: np.ndarray,
    mapping: np.ndarray,
    vocab_size: int,
    anneal_top_tokens: int,
    anneal_steps: int,
    anneal_alpha: float,
    start_temp: float,
    end_temp: float,
    seed: int,
) -> tuple[np.ndarray, dict[str, float]]:
    if anneal_steps <= 0 or anneal_top_tokens < 2:
        return mapping, {"steps": 0.0, "accepted": 0.0, "improved": 0.0}

    c_counts = counts(cipher_ids, len(mapping))
    active = np.argsort(-c_counts)[: min(anneal_top_tokens, int(np.count_nonzero(c_counts)))]
    active = active[c_counts[active] > 0].astype(np.int64)
    if len(active) < 2:
        return mapping, {"steps": 0.0, "accepted": 0.0, "improved": 0.0}

    bigram_lm = SparseBigramLM(ref_ids, vocab_size, alpha=anneal_alpha)
    pair_counts, out_edges, in_edges, code_base = build_active_cipher_pairs(cipher_ids, active)
    rng = random.Random(seed)
    current = mapping.copy()
    active_list = [int(x) for x in active]
    accepted = 0
    improved = 0

    def affected_pairs(a: int, b: int) -> set[int]:
        pairs: set[int] = set()
        for x in (a, b):
            for y in out_edges.get(x, []):
                pairs.add(x * code_base + y)
            for y in in_edges.get(x, []):
                pairs.add(y * code_base + x)
        return pairs

    def pair_score(code: int, m: np.ndarray) -> float:
        a = code // code_base
        b = code % code_base
        return pair_counts[code] * bigram_lm.logp(int(m[a]), int(m[b]))

    for step in range(anneal_steps):
        a, b = rng.sample(active_list, 2)
        if current[a] == current[b]:
            continue
        pairs = affected_pairs(a, b)
        if not pairs:
            continue
        old = 0.0
        new = 0.0
        affected_count = 0
        pa = int(current[a])
        pb = int(current[b])
        for code in pairs:
            count = pair_counts[code]
            affected_count += count
            old += pair_score(code, current)
            x = code // code_base
            y = code % code_base
            mx = pb if x == a else pa if x == b else int(current[x])
            my = pb if y == a else pa if y == b else int(current[y])
            new += count * bigram_lm.logp(mx, my)
        delta = new - old
        delta_avg = delta / max(1, affected_count)
        progress = step / max(1, anneal_steps - 1)
        temp = start_temp * ((end_temp / start_temp) ** progress) if start_temp > 0 and end_temp > 0 else 0.0
        if delta >= 0 or (temp > 0 and rng.random() < math.exp(delta_avg / temp)):
            current[a], current[b] = current[b], current[a]
            accepted += 1
            if delta > 0:
                improved += 1

    return current, {
        "steps": float(anneal_steps),
        "accepted": float(accepted),
        "improved": float(improved),
        "active_tokens": float(len(active_list)),
    }


def try_direct(
    ids: np.ndarray,
    adapter: TokenizerAdapter,
    lm: ByteNgramLM,
    sample_tokens: int,
) -> dict[str, object] | None:
    if ids.size == 0 or ids.max() >= adapter.spec.vocab_size or ids.min() < 0:
        return None
    sample = ids[:sample_tokens]
    text = adapter.decode(sample.tolist())
    metrics = text_score(text, lm)
    metrics["score"] = composite_score(metrics)
    return {
        "mode": "direct",
        "tokenizer": adapter.spec.name,
        "family": adapter.spec.family,
        "mapping": None,
        "metrics": metrics,
        "preview": text[:1000].replace("\n", "\\n"),
    }


def try_shuffled(
    ids: np.ndarray,
    adapter: TokenizerAdapter,
    ref_ids: np.ndarray,
    lm: ByteNgramLM,
    sample_tokens: int,
    top_tokens: int,
    anchors: int,
    candidate_window: int,
    rounds: int,
    freq_weight: float,
    aligner: str,
    torch_topk: int,
    torch_batch_size: int,
    torch_context_chunk: int,
    torch_device: str,
    context_order: int,
    ngram_hash_bins: int,
    ngram_weight: float,
    ml_embed_dim: int,
    ml_epochs: int,
    ml_batch_size: int,
    ml_lr: float,
    ml_temperature: float,
    ml_max_pairs: int,
    ml_min_margin: float,
    ot_iters: int,
    ot_temp: float,
    ot_weight: float,
    vocab_size_tolerance: float,
    vocab_prior_weight: float,
    anneal_steps: int,
    anneal_top_tokens: int,
    anneal_alpha: float,
    anneal_start_temp: float,
    anneal_end_temp: float,
    seed: int,
) -> dict[str, object] | None:
    if ids.size == 0:
        return None
    if int(ids.max()) >= 5_000_000:
        return None
    observed_vocab_floor = int(ids.max()) + 1
    if aligner in ("torch", "torch_ml", "torch_ml_ot"):
        mapping = align_shuffled_torch(
            ids,
            ref_ids,
            adapter.spec.vocab_size,
            top_tokens=top_tokens,
            anchors=anchors,
            candidate_window=candidate_window,
            rounds=rounds,
            freq_weight=freq_weight,
            torch_topk=torch_topk,
            torch_batch_size=torch_batch_size,
            torch_context_chunk=torch_context_chunk,
            torch_device=torch_device,
            context_order=context_order,
            ngram_hash_bins=ngram_hash_bins,
            ngram_weight=ngram_weight,
            use_ml=aligner in ("torch_ml", "torch_ml_ot"),
            ml_embed_dim=ml_embed_dim,
            ml_epochs=ml_epochs,
            ml_batch_size=ml_batch_size,
            ml_lr=ml_lr,
            ml_temperature=ml_temperature,
            ml_max_pairs=ml_max_pairs,
            ml_min_margin=ml_min_margin,
            use_ot=aligner == "torch_ml_ot",
            ot_iters=ot_iters,
            ot_temp=ot_temp,
            ot_weight=ot_weight,
        )
    else:
        mapping = align_shuffled(
            ids,
            ref_ids,
            adapter.spec.vocab_size,
            top_tokens=top_tokens,
            anchors=anchors,
            candidate_window=candidate_window,
            rounds=rounds,
            freq_weight=freq_weight,
        )
    mapped_sample = mapping[ids[:sample_tokens]]
    text = adapter.decode(mapped_sample.tolist())
    metrics = text_score(text, lm)
    vocab_ratio = adapter.spec.vocab_size / max(1, observed_vocab_floor)
    vocab_penalty = max(0.0, math.log(vocab_ratio / max(1.0, vocab_size_tolerance)))
    metrics["observed_vocab_floor"] = float(observed_vocab_floor)
    metrics["candidate_vocab_size"] = float(adapter.spec.vocab_size)
    metrics["vocab_ratio"] = float(vocab_ratio)
    metrics["vocab_penalty"] = float(vocab_penalty)
    graph_score = composite_score(metrics) + vocab_prior_weight * vocab_penalty
    metrics["score"] = graph_score
    metrics["graph_score"] = graph_score
    metrics["anneal_used"] = 0.0
    metrics["ml_used"] = float(aligner in ("torch_ml", "torch_ml_ot"))
    metrics["ot_used"] = float(aligner == "torch_ml_ot")
    metrics["context_order"] = float(context_order)
    metrics["ngram_hash_bins"] = float(ngram_hash_bins if context_order > 2 else 0)

    anneal_stats: dict[str, float] = {}
    if anneal_steps > 0:
        annealed_mapping, anneal_stats = anneal_mapping(
            ids,
            ref_ids,
            mapping,
            adapter.spec.vocab_size,
            anneal_top_tokens=anneal_top_tokens,
            anneal_steps=anneal_steps,
            anneal_alpha=anneal_alpha,
            start_temp=anneal_start_temp,
            end_temp=anneal_end_temp,
            seed=seed,
        )
        annealed_text = adapter.decode(annealed_mapping[ids[:sample_tokens]].tolist())
        annealed_metrics = text_score(annealed_text, lm)
        annealed_metrics["observed_vocab_floor"] = float(observed_vocab_floor)
        annealed_metrics["candidate_vocab_size"] = float(adapter.spec.vocab_size)
        annealed_metrics["vocab_ratio"] = float(vocab_ratio)
        annealed_metrics["vocab_penalty"] = float(vocab_penalty)
        annealed_score = composite_score(annealed_metrics) + vocab_prior_weight * vocab_penalty
        if annealed_score <= graph_score:
            mapping = annealed_mapping
            text = annealed_text
            metrics = annealed_metrics
            metrics["score"] = annealed_score
            metrics["graph_score"] = graph_score
            metrics["annealed_score"] = annealed_score
            metrics["anneal_used"] = 1.0
        else:
            metrics["annealed_score"] = annealed_score
            metrics["anneal_used"] = 0.0
        for key, value in anneal_stats.items():
            metrics[f"anneal_{key}"] = value
    return {
        "mode": "shuffled",
        "tokenizer": adapter.spec.name,
        "family": adapter.spec.family,
        "mapping": mapping,
        "metrics": metrics,
        "preview": text[:1000].replace("\n", "\\n"),
    }


def parse_scales(scales: str) -> list[float]:
    out: list[float] = []
    for part in scales.split(","):
        part = part.strip()
        if not part:
            continue
        value = float(part)
        if value <= 0:
            raise ValueError("surrogate vocab scales must be positive")
        out.append(value)
    return out


def surrogate_vocab_sizes(inferred_vocab: int, scales: list[float], min_vocab: int, max_vocab: int) -> list[int]:
    sizes: list[int] = []
    for scale in scales:
        size = int(round(inferred_vocab * scale))
        size = max(min_vocab, min(max_vocab, size))
        # ByteLevel BPE has a 256-byte alphabet. Rounding to a multiple of 256
        # avoids noisy near-duplicate candidates while keeping the inferred
        # scale close enough for graph alignment.
        size = max(256, int(round(size / 256) * 256))
        if size not in sizes:
            sizes.append(size)
    return sizes


def build_candidates(
    names: list[str],
    ref_path: Path,
    work_dir: Path,
    inferred_vocab: int,
    surrogate_scales: str,
    surrogate_train_bytes: int,
    surrogate_min_vocab: int,
    surrogate_max_vocab: int,
) -> list[TokenizerAdapter]:
    candidates: list[TokenizerAdapter] = []
    surrogate_train: Path | None = None
    for name in names:
        print(f"loading tokenizer candidate: {name}", flush=True)
        try:
            if name == "surrogate_bpe":
                if surrogate_train is None:
                    surrogate_train = reference_prefix(ref_path, work_dir, surrogate_train_bytes)
                for vocab_size in surrogate_vocab_sizes(
                    inferred_vocab,
                    parse_scales(surrogate_scales),
                    surrogate_min_vocab,
                    surrogate_max_vocab,
                ):
                    adapter = TrainedByteLevelBPEAdapter(
                        name=f"surrogate_bpe_v{vocab_size}",
                        train_file=surrogate_train,
                        vocab_size=vocab_size,
                        out_dir=work_dir / f"surrogate_bpe_v{vocab_size}",
                    )
                    candidates.append(adapter)
                    print(f"  loaded {adapter.spec.name} vocab={adapter.spec.vocab_size}", flush=True)
                continue
            if name == "trained_bpe":
                if surrogate_train is None:
                    surrogate_train = reference_prefix(ref_path, work_dir, surrogate_train_bytes)
                adapter = build_tokenizer(name, train_file=surrogate_train, work_dir=work_dir, vocab_size=inferred_vocab)
            else:
                adapter = build_tokenizer(name)
            candidates.append(adapter)
            print(f"  loaded {adapter.spec.name} vocab={adapter.spec.vocab_size}", flush=True)
        except Exception as exc:
            print(f"  skipped {name}: {exc}", flush=True)
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", required=True, help="Path to token IDs: txt/csv/json/npy")
    parser.add_argument("--out", required=True, help="Recovered text output path")
    parser.add_argument("--report", default="", help="JSON report path")
    parser.add_argument("--reference-text", default=".cache/fineweb_100m.txt")
    parser.add_argument("--reference-min-bytes", type=int, default=50_000_000)
    parser.add_argument("--work-dir", default=".cache/recover_text")
    parser.add_argument("--tokenizers", default=",".join(DEFAULT_TOKENIZERS + ["surrogate_bpe"]))
    parser.add_argument("--reference-tokens", type=int, default=5_000_000)
    parser.add_argument("--surrogate-train-bytes", type=int, default=64_000_000)
    parser.add_argument("--surrogate-vocab-scales", default="0.75,1.0,1.25")
    parser.add_argument("--surrogate-min-vocab", type=int, default=512)
    parser.add_argument("--surrogate-max-vocab", type=int, default=262_144)
    parser.add_argument("--lm-bytes", type=int, default=16_000_000)
    parser.add_argument("--sample-tokens", type=int, default=200_000)
    parser.add_argument("--top-tokens", type=int, default=1200)
    parser.add_argument("--anchors", type=int, default=256)
    parser.add_argument("--candidate-window", type=int, default=160)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--freq-weight", type=float, default=0.20)
    parser.add_argument("--aligner", choices=["numpy", "torch", "torch_ml", "torch_ml_ot"], default="numpy")
    parser.add_argument("--torch-topk", type=int, default=32)
    parser.add_argument("--torch-batch-size", type=int, default=512)
    parser.add_argument("--torch-context-chunk", type=int, default=2_000_000)
    parser.add_argument("--torch-device", default="auto")
    parser.add_argument("--context-order", type=int, default=2)
    parser.add_argument("--ngram-hash-bins", type=int, default=4096)
    parser.add_argument("--ngram-weight", type=float, default=1.0)
    parser.add_argument("--ml-embed-dim", type=int, default=512)
    parser.add_argument("--ml-epochs", type=int, default=2)
    parser.add_argument("--ml-batch-size", type=int, default=512)
    parser.add_argument("--ml-lr", type=float, default=1e-3)
    parser.add_argument("--ml-temperature", type=float, default=0.05)
    parser.add_argument("--ml-max-pairs", type=int, default=16384)
    parser.add_argument("--ml-min-margin", type=float, default=0.01)
    parser.add_argument("--ot-iters", type=int, default=20)
    parser.add_argument("--ot-temp", type=float, default=0.05)
    parser.add_argument("--ot-weight", type=float, default=1.0)
    parser.add_argument("--vocab-size-tolerance", type=float, default=1.05)
    parser.add_argument("--vocab-prior-weight", type=float, default=4.0)
    parser.add_argument("--anneal-steps", type=int, default=0)
    parser.add_argument("--anneal-top-tokens", type=int, default=512)
    parser.add_argument("--anneal-alpha", type=float, default=10.0)
    parser.add_argument("--anneal-start-temp", type=float, default=0.05)
    parser.add_argument("--anneal-end-temp", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--force-mode", choices=["auto", "direct", "shuffled"], default="auto")
    parser.add_argument("--save-mapping", default="")
    args = parser.parse_args()
    if args.context_order < 2:
        raise ValueError("--context-order must be at least 2")
    if args.context_order > 2 and args.ngram_hash_bins <= 0:
        raise ValueError("--ngram-hash-bins must be positive when --context-order > 2")

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    ids = load_ids(Path(args.ids))
    if ids.size == 0:
        raise ValueError("input ID file contained no token IDs")
    inferred_vocab = int(ids.max()) + 1
    print(f"loaded_ids={len(ids):,} min_id={int(ids.min())} max_id={int(ids.max())} inferred_vocab>={inferred_vocab:,}", flush=True)

    ref_path = ensure_reference_text(Path(args.reference_text), args.reference_min_bytes)
    lm_data = ref_path.read_bytes()[: args.lm_bytes]
    lm = ByteNgramLM(order=4)
    lm.train(lm_data)

    names = [x.strip() for x in args.tokenizers.split(",") if x.strip()]
    candidates = build_candidates(
        names,
        ref_path,
        work_dir,
        inferred_vocab,
        args.surrogate_vocab_scales,
        args.surrogate_train_bytes,
        args.surrogate_min_vocab,
        args.surrogate_max_vocab,
    )
    if not candidates:
        raise RuntimeError("no tokenizer candidates loaded")

    results: list[dict[str, object]] = []
    ref_cache: dict[str, np.ndarray] = {}
    for adapter in candidates:
        if args.force_mode in ("auto", "direct"):
            direct = try_direct(ids, adapter, lm, args.sample_tokens)
            if direct is not None:
                print(f"direct {adapter.spec.name}: score={direct['metrics']['score']:.3f} preview={direct['preview'][:120]}", flush=True)
                results.append(direct)
        if args.force_mode in ("auto", "shuffled"):
            print(f"encoding reference for shuffled candidate {adapter.spec.name}", flush=True)
            ref_ids = ref_cache.get(adapter.spec.name)
            if ref_ids is None:
                ref_ids = encode_reference(adapter, ref_path, args.reference_tokens)
                ref_cache[adapter.spec.name] = ref_ids
            shuffled = try_shuffled(
                ids,
                adapter,
                ref_ids,
                lm,
                args.sample_tokens,
                args.top_tokens,
                args.anchors,
                args.candidate_window,
                args.rounds,
                args.freq_weight,
                args.aligner,
                args.torch_topk,
                args.torch_batch_size,
                args.torch_context_chunk,
                args.torch_device,
                args.context_order,
                args.ngram_hash_bins,
                args.ngram_weight,
                args.ml_embed_dim,
                args.ml_epochs,
                args.ml_batch_size,
                args.ml_lr,
                args.ml_temperature,
                args.ml_max_pairs,
                args.ml_min_margin,
                args.ot_iters,
                args.ot_temp,
                args.ot_weight,
                args.vocab_size_tolerance,
                args.vocab_prior_weight,
                args.anneal_steps,
                args.anneal_top_tokens,
                args.anneal_alpha,
                args.anneal_start_temp,
                args.anneal_end_temp,
                args.seed,
            )
            if shuffled is not None:
                print(f"shuffled {adapter.spec.name}: score={shuffled['metrics']['score']:.3f} preview={shuffled['preview'][:120]}", flush=True)
                results.append(shuffled)

    if not results:
        raise RuntimeError("no viable direct or shuffled candidates")
    results.sort(key=lambda r: r["metrics"]["score"])  # type: ignore[index]
    best = results[0]
    best_adapter = next(a for a in candidates if a.spec.name == best["tokenizer"])

    if best["mode"] == "direct":
        recovered = decode_ids(best_adapter, ids)
    else:
        mapping = best["mapping"]
        assert isinstance(mapping, np.ndarray)
        recovered = decode_ids(best_adapter, mapping[ids])
        if args.save_mapping:
            np.save(args.save_mapping, mapping)

    Path(args.out).write_text(recovered, encoding="utf-8", errors="ignore")
    report = {
        "input": str(args.ids),
        "output": str(args.out),
        "num_ids": int(len(ids)),
        "min_id": int(ids.min()),
        "max_id": int(ids.max()),
        "chosen": {
            "mode": best["mode"],
            "tokenizer": best["tokenizer"],
            "family": best["family"],
            "metrics": best["metrics"],
            "preview": best["preview"],
        },
        "ranked_candidates": [
            {
                "mode": r["mode"],
                "tokenizer": r["tokenizer"],
                "family": r["family"],
                "metrics": r["metrics"],
                "preview": r["preview"],
            }
            for r in results
        ],
    }
    report_path = Path(args.report) if args.report else Path(args.out).with_suffix(Path(args.out).suffix + ".report.json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {args.out}", flush=True)
    print(f"wrote {report_path}", flush=True)
    print(json.dumps(report["chosen"], indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
