"""Mutable detokenizer experiment.

This file is the hillclimb target. The baseline implements the current
frequency + bigram-context graph aligner for a shuffled token-ID stream. Agents
should modify this file only, run `uv run train.py`, and keep changes that lower
cer50k.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from prepare import (
    DEFAULT_REFERENCE_TOKENS,
    DEFAULT_SAMPLE_TOKENS,
    DEFAULT_SEED,
    DEFAULT_TARGET_TOKENS,
    CACHE_DIR,
    evaluate_recovery,
    load_task,
)

# ---------------------------------------------------------------------------
# Experiment knobs. Agents may edit these directly.
# ---------------------------------------------------------------------------

SOURCE_TOKENIZER = os.environ.get("DETOK_SOURCE", "kimi_k2")
TARGET_TOKENIZER = os.environ.get("DETOK_TARGET", "openai_o200k")
TARGET_TOKENS = int(os.environ.get("DETOK_TARGET_TOKENS", DEFAULT_TARGET_TOKENS))
REFERENCE_TOKENS = int(os.environ.get("DETOK_REFERENCE_TOKENS", DEFAULT_REFERENCE_TOKENS))
SAMPLE_TOKENS = int(os.environ.get("DETOK_SAMPLE_TOKENS", DEFAULT_SAMPLE_TOKENS))
SEED = int(os.environ.get("DETOK_SEED", DEFAULT_SEED))
UNSHUFFLED_SOURCE_IDS = os.environ.get("DETOK_UNSHUFFLED_SOURCE", "1") == "1"
ID_PROXIMITY_WEIGHT = float(os.environ.get("DETOK_ID_PROXIMITY_WEIGHT", "0.05"))
_ID_PROXIMITY_FINAL_WEIGHT = os.environ.get("DETOK_ID_PROXIMITY_FINAL_WEIGHT")
ID_PROXIMITY_FINAL_WEIGHT = (
    None if _ID_PROXIMITY_FINAL_WEIGHT is None else float(_ID_PROXIMITY_FINAL_WEIGHT)
)
ID_PROXIMITY_DECAY_POWER = float(os.environ.get("DETOK_ID_PROXIMITY_DECAY_POWER", "1.0"))
ID_CANDIDATE_WINDOW = int(os.environ.get("DETOK_ID_CANDIDATE_WINDOW", "0"))
ID_PROXIMITY_MODE = os.environ.get("DETOK_ID_PROXIMITY_MODE", "absolute")
ID_INIT_MODE = os.environ.get("DETOK_ID_INIT_MODE", "freq")
ID_EXACT_BONUS = float(os.environ.get("DETOK_ID_EXACT_BONUS", "0.0"))
ID_LOCK_PREFIX = int(os.environ.get("DETOK_ID_LOCK_PREFIX", "0"))
ID_LOCK_MODE = os.environ.get("DETOK_ID_LOCK_MODE", "absolute")
DESHUFFLE_BY_FREQUENCY = os.environ.get("DETOK_DESHUFFLE_BY_FREQUENCY", "0") == "1"
DESHUFFLE_FREQ_ORDER = os.environ.get("DETOK_DESHUFFLE_FREQ_ORDER", "desc")
STRUCTURAL_ANCHOR_AUDIT = os.environ.get("DETOK_STRUCTURAL_ANCHOR_AUDIT", "0") == "1"
STRUCTURAL_AUDIT_TOP_N = int(os.environ.get("DETOK_STRUCTURAL_AUDIT_TOP_N", "5000"))
STRUCTURAL_AUDIT_REPORT_TOP = int(os.environ.get("DETOK_STRUCTURAL_AUDIT_REPORT_TOP", "25"))
STRUCTURAL_AUDIT_SPECTRAL_DIMS = int(os.environ.get("DETOK_STRUCTURAL_AUDIT_SPECTRAL_DIMS", "16"))
STRUCTURAL_AUDIT_SPECTRAL_ITERS = int(os.environ.get("DETOK_STRUCTURAL_AUDIT_SPECTRAL_ITERS", "8"))
STRUCTURAL_AUDIT_TOKEN_LIMIT = int(os.environ.get("DETOK_STRUCTURAL_AUDIT_TOKENS", "1000000"))
STRUCTURAL_AUDIT_LABEL_TOP = int(os.environ.get("DETOK_STRUCTURAL_AUDIT_LABEL_TOP", "20000"))
STRUCTURAL_SEED_ANCHORS = os.environ.get("DETOK_STRUCTURAL_SEED_ANCHORS", "0") == "1"
STRUCTURAL_SEED_LOCK = os.environ.get("DETOK_STRUCTURAL_SEED_LOCK", "1") == "1"
STRUCTURAL_SEED_COMMON_SURFACES = tuple(
    s for s in os.environ.get("DETOK_STRUCTURAL_SEED_COMMON_SURFACES", "").split("|") if s
)
STRUCTURAL_FEATURES = os.environ.get("DETOK_STRUCTURAL_FEATURES", "0") == "1"
STRUCTURAL_FEATURE_TOP_N = int(os.environ.get("DETOK_STRUCTURAL_FEATURE_TOP_N", "5000"))
STRUCTURAL_FEATURE_WEIGHT = float(os.environ.get("DETOK_STRUCTURAL_FEATURE_WEIGHT", "0.25"))

TOP_TOKENS = 50_000
ANCHORS = 8_192
CANDIDATE_WINDOW = 10_000
ROUNDS = 6
FREQ_WEIGHT = 0.12
TORCH_TOPK = 64
TORCH_BATCH_SIZE = 256
TORCH_CONTEXT_CHUNK = 5_000_000
SKIP_CONTEXT_MIN_TOKENS = 100_000
SKIP_CONTEXT_WEIGHT = 1.0
LEARN_SKIP_WEIGHT = True
LEARN_WEIGHT_SEEDS = 512
LEARN_WEIGHT_STEPS = 12
LEARN_WEIGHT_LR = 0.2
LEARN_WEIGHT_TEMP = 0.07
DYNAMIC_ANCHOR_MAX_TOKENS = 100_000
ENABLE_DYNAMIC_ASSIGNMENT_SWAPS = True
ASSIGNMENT_SWAP_MIN_GAIN = 0.01
BIGRAM_OBJECTIVE_REFINE = True
BIGRAM_REFINE_ALL_SCALES = True
BIGRAM_REFINE_TOKENS = 2_048
BIGRAM_REFINE_LARGE_TOKENS = 8_192
BIGRAM_REFINE_LARGE_TOKEN_MIN_TOKENS = 1_000_000
BIGRAM_REFINE_LARGE_TOKEN_MAX_TOKENS = 1_000_000
BIGRAM_REFINE_MAX_PROPOSALS = 100_000
BIGRAM_REFINE_PASSES = 4
BIGRAM_REFINE_LARGE_PASSES = 3
BIGRAM_REFINE_SKIP_WEIGHT = 0.4
BIGRAM_REFINE_SKIP_MIN_TOKENS = 1_000_000
BIGRAM_REFINE_SKIP_MAX_TOKENS = 1_000_000
BIGRAM_REFINE_ALPHA = 0.05
BIGRAM_UNIGRAM_BACKOFF = True
BIGRAM_UNIGRAM_BACKOFF_TAU = 1000.0
BIGRAM_UNIGRAM_BACKOFF_MAX_TOKENS = 100_000


def effective_candidate_window(num_cipher_tokens: int) -> int:
    return 25_000 if num_cipher_tokens >= 1_000_000 else CANDIDATE_WINDOW


def effective_rounds(num_cipher_tokens: int) -> int:
    return 8 if num_cipher_tokens >= 1_000_000 else ROUNDS


def effective_id_proximity_weight(round_idx: int, rounds: int) -> float:
    if ID_PROXIMITY_FINAL_WEIGHT is None:
        return ID_PROXIMITY_WEIGHT
    if rounds <= 1:
        progress = 1.0
    else:
        progress = round_idx / float(rounds - 1)
    progress = max(0.0, min(1.0, progress)) ** max(ID_PROXIMITY_DECAY_POWER, 1.0e-6)
    return ID_PROXIMITY_WEIGHT * (1.0 - progress) + ID_PROXIMITY_FINAL_WEIGHT * progress


def counts(ids: np.ndarray, size: int) -> np.ndarray:
    return np.bincount(ids.astype(np.int64, copy=False), minlength=size)


def frequency_rank_ids(ids: np.ndarray, order: str) -> np.ndarray:
    """Renumber observed token IDs by their unigram frequency rank."""
    source_counts = counts(ids, int(ids.max(initial=0)) + 1)
    observed = np.flatnonzero(source_counts)
    if order == "asc":
        ranked = observed[np.argsort(source_counts[observed], kind="stable")]
    else:
        ranked = observed[np.argsort(-source_counts[observed], kind="stable")]
    rank_by_token = np.full(len(source_counts), len(ranked), dtype=np.int64)
    rank_by_token[ranked] = np.arange(len(ranked), dtype=np.int64)
    return rank_by_token[ids].astype(np.int64, copy=False)


def apply_id_lock(mapping: np.ndarray, target_vocab_size: int, p_order_all: np.ndarray | None = None) -> None:
    if not (UNSHUFFLED_SOURCE_IDS and ID_LOCK_PREFIX > 0):
        return
    limit = min(ID_LOCK_PREFIX, len(mapping), target_vocab_size)
    if ID_LOCK_MODE == "rank" and p_order_all is not None:
        mapping[:limit] = p_order_all[:limit]
    else:
        mapping[:limit] = np.arange(limit, dtype=np.int64)


def display_surface(text: str, max_len: int = 48) -> str:
    shown = text.encode("unicode_escape", errors="backslashreplace").decode("ascii", errors="replace")
    if len(shown) > max_len:
        return shown[: max_len - 3] + "..."
    return shown


def decode_token_text(adapter, token_id: int) -> str:
    try:
        return adapter.decode([int(token_id)])
    except Exception:
        try:
            return adapter.token_bytes(int(token_id)).decode("utf-8", errors="replace")
        except Exception:
            return f"<INVALID:{int(token_id)}>"


def surface_class(text: str) -> str:
    if text == "":
        return "empty"
    stripped = text.strip(" \t\r\n")
    if "\n" in text:
        return "newline"
    if text.isspace():
        return "whitespace"
    prefix = "leading_space_" if text.startswith((" ", "Ġ", "▁")) else ""
    core = stripped.lstrip("Ġ▁")
    if core == "":
        return prefix + "marker"
    if core in {".", "!", "?", ";", ":"}:
        return prefix + "sentence_punct"
    if core == ",":
        return prefix + "comma"
    if all(ch in "\"'`“”‘’()[]{}<>-/\\|_*&^%$#@~+=،。．，、" for ch in core):
        return prefix + "punct"
    if core.isdigit():
        return prefix + "digit"
    if core.isalpha():
        return prefix + "alpha"
    if any(ch.isalpha() for ch in core) and any(ch.isdigit() for ch in core):
        return prefix + "alnum"
    if any(ch.isalpha() for ch in core):
        return prefix + "mixed_alpha"
    return prefix + "other"


def anchor_patterns() -> list[tuple[str, object]]:
    return [
        ("bare_space", lambda text: text in {" ", "Ġ", "▁"}),
        ("whitespace_no_newline", lambda text: text.isspace() and "\n" not in text),
        ("newline", lambda text: "\n" in text),
        ("period", lambda text: text.strip(" \t\r\nĠ▁") == "."),
        ("comma", lambda text: text.strip(" \t\r\nĠ▁") == ","),
        ("sentence_punct", lambda text: text.strip(" \t\r\nĠ▁") in {".", "!", "?"}),
        ("leading_space", lambda text: text.startswith((" ", "Ġ", "▁")) and len(text.strip(" Ġ▁\t\r\n")) > 0),
        ("leading_the", lambda text: text.lower() in {" the", "ġthe", "▁the"}),
        ("leading_of", lambda text: text.lower() in {" of", "ġof", "▁of"}),
        ("leading_and", lambda text: text.lower() in {" and", "ġand", "▁and"}),
        ("digit", lambda text: text.strip(" \t\r\nĠ▁").isdigit()),
    ]


def zscore(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(values, dtype=np.float32)
    if not bool(mask.any()):
        return out
    subset = values[mask].astype(np.float32, copy=False)
    mean = float(subset.mean())
    std = float(subset.std())
    if std < 1e-8:
        return out
    out[mask] = (values[mask].astype(np.float32, copy=False) - mean) / std
    return out


def unique_neighbor_stats(
    rows: np.ndarray,
    cols: np.ndarray,
    head_size: int,
    vocab_floor: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(rows) == 0:
        empty = np.zeros(head_size, dtype=np.float32)
        return empty, empty.copy(), empty.copy()
    flat = rows.astype(np.int64, copy=False) * int(vocab_floor) + cols.astype(np.int64, copy=False)
    uniq, pair_counts = np.unique(flat, return_counts=True)
    uniq_rows = (uniq // int(vocab_floor)).astype(np.int64, copy=False)
    totals = np.bincount(uniq_rows, weights=pair_counts.astype(np.float64), minlength=head_size)
    distinct = np.bincount(uniq_rows, minlength=head_size).astype(np.float32)
    probs = pair_counts.astype(np.float64) / np.maximum(totals[uniq_rows], 1.0)
    entropy = np.bincount(uniq_rows, weights=-(probs * np.log2(np.maximum(probs, 1e-12))), minlength=head_size)
    return distinct.astype(np.float32), entropy.astype(np.float32), totals.astype(np.float32)


def pagerank_from_edges(
    rows: np.ndarray,
    cols: np.ndarray,
    weights: np.ndarray,
    size: int,
    iters: int = 40,
    damping: float = 0.85,
) -> np.ndarray:
    if size == 0:
        return np.empty(0, dtype=np.float32)
    out_weight = np.bincount(rows, weights=weights.astype(np.float64), minlength=size)
    valid = out_weight[rows] > 0
    rows = rows[valid]
    cols = cols[valid]
    weights = weights[valid].astype(np.float64, copy=False)
    pr = np.full(size, 1.0 / size, dtype=np.float64)
    teleport = (1.0 - damping) / size
    dangling_mask = out_weight <= 0
    for _ in range(iters):
        nxt = np.full(size, teleport, dtype=np.float64)
        if len(rows):
            contrib = pr[rows] * weights / np.maximum(out_weight[rows], 1e-12)
            np.add.at(nxt, cols, damping * contrib)
        if bool(dangling_mask.any()):
            nxt += damping * float(pr[dangling_mask].sum()) / size
        total = float(nxt.sum())
        if total > 0:
            nxt /= total
        pr = nxt
    return pr.astype(np.float32)


def approximate_spectral_features(
    rows: np.ndarray,
    cols: np.ndarray,
    weights: np.ndarray,
    size: int,
    dims: int,
    iters: int,
) -> np.ndarray:
    if size == 0 or dims <= 0 or len(rows) == 0:
        return np.zeros((size, 0), dtype=np.float32)
    sym_rows = np.concatenate([rows, cols]).astype(np.int64, copy=False)
    sym_cols = np.concatenate([cols, rows]).astype(np.int64, copy=False)
    sym_weights = np.concatenate([weights, weights]).astype(np.float32, copy=False)
    degree = np.bincount(sym_rows, weights=sym_weights.astype(np.float64), minlength=size).astype(np.float32)
    keep = (degree[sym_rows] > 0) & (degree[sym_cols] > 0)
    sym_rows = sym_rows[keep]
    sym_cols = sym_cols[keep]
    norm_weights = sym_weights[keep] / np.sqrt(degree[sym_rows] * degree[sym_cols])
    rng = np.random.default_rng(SEED)
    q = rng.standard_normal((size, dims)).astype(np.float32)
    q, _ = np.linalg.qr(q)
    q = q.astype(np.float32, copy=False)
    for _ in range(max(1, iters)):
        z = np.zeros_like(q)
        np.add.at(z, sym_cols, norm_weights[:, None] * q[sym_rows])
        q, _ = np.linalg.qr(z)
        q = q.astype(np.float32, copy=False)
    return q


def structural_graph_stats(ids: np.ndarray, vocab_size: int, top_n: int, spectral_dims: int) -> dict[str, np.ndarray]:
    ids = np.asarray(ids[: min(len(ids), STRUCTURAL_AUDIT_TOKEN_LIMIT)], dtype=np.int64)
    vocab_floor = int(max(vocab_size, int(ids.max(initial=0)) + 1))
    token_counts = counts(ids, vocab_floor)
    observed = int(np.count_nonzero(token_counts))
    head_size = min(top_n, observed)
    head = np.argsort(-token_counts)[:head_size].astype(np.int64)
    lookup = np.full(vocab_floor, -1, dtype=np.int32)
    lookup[head] = np.arange(head_size, dtype=np.int32)

    prev = ids[:-1]
    nxt = ids[1:]
    prev_head = lookup[prev]
    next_head = lookup[nxt]

    right_mask = prev_head >= 0
    right_distinct, right_entropy, right_total = unique_neighbor_stats(
        prev_head[right_mask], nxt[right_mask], head_size, vocab_floor
    )
    left_mask = next_head >= 0
    left_distinct, left_entropy, left_total = unique_neighbor_stats(
        next_head[left_mask], prev[left_mask], head_size, vocab_floor
    )

    both_mask = (prev_head >= 0) & (next_head >= 0)
    if bool(both_mask.any()):
        flat = prev_head[both_mask].astype(np.int64, copy=False) * head_size + next_head[both_mask].astype(
            np.int64, copy=False
        )
        uniq, edge_weights = np.unique(flat, return_counts=True)
        edge_rows = (uniq // head_size).astype(np.int64, copy=False)
        edge_cols = (uniq % head_size).astype(np.int64, copy=False)
        edge_weights = edge_weights.astype(np.float32, copy=False)
    else:
        edge_rows = np.empty(0, dtype=np.int64)
        edge_cols = np.empty(0, dtype=np.int64)
        edge_weights = np.empty(0, dtype=np.float32)

    pagerank = pagerank_from_edges(edge_rows, edge_cols, edge_weights, head_size)
    spectral = approximate_spectral_features(
        edge_rows,
        edge_cols,
        edge_weights,
        head_size,
        spectral_dims,
        STRUCTURAL_AUDIT_SPECTRAL_ITERS,
    )

    full = {
        "count": np.zeros(vocab_floor, dtype=np.float32),
        "right_entropy": np.zeros(vocab_floor, dtype=np.float32),
        "left_entropy": np.zeros(vocab_floor, dtype=np.float32),
        "right_distinct": np.zeros(vocab_floor, dtype=np.float32),
        "left_distinct": np.zeros(vocab_floor, dtype=np.float32),
        "pagerank": np.zeros(vocab_floor, dtype=np.float32),
        "hub_score": np.full(vocab_floor, -1.0e9, dtype=np.float32),
        "punct_score": np.full(vocab_floor, -1.0e9, dtype=np.float32),
    }
    full["count"][: len(token_counts)] = token_counts.astype(np.float32)
    for name, values in (
        ("right_entropy", right_entropy),
        ("left_entropy", left_entropy),
        ("right_distinct", right_distinct),
        ("left_distinct", left_distinct),
        ("pagerank", pagerank),
    ):
        full[name][head] = values

    head_mask = np.zeros(vocab_floor, dtype=bool)
    head_mask[head] = True
    log_count = np.zeros(vocab_floor, dtype=np.float32)
    log_count[head] = np.log1p(token_counts[head]).astype(np.float32)
    log_right_distinct = np.zeros(vocab_floor, dtype=np.float32)
    log_left_distinct = np.zeros(vocab_floor, dtype=np.float32)
    log_pagerank = np.zeros(vocab_floor, dtype=np.float32)
    log_right_distinct[head] = np.log1p(right_distinct).astype(np.float32)
    log_left_distinct[head] = np.log1p(left_distinct).astype(np.float32)
    log_pagerank[head] = np.log(np.maximum(pagerank, 1e-12)).astype(np.float32)
    full["hub_score"][head] = (
        zscore(log_count, head_mask)[head]
        + zscore(full["right_entropy"], head_mask)[head]
        + zscore(log_right_distinct, head_mask)[head]
        + zscore(log_pagerank, head_mask)[head]
    )
    full["punct_score"][head] = (
        zscore(full["right_entropy"], head_mask)[head]
        + zscore(full["left_entropy"], head_mask)[head]
        + zscore(log_right_distinct, head_mask)[head]
        + zscore(log_left_distinct, head_mask)[head]
    )

    feature_parts = [
        zscore(log_count, head_mask)[head],
        zscore(full["right_entropy"], head_mask)[head],
        zscore(full["left_entropy"], head_mask)[head],
        zscore(log_right_distinct, head_mask)[head],
        zscore(log_left_distinct, head_mask)[head],
        zscore(log_pagerank, head_mask)[head],
    ]
    features = np.column_stack(feature_parts).astype(np.float32)
    if spectral.shape[1]:
        features = np.concatenate([features, spectral.astype(np.float32)], axis=1)
    norm = np.linalg.norm(features, axis=1, keepdims=True)
    features = features / np.maximum(norm, 1e-8)

    return {
        "head": head,
        "token_counts": token_counts.astype(np.int64, copy=False),
        "right_total": right_total,
        "left_total": left_total,
        "edge_rows": edge_rows,
        "edge_cols": edge_cols,
        "edge_weights": edge_weights,
        "features": features,
        **full,
    }


def rank_array(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores)
    ranks = np.empty(len(scores), dtype=np.int64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.int64)
    return ranks


def structural_anchor_audit(task) -> dict[str, object]:
    source_vocab = task.source_adapter.spec.vocab_size
    target_vocab = task.target_adapter.spec.vocab_size
    token_limit = min(len(task.cipher_ids), STRUCTURAL_AUDIT_TOKEN_LIMIT)
    ref_limit = min(len(task.ref_ids), STRUCTURAL_AUDIT_TOKEN_LIMIT)
    print(
        f"structural_anchor_audit: top_n={STRUCTURAL_AUDIT_TOP_N} tokens={token_limit} "
        f"ref_tokens={ref_limit} spectral_dims={STRUCTURAL_AUDIT_SPECTRAL_DIMS}",
        flush=True,
    )
    cipher_stats = structural_graph_stats(
        task.cipher_ids[:token_limit],
        max(source_vocab, int(task.cipher_ids.max(initial=0)) + 1),
        STRUCTURAL_AUDIT_TOP_N,
        STRUCTURAL_AUDIT_SPECTRAL_DIMS,
    )
    target_stats = structural_graph_stats(
        task.ref_ids[:ref_limit],
        target_vocab,
        STRUCTURAL_AUDIT_TOP_N,
        STRUCTURAL_AUDIT_SPECTRAL_DIMS,
    )
    inverse_perm = np.empty(len(task.perm), dtype=np.int64)
    inverse_perm[np.asarray(task.perm, dtype=np.int64)] = np.arange(len(task.perm), dtype=np.int64)

    source_label_cache: dict[int, tuple[str, str, str]] = {}
    target_label_cache: dict[int, tuple[str, str, str]] = {}

    def source_label(cipher_id: int) -> tuple[int, str, str, str]:
        source_id = int(inverse_perm[int(cipher_id)]) if int(cipher_id) < len(inverse_perm) else -1
        if source_id not in source_label_cache:
            text = decode_token_text(task.source_adapter, source_id) if source_id >= 0 else ""
            source_label_cache[source_id] = (surface_class(text), text, display_surface(text))
        cls, text, shown = source_label_cache[source_id]
        return source_id, cls, text, shown

    def target_label(token_id: int) -> tuple[str, str, str]:
        token_id = int(token_id)
        if token_id not in target_label_cache:
            text = decode_token_text(task.target_adapter, token_id)
            target_label_cache[token_id] = (surface_class(text), text, display_surface(text))
        return target_label_cache[token_id]

    def top_records(stats: dict[str, np.ndarray], score_name: str, source: bool) -> list[dict[str, object]]:
        scores = stats[score_name]
        order = np.argsort(-scores)[:STRUCTURAL_AUDIT_REPORT_TOP]
        records: list[dict[str, object]] = []
        for rank, token_id in enumerate(order, start=1):
            if scores[token_id] <= -1.0e8:
                continue
            if source:
                src_id, cls, _, shown = source_label(int(token_id))
                record = {"rank": rank, "cipher_id": int(token_id), "source_id": src_id}
            else:
                cls, _, shown = target_label(int(token_id))
                record = {"rank": rank, "target_id": int(token_id)}
            record.update(
                {
                    "score": float(scores[token_id]),
                    "count": int(stats["count"][token_id]),
                    "pagerank": float(stats["pagerank"][token_id]),
                    "right_entropy": float(stats["right_entropy"][token_id]),
                    "left_entropy": float(stats["left_entropy"][token_id]),
                    "right_distinct": int(stats["right_distinct"][token_id]),
                    "left_distinct": int(stats["left_distinct"][token_id]),
                    "class": cls,
                    "surface": shown,
                }
            )
            records.append(record)
        return records

    score_names = ["count", "pagerank", "hub_score", "right_entropy", "right_distinct", "punct_score"]
    cipher_top = {name: top_records(cipher_stats, name, source=True) for name in score_names}
    target_top = {name: top_records(target_stats, name, source=False) for name in score_names}

    source_counts = counts(task.secret_ids[:token_limit], source_vocab)
    source_head = np.argsort(-source_counts)[: min(STRUCTURAL_AUDIT_LABEL_TOP, np.count_nonzero(source_counts))]
    score_ranks = {name: rank_array(cipher_stats[name]) for name in score_names}
    anchor_results: list[dict[str, object]] = []
    for anchor_name, predicate in anchor_patterns():
        best_source = -1
        best_text = ""
        for source_id in source_head:
            text = decode_token_text(task.source_adapter, int(source_id))
            if predicate(text):  # type: ignore[operator]
                best_source = int(source_id)
                best_text = text
                break
        if best_source < 0:
            anchor_results.append({"anchor": anchor_name, "found": False})
            continue
        cipher_id = int(task.perm[best_source])
        result = {
            "anchor": anchor_name,
            "found": True,
            "source_id": best_source,
            "cipher_id": cipher_id,
            "source_count": int(source_counts[best_source]),
            "class": surface_class(best_text),
            "surface": display_surface(best_text),
            "metric_ranks": {name: int(score_ranks[name][cipher_id]) for name in score_names},
            "metric_scores": {name: float(cipher_stats[name][cipher_id]) for name in score_names},
        }
        anchor_results.append(result)

    c_head = cipher_stats["head"][: min(1000, len(cipher_stats["head"]))]
    t_head = target_stats["head"][: min(1000, len(target_stats["head"]))]
    c_features = cipher_stats["features"][: len(c_head)]
    t_features = target_stats["features"][: len(t_head)]
    class_match_summary: dict[str, dict[str, float]] = {}
    nearest_examples: list[dict[str, object]] = []
    if len(c_head) and len(t_head):
        sims = c_features @ t_features.T
        nearest = np.argmax(sims, axis=1)
        pred_classes = []
        true_classes = []
        for idx, cipher_id in enumerate(c_head):
            _, true_cls, _, shown = source_label(int(cipher_id))
            pred_token = int(t_head[int(nearest[idx])])
            pred_cls, _, pred_shown = target_label(pred_token)
            pred_classes.append(pred_cls)
            true_classes.append(true_cls)
            if len(nearest_examples) < STRUCTURAL_AUDIT_REPORT_TOP:
                nearest_examples.append(
                    {
                        "cipher_id": int(cipher_id),
                        "source_id": int(inverse_perm[int(cipher_id)]),
                        "source_class": true_cls,
                        "source_surface": shown,
                        "nearest_target_id": pred_token,
                        "nearest_target_class": pred_cls,
                        "nearest_target_surface": pred_shown,
                        "similarity": float(sims[idx, nearest[idx]]),
                    }
                )
        pred_arr = np.asarray(pred_classes)
        true_arr = np.asarray(true_classes)
        for k in (100, 500, 1000):
            n = min(k, len(true_arr))
            if n:
                class_match_summary[f"top{n}"] = {
                    "exact_class_accuracy": float(np.mean(pred_arr[:n] == true_arr[:n])),
                    "leading_space_rate_true": float(np.mean(np.char.startswith(true_arr[:n].astype(str), "leading_space"))),
                    "leading_space_rate_pred": float(np.mean(np.char.startswith(pred_arr[:n].astype(str), "leading_space"))),
                }

    report: dict[str, object] = {
        "source_tokenizer": task.source_adapter.spec.name,
        "target_tokenizer": task.target_adapter.spec.name,
        "token_limit": int(token_limit),
        "ref_limit": int(ref_limit),
        "top_n": int(STRUCTURAL_AUDIT_TOP_N),
        "spectral_dims": int(STRUCTURAL_AUDIT_SPECTRAL_DIMS),
        "cipher_top": cipher_top,
        "target_top": target_top,
        "anchor_results": anchor_results,
        "nearest_target_class_summary": class_match_summary,
        "nearest_target_examples": nearest_examples,
    }

    print("--- structural top cipher nodes ---", flush=True)
    for name in score_names:
        print(f"[cipher:{name}]", flush=True)
        for rec in cipher_top[name][: min(8, len(cipher_top[name]))]:
            print(
                f"  #{rec['rank']} c={rec['cipher_id']} src={rec['source_id']} "
                f"count={rec['count']} pr={rec['pagerank']:.3e} cls={rec['class']} surf={rec['surface']}",
                flush=True,
            )
    print("--- structural anchor oracle ranks ---", flush=True)
    for result in anchor_results:
        if not result.get("found"):
            print(f"  {result['anchor']}: not found in top source labels", flush=True)
            continue
        ranks = result["metric_ranks"]
        print(
            f"  {result['anchor']}: c={result['cipher_id']} src={result['source_id']} "
            f"count={result['source_count']} cls={result['class']} surf={result['surface']} "
            f"ranks={ranks}",
            flush=True,
        )
    print("--- nearest target class summary ---", flush=True)
    print(json.dumps(class_match_summary, indent=2), flush=True)
    return report


def find_target_surface_id(adapter, token_counts: np.ndarray, surfaces: tuple[str, ...]) -> int | None:
    wanted = set(surfaces)
    best_id: int | None = None
    best_count = -1
    for token_id in np.argsort(-token_counts):
        text = decode_token_text(adapter, int(token_id))
        if text not in wanted:
            continue
        count = int(token_counts[int(token_id)])
        if count > best_count:
            best_id = int(token_id)
            best_count = count
    return best_id


def first_ranked_unused(scores: np.ndarray, used: set[int]) -> int | None:
    for token_id in np.argsort(-scores):
        token_id = int(token_id)
        if scores[token_id] <= -1.0e8:
            return None
        if token_id not in used:
            return token_id
    return None


def period_like_candidate(stats: dict[str, np.ndarray], used: set[int]) -> int | None:
    punct_rank = rank_array(stats["punct_score"])
    right_entropy_rank = rank_array(stats["right_entropy"])
    for token_id in np.argsort(-stats["pagerank"]):
        token_id = int(token_id)
        if stats["pagerank"][token_id] <= 0:
            return None
        if token_id in used:
            continue
        if punct_rank[token_id] <= 12 and right_entropy_rank[token_id] >= 20:
            return token_id
    return None


def structural_seed_anchor_pairs(
    cipher_ids: np.ndarray,
    ref_ids: np.ndarray,
    target_adapter,
    source_vocab_size: int,
    target_vocab_size: int,
) -> dict[int, int]:
    if not STRUCTURAL_SEED_ANCHORS:
        return {}
    stats = structural_graph_stats(
        cipher_ids,
        max(source_vocab_size, int(cipher_ids.max(initial=0)) + 1),
        STRUCTURAL_AUDIT_TOP_N,
        0,
    )
    target_counts = counts(ref_ids, target_vocab_size)
    target_by_surface: dict[str, int] = {}
    for surface in (".", ",", " the", *STRUCTURAL_SEED_COMMON_SURFACES):
        token_id = find_target_surface_id(target_adapter, target_counts, (surface,))
        if token_id is not None:
            target_by_surface[surface] = token_id

    anchors: dict[int, int] = {}
    used_c: set[int] = set()
    used_p: set[int] = set()

    def add(surface: str, cipher_id: int | None) -> None:
        if cipher_id is None:
            return
        target_id = target_by_surface.get(surface)
        if target_id is None or cipher_id in used_c or target_id in used_p:
            return
        anchors[int(cipher_id)] = int(target_id)
        used_c.add(int(cipher_id))
        used_p.add(int(target_id))

    add(" the", first_ranked_unused(stats["hub_score"], used_c))
    add(",", first_ranked_unused(stats["punct_score"], used_c))
    add(".", period_like_candidate(stats, used_c))
    for surface in STRUCTURAL_SEED_COMMON_SURFACES:
        add(surface, first_ranked_unused(stats["hub_score"], used_c))

    if anchors:
        print("structural_seed_anchors:", flush=True)
        for c, p in anchors.items():
            print(
                f"  c={c} -> p={p} target={display_surface(decode_token_text(target_adapter, p))}",
                flush=True,
            )
    return anchors


def apply_fixed_anchors(mapping: np.ndarray, anchors: dict[int, int]) -> None:
    for c, p in anchors.items():
        if c < len(mapping):
            mapping[int(c)] = int(p)


def standardize_feature_column(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32, copy=False)
    mask = np.isfinite(values) & (values != 0)
    out = np.zeros_like(values, dtype=np.float32)
    if not bool(mask.any()):
        return out
    mean = float(values[mask].mean())
    std = float(values[mask].std())
    if std < 1e-8:
        return out
    out[mask] = (values[mask] - mean) / std
    return out


def structural_feature_matrix(stats: dict[str, np.ndarray], focus: np.ndarray) -> np.ndarray:
    columns = [
        stats["right_entropy"][focus],
        stats["left_entropy"][focus],
        np.log1p(stats["right_distinct"][focus]),
        np.log1p(stats["left_distinct"][focus]),
        np.log(np.maximum(stats["pagerank"][focus], 1e-12)),
    ]
    features = np.column_stack([standardize_feature_column(col) for col in columns]).astype(np.float32)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, 1e-8)


def torch_context_maps(ids: np.ndarray, focus: np.ndarray, anchors: np.ndarray, device: str, offset: int = 1):
    vocab_floor = int(max(int(ids.max(initial=0)), int(focus.max(initial=0)), int(anchors.max(initial=0)))) + 1
    focus_lookup = torch.full((vocab_floor,), -1, dtype=torch.int32, device=device)
    anchor_lookup = torch.full((vocab_floor,), -1, dtype=torch.int32, device=device)
    focus_lookup[torch.as_tensor(focus.astype(np.int64), dtype=torch.long, device=device)] = torch.arange(
        len(focus), dtype=torch.int32, device=device
    )
    anchor_lookup[torch.as_tensor(anchors.astype(np.int64), dtype=torch.long, device=device)] = torch.arange(
        len(anchors), dtype=torch.int32, device=device
    )
    left_flat = torch.zeros(len(focus) * len(anchors), dtype=torch.float32, device=device)
    right_flat = torch.zeros_like(left_flat)
    base = len(anchors)
    for start in range(0, max(0, len(ids) - offset), TORCH_CONTEXT_CHUNK):
        stop = min(len(ids) - offset, start + TORCH_CONTEXT_CHUNK)
        prev = torch.as_tensor(ids[start:stop].astype(np.int64, copy=False), dtype=torch.long, device=device)
        nxt = torch.as_tensor(
            ids[start + offset : stop + offset].astype(np.int64, copy=False),
            dtype=torch.long,
            device=device,
        )
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

    return left_flat.view(len(focus), len(anchors)), right_flat.view(len(focus), len(anchors))


def normalize_features(x, token_counts: np.ndarray, device: str):
    counts_t = torch.as_tensor(np.maximum(1.0, token_counts.astype(np.float32)), dtype=torch.float32, device=device)
    x.div_(torch.sqrt(counts_t)[:, None])
    norm = torch.linalg.vector_norm(x, dim=1).clamp_min(1e-12)
    x.div_(norm[:, None])
    return x


def learn_skip_weight(
    c_left,
    c_right,
    c_left2,
    c_right2,
    p_left,
    p_right,
    p_left2,
    p_right2,
    c_anchor_rows: np.ndarray,
    p_focus: np.ndarray,
    p_anchors: np.ndarray,
    device: str,
) -> float:
    p_positions = {int(token_id): row for row, token_id in enumerate(p_focus)}
    c_rows: list[int] = []
    p_rows: list[int] = []
    for anchor_idx, p_token in enumerate(p_anchors):
        p_row = p_positions.get(int(p_token))
        if p_row is None:
            continue
        c_rows.append(int(c_anchor_rows[anchor_idx]))
        p_rows.append(p_row)
        if len(c_rows) >= LEARN_WEIGHT_SEEDS:
            break
    if len(c_rows) < 64:
        return SKIP_CONTEXT_WEIGHT

    c_idx = torch.as_tensor(c_rows, dtype=torch.long, device=device)
    p_idx = torch.as_tensor(p_rows, dtype=torch.long, device=device)
    with torch.enable_grad():
        c_base = torch.cat([c_left[c_idx], c_right[c_idx]], dim=1).detach()
        p_base = torch.cat([p_left[p_idx], p_right[p_idx]], dim=1).detach()
        c_skip = torch.cat([c_left2[c_idx], c_right2[c_idx]], dim=1).detach()
        p_skip = torch.cat([p_left2[p_idx], p_right2[p_idx]], dim=1).detach()
        raw_weight = torch.tensor(0.54132485, dtype=torch.float32, device=device, requires_grad=True)
        optimizer = torch.optim.Adam([raw_weight], lr=LEARN_WEIGHT_LR)
        target = torch.arange(len(c_rows), dtype=torch.long, device=device)
        for _ in range(LEARN_WEIGHT_STEPS):
            optimizer.zero_grad(set_to_none=True)
            weight = torch.nn.functional.softplus(raw_weight).clamp(0.05, 4.0)
            c_vec = torch.cat([c_base, c_skip * weight], dim=1)
            p_vec = torch.cat([p_base, p_skip * weight], dim=1)
            c_vec = c_vec / torch.linalg.vector_norm(c_vec, dim=1, keepdim=True).clamp_min(1e-12)
            p_vec = p_vec / torch.linalg.vector_norm(p_vec, dim=1, keepdim=True).clamp_min(1e-12)
            logits = (c_vec @ p_vec.T) / LEARN_WEIGHT_TEMP
            loss = 0.5 * (
                torch.nn.functional.cross_entropy(logits, target)
                + torch.nn.functional.cross_entropy(logits.T, target)
            )
            loss.backward()
            optimizer.step()
        learned = float(torch.nn.functional.softplus(raw_weight).clamp(0.05, 4.0).detach().cpu())
        del c_base, p_base, c_skip, p_skip, c_vec, p_vec, logits, loss
    return learned


def topk_edges(
    c_vec,
    p_vec,
    c_focus: np.ndarray,
    p_focus: np.ndarray,
    c_log: np.ndarray,
    p_log: np.ndarray,
    mapping: np.ndarray,
    p_rank: np.ndarray,
    candidate_window: int,
    id_scale: float,
    id_proximity_weight: float,
    device: str,
) -> list[tuple[float, int, int]]:
    p_log_t = torch.as_tensor(p_log[p_focus].astype(np.float32), dtype=torch.float32, device=device)
    p_rank_t = torch.as_tensor(p_rank[p_focus].astype(np.int64), dtype=torch.long, device=device)
    p_id_t = torch.as_tensor(p_focus.astype(np.float32), dtype=torch.float32, device=device)
    id_norm = math.log1p(max(int(c_focus.max(initial=0)), int(p_focus.max(initial=0)), 1))
    edges: list[tuple[float, int, int]] = []
    k = min(TORCH_TOPK, len(p_focus))
    for start in range(0, len(c_focus), TORCH_BATCH_SIZE):
        stop = min(len(c_focus), start + TORCH_BATCH_SIZE)
        c_ids = c_focus[start:stop]
        sim = c_vec[start:stop] @ p_vec.T
        c_log_t = torch.as_tensor(c_log[c_ids].astype(np.float32), dtype=torch.float32, device=device)
        sim -= FREQ_WEIGHT * torch.abs(c_log_t[:, None] - p_log_t[None, :])
        id_delta = None
        if UNSHUFFLED_SOURCE_IDS and id_proximity_weight:
            c_id_t = torch.as_tensor(c_ids.astype(np.float32), dtype=torch.float32, device=device)
            if ID_PROXIMITY_MODE == "scaled":
                id_center = c_id_t * float(id_scale)
                p_id_for_delta = p_id_t
            elif ID_PROXIMITY_MODE == "rank":
                id_center = c_id_t
                p_id_for_delta = p_rank_t.to(torch.float32)
            else:
                id_center = c_id_t
                p_id_for_delta = p_id_t
            id_delta = torch.abs(id_center[:, None] - p_id_for_delta[None, :])
            sim -= id_proximity_weight * torch.log1p(id_delta) / id_norm
            if ID_EXACT_BONUS:
                sim += ID_EXACT_BONUS * (id_delta == 0).to(torch.float32)
        if candidate_window > 0:
            centers = torch.as_tensor(p_rank[mapping[c_ids]].astype(np.int64), dtype=torch.long, device=device)
            mask = torch.abs(p_rank_t[None, :] - centers[:, None]) <= candidate_window
            if UNSHUFFLED_SOURCE_IDS and ID_CANDIDATE_WINDOW > 0:
                if id_delta is None:
                    c_id_t = torch.as_tensor(c_ids.astype(np.float32), dtype=torch.float32, device=device)
                    if ID_PROXIMITY_MODE == "scaled":
                        id_center = c_id_t * float(id_scale)
                        p_id_for_delta = p_id_t
                    elif ID_PROXIMITY_MODE == "rank":
                        id_center = c_id_t
                        p_id_for_delta = p_rank_t.to(torch.float32)
                    else:
                        id_center = c_id_t
                        p_id_for_delta = p_id_t
                    id_delta = torch.abs(id_center[:, None] - p_id_for_delta[None, :])
                mask = mask | (id_delta <= float(ID_CANDIDATE_WINDOW))
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


def dense_bigram_counts(ids: np.ndarray, nodes: np.ndarray, vocab_floor: int, offset: int = 1) -> np.ndarray:
    lookup = np.full(vocab_floor, -1, dtype=np.int32)
    valid = nodes < vocab_floor
    lookup[nodes[valid]] = np.arange(len(nodes), dtype=np.int32)[valid]
    prev = lookup[ids[:-offset]]
    nxt = lookup[ids[offset:]]
    mask = (prev >= 0) & (nxt >= 0)
    k = len(nodes)
    if not bool(mask.any()):
        return np.zeros((k, k), dtype=np.float32)
    flat = prev[mask].astype(np.int64, copy=False) * k + nxt[mask].astype(np.int64, copy=False)
    return np.bincount(flat, minlength=k * k).reshape(k, k).astype(np.float32)


def bigram_swap_delta(c_big: np.ndarray, p_log: np.ndarray, perm: np.ndarray, a: int, b: int) -> float:
    pa = int(perm[a])
    pb = int(perm[b])
    old = (
        float(c_big[a, :] @ p_log[pa, perm])
        + float(c_big[b, :] @ p_log[pb, perm])
        + float(c_big[:, a] @ p_log[perm, pa])
        + float(c_big[:, b] @ p_log[perm, pb])
    )
    new_perm = perm.copy()
    new_perm[a], new_perm[b] = new_perm[b], new_perm[a]
    new = (
        float(c_big[a, :] @ p_log[pb, new_perm])
        + float(c_big[b, :] @ p_log[pa, new_perm])
        + float(c_big[:, a] @ p_log[new_perm, pb])
        + float(c_big[:, b] @ p_log[new_perm, pa])
    )
    for i in (a, b):
        for j in (a, b):
            old -= float(c_big[i, j] * p_log[int(perm[i]), int(perm[j])])
            new -= float(c_big[i, j] * p_log[int(new_perm[i]), int(new_perm[j])])
    return new - old


def refine_with_bigram_objective(
    cipher_ids: np.ndarray,
    ref_ids: np.ndarray,
    mapping: np.ndarray,
    c_focus: np.ndarray,
    edges: list[tuple[float, int, int]],
    target_vocab_size: int,
    p_counts: np.ndarray,
) -> np.ndarray:
    if not BIGRAM_OBJECTIVE_REFINE:
        return mapping
    token_budget = (
        BIGRAM_REFINE_LARGE_TOKENS
        if BIGRAM_REFINE_LARGE_TOKEN_MIN_TOKENS <= len(cipher_ids) <= BIGRAM_REFINE_LARGE_TOKEN_MAX_TOKENS
        else BIGRAM_REFINE_TOKENS
    )
    k = min(token_budget, len(c_focus))
    c_nodes = c_focus[:k]
    p_nodes_raw = mapping[c_nodes].astype(np.int64, copy=True)
    keep = np.zeros(k, dtype=bool)
    seen: set[int] = set()
    for idx, p in enumerate(p_nodes_raw):
        if int(p) in seen:
            continue
        seen.add(int(p))
        keep[idx] = True
    c_nodes = c_nodes[keep]
    p_nodes = p_nodes_raw[keep]
    k = len(c_nodes)
    if k < 64:
        return mapping

    use_skip_refine = BIGRAM_REFINE_SKIP_MIN_TOKENS <= len(cipher_ids) <= BIGRAM_REFINE_SKIP_MAX_TOKENS
    print(f"bigram_refine_token_budget={token_budget}", flush=True)
    print(f"bigram_refine_tokens={k}", flush=True)
    print(f"bigram_refine_skip={use_skip_refine}", flush=True)
    c_big = dense_bigram_counts(cipher_ids, c_nodes, len(mapping))
    p_big = dense_bigram_counts(ref_ids, p_nodes, target_vocab_size)
    if use_skip_refine:
        c_big += BIGRAM_REFINE_SKIP_WEIGHT * dense_bigram_counts(cipher_ids, c_nodes, len(mapping), offset=2)
        p_big += BIGRAM_REFINE_SKIP_WEIGHT * dense_bigram_counts(ref_ids, p_nodes, target_vocab_size, offset=2)
    row_totals = p_big.sum(axis=1, keepdims=True)
    block_probs = (p_big + BIGRAM_REFINE_ALPHA) / (row_totals + BIGRAM_REFINE_ALPHA * k)
    if BIGRAM_UNIGRAM_BACKOFF and len(cipher_ids) <= BIGRAM_UNIGRAM_BACKOFF_MAX_TOKENS:
        unigram = p_counts[p_nodes].astype(np.float32)
        unigram = (unigram + BIGRAM_REFINE_ALPHA) / (float(unigram.sum()) + BIGRAM_REFINE_ALPHA * k)
        lam = row_totals / (row_totals + BIGRAM_UNIGRAM_BACKOFF_TAU)
        block_probs = lam * block_probs + (1.0 - lam) * unigram[None, :]
        print(f"bigram_unigram_backoff_tau={BIGRAM_UNIGRAM_BACKOFF_TAU}", flush=True)
    p_log = np.log(block_probs).astype(np.float32)
    perm = np.arange(k, dtype=np.int32)
    owner = np.arange(k, dtype=np.int32)
    c_to_i = {int(c): i for i, c in enumerate(c_nodes)}
    p_to_i = {int(p): i for i, p in enumerate(p_nodes)}

    proposals: list[tuple[int, int]] = []
    seen_proposals: set[tuple[int, int]] = set()
    for _, c, p in edges:
        i = c_to_i.get(c)
        p_idx = p_to_i.get(p)
        if i is None or p_idx is None:
            continue
        key = (i, p_idx)
        if key in seen_proposals:
            continue
        seen_proposals.add(key)
        proposals.append(key)
        if len(proposals) >= BIGRAM_REFINE_MAX_PROPOSALS:
            break

    swaps = 0
    passes_run = 0
    pass_budget = (
        BIGRAM_REFINE_LARGE_PASSES
        if BIGRAM_REFINE_LARGE_TOKEN_MIN_TOKENS <= len(cipher_ids) <= BIGRAM_REFINE_LARGE_TOKEN_MAX_TOKENS
        else BIGRAM_REFINE_PASSES
    )
    for _ in range(pass_budget):
        pass_swaps = 0
        passes_run += 1
        for i, p_idx in proposals:
            j = int(owner[p_idx])
            if i == j:
                continue
            delta = bigram_swap_delta(c_big, p_log, perm, i, j)
            if delta <= 0.0:
                continue
            pi = int(perm[i])
            pj = int(perm[j])
            perm[i], perm[j] = perm[j], perm[i]
            owner[pi], owner[pj] = owner[pj], owner[pi]
            pass_swaps += 1
        swaps += pass_swaps
        if pass_swaps == 0:
            break

    if swaps:
        refined = mapping.copy()
        refined[c_nodes] = p_nodes[perm]
        mapping = refined
    print(f"bigram_refine_proposals={len(proposals)}", flush=True)
    print(f"bigram_refine_pass_budget={pass_budget}", flush=True)
    print(f"bigram_refine_passes={passes_run}", flush=True)
    print(f"bigram_refine_swaps={swaps}", flush=True)
    return mapping


def align_shuffled(cipher_ids: np.ndarray, ref_ids: np.ndarray, target_adapter) -> np.ndarray:
    target_vocab_size = target_adapter.spec.vocab_size
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)
    candidate_window = effective_candidate_window(len(cipher_ids))
    rounds = effective_rounds(len(cipher_ids))
    use_skip_context = len(cipher_ids) >= SKIP_CONTEXT_MIN_TOKENS
    use_dynamic_anchors = len(cipher_ids) <= DYNAMIC_ANCHOR_MAX_TOKENS
    print(f"candidate_window: {candidate_window}", flush=True)
    print(f"skip_context: {use_skip_context}", flush=True)
    print(f"dynamic_anchors: {use_dynamic_anchors}", flush=True)
    c_counts = counts(cipher_ids, int(max(target_vocab_size, int(cipher_ids.max()) + 1)))
    p_counts = counts(ref_ids, target_vocab_size)
    c_order_all = np.argsort(-c_counts)
    p_order_all = np.argsort(-p_counts)
    c_focus = c_order_all[: min(TOP_TOKENS, np.count_nonzero(c_counts))].astype(np.int64)
    p_focus = p_order_all[: min(TOP_TOKENS, np.count_nonzero(p_counts))].astype(np.int64)
    id_scale = target_vocab_size / max(1, len(c_counts))

    mapping = np.zeros(max(len(c_counts), target_vocab_size), dtype=np.int64)
    init = c_order_all[: len(p_order_all)]
    mapping[init] = p_order_all[: len(init)]
    if UNSHUFFLED_SOURCE_IDS and ID_INIT_MODE != "freq":
        source_ids = np.arange(len(c_counts), dtype=np.float64)
        if ID_INIT_MODE == "scaled":
            mapped = np.rint(source_ids * id_scale).astype(np.int64)
        else:
            mapped = source_ids.astype(np.int64)
        mapping[: len(c_counts)] = np.clip(mapped, 0, target_vocab_size - 1)
    apply_id_lock(mapping, target_vocab_size, p_order_all)
    fixed_anchors = structural_seed_anchor_pairs(
        cipher_ids,
        ref_ids,
        target_adapter,
        len(c_counts),
        target_vocab_size,
    )
    if STRUCTURAL_SEED_LOCK:
        apply_fixed_anchors(mapping, fixed_anchors)
    c_struct = p_struct = None
    if STRUCTURAL_FEATURES:
        c_stats = structural_graph_stats(
            cipher_ids,
            len(c_counts),
            STRUCTURAL_FEATURE_TOP_N,
            0,
        )
        p_stats = structural_graph_stats(
            ref_ids,
            target_vocab_size,
            STRUCTURAL_FEATURE_TOP_N,
            0,
        )
        c_struct = structural_feature_matrix(c_stats, c_focus)
        p_struct = structural_feature_matrix(p_stats, p_focus)
        print(
            f"structural_features: top_n={STRUCTURAL_FEATURE_TOP_N} weight={STRUCTURAL_FEATURE_WEIGHT}",
            flush=True,
        )

    c_log = np.log(np.maximum(c_counts, 1) / max(1, int(c_counts.sum())))
    p_log = np.log(np.maximum(p_counts, 1) / max(1, int(p_counts.sum())))
    p_rank = np.empty(target_vocab_size, dtype=np.int64)
    p_rank[p_order_all] = np.arange(target_vocab_size)
    c_focus_pos = {int(token_id): row for row, token_id in enumerate(c_focus)}
    anchor_rows = np.arange(min(ANCHORS, len(c_focus)), dtype=np.int64)

    for round_idx in range(rounds):
        id_proximity_weight = effective_id_proximity_weight(round_idx, rounds)
        c_anchors = c_focus[anchor_rows]
        p_anchors = mapping[c_anchors]
        print(f"round {round_idx + 1}/{rounds}: focus={len(c_focus)} anchors={len(c_anchors)}", flush=True)
        if ID_PROXIMITY_FINAL_WEIGHT is not None:
            print(f"id_proximity_weight: {id_proximity_weight:.4f}", flush=True)
        with torch.no_grad():
            c_left, c_right = torch_context_maps(cipher_ids, c_focus, c_anchors, device)
            p_left, p_right = torch_context_maps(ref_ids, p_focus, p_anchors, device)
            c_parts = [c_left, c_right]
            p_parts = [p_left, p_right]
            if use_skip_context:
                c_left2, c_right2 = torch_context_maps(cipher_ids, c_focus, c_anchors, device, offset=2)
                p_left2, p_right2 = torch_context_maps(ref_ids, p_focus, p_anchors, device, offset=2)
                skip_weight = SKIP_CONTEXT_WEIGHT
                if LEARN_SKIP_WEIGHT:
                    skip_weight = learn_skip_weight(
                        c_left,
                        c_right,
                        c_left2,
                        c_right2,
                        p_left,
                        p_right,
                        p_left2,
                        p_right2,
                        anchor_rows,
                        p_focus,
                        p_anchors,
                        device,
                    )
                    print(f"learned_skip_weight: {skip_weight:.4f}", flush=True)
                c_left2.mul_(skip_weight)
                c_right2.mul_(skip_weight)
                p_left2.mul_(skip_weight)
                p_right2.mul_(skip_weight)
                c_parts.extend([c_left2, c_right2])
                p_parts.extend([p_left2, p_right2])
            c_vec = normalize_features(torch.cat(c_parts, dim=1), c_counts[c_focus], device)
            p_vec = normalize_features(torch.cat(p_parts, dim=1), p_counts[p_focus], device)
            if c_struct is not None and p_struct is not None:
                c_extra = torch.as_tensor(c_struct, dtype=torch.float32, device=device) * STRUCTURAL_FEATURE_WEIGHT
                p_extra = torch.as_tensor(p_struct, dtype=torch.float32, device=device) * STRUCTURAL_FEATURE_WEIGHT
                c_vec = torch.cat([c_vec, c_extra], dim=1)
                p_vec = torch.cat([p_vec, p_extra], dim=1)
                c_vec = c_vec / torch.linalg.vector_norm(c_vec, dim=1, keepdim=True).clamp_min(1e-12)
                p_vec = p_vec / torch.linalg.vector_norm(p_vec, dim=1, keepdim=True).clamp_min(1e-12)
            del c_parts, p_parts, c_left, c_right, p_left, p_right
            if use_skip_context:
                del c_left2, c_right2, p_left2, p_right2
            edges = topk_edges(
                c_vec,
                p_vec,
                c_focus,
                p_focus,
                c_log,
                p_log,
                mapping,
                p_rank,
                candidate_window,
                id_scale,
                id_proximity_weight,
                device,
            )
            del c_vec, p_vec
            if device == "cuda":
                torch.cuda.empty_cache()

        edges.sort(reverse=True)
        used_c: set[int] = set(fixed_anchors) if STRUCTURAL_SEED_LOCK else set()
        used_p: set[int] = set(fixed_anchors.values()) if STRUCTURAL_SEED_LOCK else set()
        assigned_p_by_c: dict[int, int] = dict(fixed_anchors) if STRUCTURAL_SEED_LOCK else {}
        assigned_c_by_p: dict[int, int] = {p: c for c, p in fixed_anchors.items()} if STRUCTURAL_SEED_LOCK else {}
        assigned_score_by_c: dict[int, float] = {c: 1.0e9 for c in fixed_anchors} if STRUCTURAL_SEED_LOCK else {}
        score_by_c: dict[int, dict[int, float]] | None = {} if use_dynamic_anchors and ENABLE_DYNAMIC_ASSIGNMENT_SWAPS else None
        if score_by_c is not None:
            for score, c, p in edges:
                score_by_c.setdefault(c, {})[p] = score

        for score, c, p in edges:
            if c in used_c or p in used_p:
                continue
            assigned_p_by_c[c] = p
            assigned_c_by_p[p] = c
            assigned_score_by_c[c] = score
            used_c.add(c)
            used_p.add(p)

        swap_count = 0
        if score_by_c is not None:
            for score, c, p in edges:
                current_p = assigned_p_by_c.get(c)
                other_c = assigned_c_by_p.get(p)
                if current_p is None or other_c is None or current_p == p or other_c == c:
                    continue
                if STRUCTURAL_SEED_LOCK and (c in fixed_anchors or other_c in fixed_anchors):
                    continue
                other_scores = score_by_c.get(other_c)
                if other_scores is None:
                    continue
                other_new_score = other_scores.get(current_p)
                if other_new_score is None:
                    continue
                current_score = assigned_score_by_c[c]
                other_current_score = assigned_score_by_c[other_c]
                if score + other_new_score <= current_score + other_current_score + ASSIGNMENT_SWAP_MIN_GAIN:
                    continue
                assigned_p_by_c[c] = p
                assigned_p_by_c[other_c] = current_p
                assigned_c_by_p[p] = c
                assigned_c_by_p[current_p] = other_c
                assigned_score_by_c[c] = score
                assigned_score_by_c[other_c] = other_new_score
                swap_count += 1
            print(f"assignment_swaps={swap_count}", flush=True)

        assigned_scores = [(score, c) for c, score in assigned_score_by_c.items()]
        next_mapping = mapping.copy()
        for c, p in assigned_p_by_c.items():
            next_mapping[c] = p
        apply_id_lock(next_mapping, target_vocab_size, p_order_all)
        if STRUCTURAL_SEED_LOCK:
            apply_fixed_anchors(next_mapping, fixed_anchors)
        mapping = next_mapping
        if round_idx == rounds - 1 and (use_dynamic_anchors or BIGRAM_REFINE_ALL_SCALES):
            mapping = refine_with_bigram_objective(
                cipher_ids,
                ref_ids,
                mapping,
                c_focus,
                edges,
                target_vocab_size,
                p_counts,
            )
            apply_id_lock(mapping, target_vocab_size, p_order_all)
            if STRUCTURAL_SEED_LOCK:
                apply_fixed_anchors(mapping, fixed_anchors)
        if use_dynamic_anchors and len(assigned_scores) >= 64:
            assigned_scores.sort(reverse=True)
            next_anchor_rows: list[int] = []
            seen_rows: set[int] = set()
            for _, c in assigned_scores:
                row = c_focus_pos.get(int(c))
                if row is None or row in seen_rows:
                    continue
                next_anchor_rows.append(row)
                seen_rows.add(row)
                if len(next_anchor_rows) >= min(ANCHORS, len(c_focus)):
                    break
            if len(next_anchor_rows) >= 64:
                anchor_rows = np.asarray(next_anchor_rows, dtype=np.int64)
                print(f"anchor_refresh={len(anchor_rows)}", flush=True)
        print(f"assigned={len(used_c)}", flush=True)
    return mapping


def main() -> None:
    t0 = time.time()
    task = load_task(
        SOURCE_TOKENIZER,
        TARGET_TOKENIZER,
        target_tokens=TARGET_TOKENS,
        reference_tokens=REFERENCE_TOKENS,
        seed=SEED,
    )
    print(f"source_tokenizer: {task.source_adapter.spec.name}")
    print(f"target_tokenizer: {task.target_adapter.spec.name}")
    cipher_stream = task.secret_ids if UNSHUFFLED_SOURCE_IDS else task.cipher_ids
    if DESHUFFLE_BY_FREQUENCY:
        cipher_stream = frequency_rank_ids(task.cipher_ids, DESHUFFLE_FREQ_ORDER)
    print(f"cipher_tokens: {len(cipher_stream):,}")
    print(f"reference_tokens: {len(task.ref_ids):,}")
    print(f"unshuffled_source_ids: {UNSHUFFLED_SOURCE_IDS}")
    print(f"deshuffle_by_frequency: {DESHUFFLE_BY_FREQUENCY}")

    if STRUCTURAL_ANCHOR_AUDIT:
        report = structural_anchor_audit(task)
        out_dir = CACHE_DIR / "runs"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "structural_anchor_audit.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"wrote_structural_anchor_audit: {out_dir / 'structural_anchor_audit.json'}", flush=True)
        return

    mapping = align_shuffled(cipher_stream, task.ref_ids, task.target_adapter)
    mapped_sample = mapping[cipher_stream[:SAMPLE_TOKENS]]
    recovered_sample = task.target_adapter.decode(mapped_sample.tolist())
    metrics = evaluate_recovery(task, recovered_sample, SAMPLE_TOKENS)

    out_dir = CACHE_DIR / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "last_recovered.txt").write_text(recovered_sample, encoding="utf-8", errors="ignore")
    report = {
        "source_tokenizer": task.source_adapter.spec.name,
        "target_tokenizer": task.target_adapter.spec.name,
        "target_tokens": int(len(cipher_stream)),
        "reference_tokens": int(len(task.ref_ids)),
        "sample_tokens": SAMPLE_TOKENS,
        "top_tokens": TOP_TOKENS,
        "anchors": ANCHORS,
        "candidate_window": effective_candidate_window(len(task.cipher_ids)),
        "rounds": effective_rounds(len(task.cipher_ids)),
        "freq_weight": FREQ_WEIGHT,
        "torch_topk": TORCH_TOPK,
        "skip_context": len(task.cipher_ids) >= SKIP_CONTEXT_MIN_TOKENS,
        "dynamic_anchors": len(task.cipher_ids) <= DYNAMIC_ANCHOR_MAX_TOKENS,
        "dynamic_assignment_swaps": ENABLE_DYNAMIC_ASSIGNMENT_SWAPS,
        "bigram_refine_all_scales": BIGRAM_REFINE_ALL_SCALES,
        "skip_context_weight": SKIP_CONTEXT_WEIGHT,
        "learn_skip_weight": LEARN_SKIP_WEIGHT,
        "unshuffled_source_ids": UNSHUFFLED_SOURCE_IDS,
        "id_proximity_weight": ID_PROXIMITY_WEIGHT,
        "id_proximity_final_weight": ID_PROXIMITY_FINAL_WEIGHT,
        "id_proximity_decay_power": ID_PROXIMITY_DECAY_POWER,
        "id_candidate_window": ID_CANDIDATE_WINDOW,
        "id_proximity_mode": ID_PROXIMITY_MODE,
        "id_init_mode": ID_INIT_MODE,
        "id_exact_bonus": ID_EXACT_BONUS,
        "id_lock_prefix": ID_LOCK_PREFIX,
        "id_lock_mode": ID_LOCK_MODE,
        "deshuffle_by_frequency": DESHUFFLE_BY_FREQUENCY,
        "deshuffle_freq_order": DESHUFFLE_FREQ_ORDER,
        "structural_seed_anchors": STRUCTURAL_SEED_ANCHORS,
        "structural_seed_lock": STRUCTURAL_SEED_LOCK,
        "structural_seed_common_surfaces": STRUCTURAL_SEED_COMMON_SURFACES,
        "structural_features": STRUCTURAL_FEATURES,
        "structural_feature_top_n": STRUCTURAL_FEATURE_TOP_N,
        "structural_feature_weight": STRUCTURAL_FEATURE_WEIGHT,
        "elapsed_seconds": time.time() - t0,
        "metrics": metrics,
        "preview": recovered_sample[:1000],
    }
    (out_dir / "last_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("---")
    print(f"cer50k:           {metrics['cer50k']:.6f}")
    print(f"byte_lm_bpb:      {metrics['byte_lm_bpb']:.6f}")
    print(f"replacement_rate: {metrics['replacement_rate']:.8f}")
    print(f"printable_rate:   {metrics['printable_rate']:.8f}")
    print(f"elapsed_seconds:  {time.time() - t0:.1f}")
    print(f"target_tokens_M:  {len(cipher_stream) / 1e6:.3f}")
    print(f"reference_tokens_M: {len(task.ref_ids) / 1e6:.3f}")
    print(f"top_tokens:       {TOP_TOKENS}")
    print(f"anchors:          {ANCHORS}")
    print(f"rounds:           {effective_rounds(len(task.cipher_ids))}")


if __name__ == "__main__":
    main()
