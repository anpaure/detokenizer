"""Mutable detokenizer experiment.

This file is the hillclimb target. The baseline implements the current
frequency + bigram-context graph aligner for a shuffled token-ID stream. Agents
should modify this file only, run `uv run train.py`, and keep changes that lower
cer50k.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
from rapidfuzz.distance import Levenshtein

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
ENABLE_DIAGNOSTICS = os.environ.get("DETOK_DIAGNOSTICS", "1") != "0"
ORACLE_RERANKER_DIAGNOSTIC = os.environ.get("DETOK_ORACLE_RERANKER_DIAGNOSTIC", "0") == "1"
SYNTHETIC_RERANKER_DIAGNOSTIC = os.environ.get("DETOK_SYNTHETIC_RERANKER_DIAGNOSTIC", "0") == "1"

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
LCB_EXTRA_BIGRAM_PASS = False
LCB_EXTRA_BIGRAM_MAX_TOKENS = 100_000
LCB_EXTRA_BIGRAM_SHARDS = 5
LCB_EXTRA_BIGRAM_LAMBDA = 1.0
LCB_EXTRA_BIGRAM_MIN_DELTA = 0.25
LCB_EXTRA_BIGRAM_MIN_SHARD_POSITIVE = 3
LCB_EXTRA_BIGRAM_SKIP_FLOOR = -0.10
BIGRAM_UNIGRAM_BACKOFF = True
BIGRAM_UNIGRAM_BACKOFF_TAU = 1000.0
BIGRAM_UNIGRAM_BACKOFF_MAX_TOKENS = 100_000
TAIL_REPAIR_MAX_TOKENS = 100_000
TAIL_REPAIR_NODES = 2_048
TAIL_REPAIR_CONTEXTS = 8_192
TAIL_REPAIR_CANDIDATES = 8
TAIL_REPAIR_MIN_GAIN_PER_OCC = 0.30
EXTERNAL_OWNER_REPAIR = True
EXTERNAL_OWNER_MAX_TOKENS = 100_000
EXTERNAL_OWNER_NODES = 2_048
EXTERNAL_OWNER_CONTEXTS = 8_192
EXTERNAL_OWNER_CANDIDATES = 5
EXTERNAL_OWNER_MIN_COUNT = 5
EXTERNAL_OWNER_MIN_GAIN_PER_OCC = 0.35
VARIABLE_EMISSION_REPAIR = True
VARIABLE_EMISSION_MAX_TOKENS = 100_000
VARIABLE_EMISSION_NODES = 512
VARIABLE_EMISSION_CONTEXTS = 8_192
VARIABLE_EMISSION_REF_POOL = 20_000
VARIABLE_EMISSION_BIGRAM_CANDIDATES = 6
VARIABLE_EMISSION_MIN_COUNT = 10
VARIABLE_EMISSION_MIN_GAIN_PER_OCC = 5.00
VARIABLE_EMISSION_MAX_ACCEPTED = 2
VARIABLE_EMISSION_FREE_LOCAL_PAIRS = True
STRING_LEXICON_REPAIR = False
STRING_LEXICON_MAX_TOKENS = 100_000
STRING_LEXICON_NODES = 512
STRING_LEXICON_BIGRAMS = 8_192
STRING_LEXICON_MIN_PAIR_COUNT = 4
STRING_LEXICON_MIN_PAIR_COVERAGE = 0.35
STRING_LEXICON_MAX_COMBO_CHARS = 32
STRING_LEXICON_MAX_SHIFT = 4
STRING_LEXICON_CONTEXT_RADIUS = 3
STRING_LEXICON_MAX_CONTEXTS = 768
STRING_LEXICON_MIN_GAIN_PER_BYTE = 0.050
STRING_LEXICON_MAX_ACCEPTED = 2
STRING_LEXICON_ALLOW_EMPTY_SPLITS = False
STRING_LEXICON_FORMAT_ONLY = True
STRING_CANDIDATE_REPAIR = False
STRING_CANDIDATE_MAX_TOKENS = 100_000
STRING_CANDIDATE_NODES = 128
STRING_CANDIDATE_REF_TOKENS = 8_192
STRING_CANDIDATE_RAW_SPANS = 2_048
STRING_CANDIDATE_SUBSTRINGS = 2_048
STRING_CANDIDATE_MAX_PER_TOKEN = 32
STRING_CANDIDATE_MAX_CONTEXTS = 512
STRING_CANDIDATE_CONTEXT_RADIUS = 4
STRING_CANDIDATE_MAX_CHARS = 24
STRING_CANDIDATE_MIN_GAIN_PER_BYTE = 0.10
STRING_CANDIDATE_MAX_ACCEPTED = 1
STRING_CANDIDATE_FORMAT_ONLY = True
SINKHORN_EDGE_REWEIGHT = False
SINKHORN_MAX_TOKENS = 100_000
SINKHORN_NODES = 2_048
SINKHORN_ITERS = 20
SINKHORN_TEMP = 0.08
SINKHORN_WEIGHT = 0.25
ORACLE_CANDIDATE_DIAGNOSTICS = True
ORACLE_CANDIDATE_TOP_N = 4096
ORACLE_CANDIDATE_KS = (1, 2, 5, 16, 64)
ORACLE_RERANKER_TOP_K = 64
ORACLE_RERANKER_STEPS = 220
ORACLE_RERANKER_LR = 0.08
ORACLE_RERANKER_L2 = 1.0e-3
ORACLE_RERANKER_WEIGHT_CAP = 25.0
SYNTHETIC_RERANKER_TOKENS = int(os.environ.get("DETOK_SYNTHETIC_RERANKER_TOKENS", "100000"))
SYNTHETIC_RERANKER_REF_TOKENS = int(os.environ.get("DETOK_SYNTHETIC_RERANKER_REF_TOKENS", "500000"))
SYNTHETIC_RERANKER_TASKS = tuple(
    item.strip()
    for item in os.environ.get("DETOK_SYNTHETIC_RERANKER_TASKS", "mixed,merge,split").split(",")
    if item.strip()
)
SYNTHETIC_RERANKER_MIN_TOP1_GAIN = 0.03
POST_REPAIR_EDGE_DIAGNOSTIC = os.environ.get("DETOK_POST_REPAIR_EDGE_DIAGNOSTIC", "0") == "1"
POST_REPAIR_REFRESH_REPAIR = os.environ.get("DETOK_POST_REPAIR_REFRESH_REPAIR", "0") == "1"

LAST_FINAL_EDGES: list[tuple[float, int, int]] = []
LAST_PRE_REPAIR_EDGES: list[tuple[float, int, int]] = []
LAST_POST_REPAIR_EDGES: list[tuple[float, int, int]] = []


def effective_candidate_window(num_cipher_tokens: int) -> int:
    return 25_000 if num_cipher_tokens >= 1_000_000 else CANDIDATE_WINDOW


def effective_rounds(num_cipher_tokens: int) -> int:
    return 8 if num_cipher_tokens >= 1_000_000 else ROUNDS


def counts(ids: np.ndarray, size: int) -> np.ndarray:
    return np.bincount(ids.astype(np.int64, copy=False), minlength=size)


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
    device: str,
) -> list[tuple[float, int, int]]:
    p_log_t = torch.as_tensor(p_log[p_focus].astype(np.float32), dtype=torch.float32, device=device)
    p_rank_t = torch.as_tensor(p_rank[p_focus].astype(np.int64), dtype=torch.long, device=device)
    edges: list[tuple[float, int, int]] = []
    k = min(TORCH_TOPK, len(p_focus))
    for start in range(0, len(c_focus), TORCH_BATCH_SIZE):
        stop = min(len(c_focus), start + TORCH_BATCH_SIZE)
        c_ids = c_focus[start:stop]
        sim = c_vec[start:stop] @ p_vec.T
        c_log_t = torch.as_tensor(c_log[c_ids].astype(np.float32), dtype=torch.float32, device=device)
        sim -= FREQ_WEIGHT * torch.abs(c_log_t[:, None] - p_log_t[None, :])
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


def build_context_edges_for_mapping(
    cipher_ids: np.ndarray,
    ref_ids: np.ndarray,
    mapping: np.ndarray,
    c_focus: np.ndarray,
    p_focus: np.ndarray,
    c_counts: np.ndarray,
    p_counts: np.ndarray,
    c_log: np.ndarray,
    p_log: np.ndarray,
    p_rank: np.ndarray,
    anchor_rows: np.ndarray,
    candidate_window: int,
    use_skip_context: bool,
    device: str,
    label: str = "",
) -> list[tuple[float, int, int]]:
    c_anchors = c_focus[anchor_rows]
    p_anchors = mapping[c_anchors]
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
                print(f"{label}learned_skip_weight: {skip_weight:.4f}", flush=True)
            c_left2.mul_(skip_weight)
            c_right2.mul_(skip_weight)
            p_left2.mul_(skip_weight)
            p_right2.mul_(skip_weight)
            c_parts.extend([c_left2, c_right2])
            p_parts.extend([p_left2, p_right2])
        c_vec = normalize_features(torch.cat(c_parts, dim=1), c_counts[c_focus], device)
        p_vec = normalize_features(torch.cat(p_parts, dim=1), p_counts[p_focus], device)
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
            device,
        )
        del c_vec, p_vec
        if device == "cuda":
            torch.cuda.empty_cache()

    if len(cipher_ids) <= SINKHORN_MAX_TOKENS:
        edges = sinkhorn_reweight_edges(edges, c_focus)
    return edges


def sinkhorn_reweight_edges(edges: list[tuple[float, int, int]], c_focus: np.ndarray) -> list[tuple[float, int, int]]:
    if not SINKHORN_EDGE_REWEIGHT or not edges:
        return edges
    selected_c = set(map(int, c_focus[: min(SINKHORN_NODES, len(c_focus))]))
    selected_edges = [(score, c, p) for score, c, p in edges if c in selected_c]
    if len(selected_edges) < 1024:
        return edges

    c_nodes = sorted({c for _, c, _ in selected_edges})
    p_nodes = sorted({p for _, _, p in selected_edges})
    if len(c_nodes) < 64 or len(p_nodes) < 64:
        return edges
    c_pos = {c: i for i, c in enumerate(c_nodes)}
    p_pos = {p: i for i, p in enumerate(p_nodes)}
    score_mat = np.full((len(c_nodes), len(p_nodes)), -np.inf, dtype=np.float32)
    for score, c, p in selected_edges:
        row = c_pos[c]
        col = p_pos[p]
        if score > score_mat[row, col]:
            score_mat[row, col] = score

    row_max = np.max(score_mat, axis=1, keepdims=True)
    row_max[~np.isfinite(row_max)] = 0.0
    weights = np.exp((score_mat - row_max) / max(SINKHORN_TEMP, 1.0e-4), where=np.isfinite(score_mat), out=np.zeros_like(score_mat))
    for _ in range(SINKHORN_ITERS):
        row_sum = weights.sum(axis=1, keepdims=True)
        weights /= np.maximum(row_sum, 1.0e-12)
        col_sum = weights.sum(axis=0, keepdims=True)
        weights /= np.maximum(col_sum, 1.0)
    row_sum = weights.sum(axis=1, keepdims=True)
    weights /= np.maximum(row_sum, 1.0e-12)

    reweighted: list[tuple[float, int, int]] = []
    touched = 0
    for score, c, p in edges:
        row = c_pos.get(c)
        col = p_pos.get(p)
        if row is None or col is None:
            reweighted.append((score, c, p))
            continue
        prob = float(weights[row, col])
        reweighted.append((score + SINKHORN_WEIGHT * prob, c, p))
        touched += 1
    print(
        f"sinkhorn_reweight_c={len(c_nodes)} p={len(p_nodes)} edges={touched} "
        f"prob_max={float(weights.max()):.6f}",
        flush=True,
    )
    return reweighted


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


def dense_cross_bigram_counts(
    ids: np.ndarray,
    left_nodes: np.ndarray,
    right_nodes: np.ndarray,
    vocab_floor: int,
    offset: int = 1,
) -> np.ndarray:
    left_lookup = np.full(vocab_floor, -1, dtype=np.int32)
    right_lookup = np.full(vocab_floor, -1, dtype=np.int32)
    left_valid = left_nodes < vocab_floor
    right_valid = right_nodes < vocab_floor
    left_lookup[left_nodes[left_valid]] = np.arange(len(left_nodes), dtype=np.int32)[left_valid]
    right_lookup[right_nodes[right_valid]] = np.arange(len(right_nodes), dtype=np.int32)[right_valid]
    prev = left_lookup[ids[:-offset]]
    nxt = right_lookup[ids[offset:]]
    mask = (prev >= 0) & (nxt >= 0)
    rows = len(left_nodes)
    cols = len(right_nodes)
    if not bool(mask.any()):
        return np.zeros((rows, cols), dtype=np.float32)
    flat = prev[mask].astype(np.int64, copy=False) * cols + nxt[mask].astype(np.int64, copy=False)
    return np.bincount(flat, minlength=rows * cols).reshape(rows, cols).astype(np.float32)


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


def bigram_swap_directional_deltas(
    c_big: np.ndarray, p_log: np.ndarray, perm: np.ndarray, a: int, b: int
) -> tuple[float, float]:
    pa = int(perm[a])
    pb = int(perm[b])
    old_out = float(c_big[a, :] @ p_log[pa, perm]) + float(c_big[b, :] @ p_log[pb, perm])
    old_in = float(c_big[:, a] @ p_log[perm, pa]) + float(c_big[:, b] @ p_log[perm, pb])
    new_perm = perm.copy()
    new_perm[a], new_perm[b] = new_perm[b], new_perm[a]
    new_out = float(c_big[a, :] @ p_log[pb, new_perm]) + float(c_big[b, :] @ p_log[pa, new_perm])
    new_in = float(c_big[:, a] @ p_log[new_perm, pb]) + float(c_big[:, b] @ p_log[new_perm, pa])
    return new_out - old_out, new_in - old_in


def log_probs_from_bigram_counts(p_big: np.ndarray, alpha: float = BIGRAM_REFINE_ALPHA) -> np.ndarray:
    k = p_big.shape[0]
    return np.log((p_big + alpha) / (p_big.sum(axis=1, keepdims=True) + alpha * k)).astype(np.float32)


def lcb_extra_bigram_pass(
    cipher_ids: np.ndarray,
    ref_ids: np.ndarray,
    c_nodes: np.ndarray,
    p_nodes: np.ndarray,
    proposals: list[tuple[int, int]],
    perm: np.ndarray,
    owner: np.ndarray,
    c_big: np.ndarray,
    p_log: np.ndarray,
    vocab_floor: int,
    target_vocab_size: int,
) -> tuple[int, list[float]]:
    if not LCB_EXTRA_BIGRAM_PASS or len(cipher_ids) > LCB_EXTRA_BIGRAM_MAX_TOKENS:
        return 0, []

    c_skip = dense_bigram_counts(cipher_ids, c_nodes, vocab_floor, offset=2)
    p_skip_log = log_probs_from_bigram_counts(dense_bigram_counts(ref_ids, p_nodes, target_vocab_size, offset=2))
    shard_bigs: list[np.ndarray] = []
    shard_count = max(1, LCB_EXTRA_BIGRAM_SHARDS)
    shard_len = max(2, len(cipher_ids) // shard_count)
    for shard_idx in range(shard_count):
        start = shard_idx * shard_len
        stop = len(cipher_ids) if shard_idx == shard_count - 1 else min(len(cipher_ids), (shard_idx + 1) * shard_len)
        if stop - start < 4:
            continue
        shard_bigs.append(dense_bigram_counts(cipher_ids[start:stop], c_nodes, vocab_floor))
    if not shard_bigs:
        return 0, []

    accepted = 0
    accepted_lcbs: list[float] = []
    checked = 0
    full_positive = 0
    directional_positive = 0
    shard_positive = 0
    for i, p_idx in proposals:
        j = int(owner[p_idx])
        if i == j:
            continue
        checked += 1
        full_delta = bigram_swap_delta(c_big, p_log, perm, i, j)
        if full_delta <= LCB_EXTRA_BIGRAM_MIN_DELTA:
            continue
        full_positive += 1
        out_delta, in_delta = bigram_swap_directional_deltas(c_big, p_log, perm, i, j)
        if out_delta <= 0.0 or in_delta <= 0.0:
            continue
        directional_positive += 1
        skip_delta = bigram_swap_delta(c_skip, p_skip_log, perm, i, j)
        if skip_delta < LCB_EXTRA_BIGRAM_SKIP_FLOOR * full_delta:
            continue
        shard_deltas = np.asarray([bigram_swap_delta(shard, p_log, perm, i, j) for shard in shard_bigs], dtype=np.float32)
        positive_shards = int(np.count_nonzero(shard_deltas > 0.0))
        if positive_shards < min(LCB_EXTRA_BIGRAM_MIN_SHARD_POSITIVE, len(shard_deltas)):
            continue
        median = float(np.median(shard_deltas))
        mad = float(np.median(np.abs(shard_deltas - median)))
        lcb = median - LCB_EXTRA_BIGRAM_LAMBDA * mad
        if lcb <= 0.0:
            continue
        shard_positive += 1
        accepted_lcbs.append(lcb)
        pi = int(perm[i])
        pj = int(perm[j])
        perm[i], perm[j] = perm[j], perm[i]
        owner[pi], owner[pj] = owner[pj], owner[pi]
        accepted += 1

    print(
        f"lcb_extra_bigram_checked={checked} full_positive={full_positive} "
        f"directional_positive={directional_positive} shard_positive={shard_positive}",
        flush=True,
    )
    print(f"lcb_extra_bigram_swaps={accepted}", flush=True)
    if accepted_lcbs:
        arr = np.asarray(accepted_lcbs, dtype=np.float32)
        print(f"lcb_extra_bigram_lcb_median={float(np.median(arr)):.6f}", flush=True)
        print(f"lcb_extra_bigram_lcb_p10={float(np.percentile(arr, 10)):.6f}", flush=True)
        print(f"lcb_extra_bigram_lcb_p90={float(np.percentile(arr, 90)):.6f}", flush=True)
    return accepted, accepted_lcbs


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
    candidate_c_edges = 0
    missing_p_edges = 0
    duplicate_proposals = 0
    for _, c, p in edges:
        i = c_to_i.get(c)
        p_idx = p_to_i.get(p)
        if i is None:
            continue
        candidate_c_edges += 1
        if p_idx is None:
            missing_p_edges += 1
            continue
        key = (i, p_idx)
        if key in seen_proposals:
            duplicate_proposals += 1
            continue
        seen_proposals.add(key)
        proposals.append(key)
        if len(proposals) >= BIGRAM_REFINE_MAX_PROPOSALS:
            break

    swaps = 0
    passes_run = 0
    accepted_deltas: list[float] = []
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
            accepted_deltas.append(float(delta))
            pi = int(perm[i])
            pj = int(perm[j])
            perm[i], perm[j] = perm[j], perm[i]
            owner[pi], owner[pj] = owner[pj], owner[pi]
            pass_swaps += 1
        swaps += pass_swaps
        if pass_swaps == 0:
            break

    extra_swaps = 0
    extra_lcbs: list[float] = []
    if passes_run == pass_budget:
        extra_swaps, extra_lcbs = lcb_extra_bigram_pass(
            cipher_ids,
            ref_ids,
            c_nodes,
            p_nodes,
            proposals,
            perm,
            owner,
            c_big,
            p_log,
            len(mapping),
            target_vocab_size,
        )
        swaps += extra_swaps

    if swaps:
        refined = mapping.copy()
        refined[c_nodes] = p_nodes[perm]
        mapping = refined
    print(f"bigram_refine_proposals={len(proposals)}", flush=True)
    print(f"bigram_refine_candidate_c_edges={candidate_c_edges}", flush=True)
    print(f"bigram_refine_missing_p_edges={missing_p_edges}", flush=True)
    print(f"bigram_refine_duplicate_proposals={duplicate_proposals}", flush=True)
    print(f"bigram_refine_pass_budget={pass_budget}", flush=True)
    print(f"bigram_refine_passes={passes_run}", flush=True)
    print(f"bigram_refine_swaps={swaps}", flush=True)
    print(f"bigram_refine_lcb_extra_swaps={extra_swaps}", flush=True)
    if accepted_deltas:
        delta_arr = np.asarray(accepted_deltas, dtype=np.float32)
        print(f"bigram_refine_delta_median={float(np.median(delta_arr)):.6f}", flush=True)
        print(f"bigram_refine_delta_p10={float(np.percentile(delta_arr, 10)):.6f}", flush=True)
        print(f"bigram_refine_delta_p90={float(np.percentile(delta_arr, 90)):.6f}", flush=True)
    return mapping


def tail_unary_repair(
    cipher_ids: np.ndarray,
    ref_ids: np.ndarray,
    mapping: np.ndarray,
    c_counts: np.ndarray,
    p_counts: np.ndarray,
    c_focus: np.ndarray,
    edges: list[tuple[float, int, int]],
    target_vocab_size: int,
) -> np.ndarray:
    if len(cipher_ids) > TAIL_REPAIR_MAX_TOKENS or TAIL_REPAIR_NODES <= 0:
        return mapping
    start = min(BIGRAM_REFINE_TOKENS, len(c_focus))
    tail_nodes = c_focus[start : min(len(c_focus), start + TAIL_REPAIR_NODES)]
    if len(tail_nodes) < 64:
        return mapping

    context_c: list[int] = []
    context_p: list[int] = []
    seen_p: set[int] = set()
    for c in c_focus[: max(TAIL_REPAIR_CONTEXTS * 2, TAIL_REPAIR_CONTEXTS)]:
        c_int = int(c)
        p_int = int(mapping[c_int])
        if p_int < 0 or p_int >= target_vocab_size or p_int in seen_p:
            continue
        context_c.append(c_int)
        context_p.append(p_int)
        seen_p.add(p_int)
        if len(context_c) >= TAIL_REPAIR_CONTEXTS:
            break
    if len(context_c) < 64:
        return mapping

    tail_set = set(map(int, tail_nodes))
    candidates_by_c: dict[int, list[int]] = {int(c): [int(mapping[int(c)])] for c in tail_nodes}
    for _, c, p in edges:
        if c not in tail_set:
            continue
        cand = candidates_by_c[int(c)]
        p_int = int(p)
        if len(cand) >= TAIL_REPAIR_CANDIDATES:
            continue
        if p_int < 0 or p_int >= target_vocab_size or p_int in cand:
            continue
        cand.append(p_int)

    candidate_p: list[int] = []
    candidate_pos: dict[int, int] = {}
    for cand in candidates_by_c.values():
        for p in cand:
            if p not in candidate_pos:
                candidate_pos[p] = len(candidate_p)
                candidate_p.append(p)
    if len(candidate_p) < 64:
        return mapping

    tail_arr = np.asarray(list(candidates_by_c.keys()), dtype=np.int64)
    context_c_arr = np.asarray(context_c, dtype=np.int64)
    context_p_arr = np.asarray(context_p, dtype=np.int64)
    candidate_p_arr = np.asarray(candidate_p, dtype=np.int64)
    c_right = dense_cross_bigram_counts(cipher_ids, tail_arr, context_c_arr, len(mapping))
    c_left = dense_cross_bigram_counts(cipher_ids, context_c_arr, tail_arr, len(mapping)).T
    p_right = dense_cross_bigram_counts(ref_ids, candidate_p_arr, context_p_arr, target_vocab_size)
    p_left = dense_cross_bigram_counts(ref_ids, context_p_arr, candidate_p_arr, target_vocab_size)
    p_right_log = np.log(
        (p_right + BIGRAM_REFINE_ALPHA)
        / (p_right.sum(axis=1, keepdims=True) + BIGRAM_REFINE_ALPHA * len(context_p_arr))
    ).astype(np.float32)
    p_left_log = np.log(
        (p_left + BIGRAM_REFINE_ALPHA)
        / (p_left.sum(axis=1, keepdims=True) + BIGRAM_REFINE_ALPHA * len(candidate_p_arr))
    ).astype(np.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        scores = torch.as_tensor(c_right, dtype=torch.float32, device=device) @ torch.as_tensor(
            p_right_log, dtype=torch.float32, device=device
        ).T
        scores.add_(
            torch.as_tensor(c_left, dtype=torch.float32, device=device)
            @ torch.as_tensor(p_left_log, dtype=torch.float32, device=device)
        )
        score_np = scores.cpu().numpy()

    repaired = mapping.copy()
    accepted = 0
    accepted_gain_per_occ: list[float] = []
    for row, c in enumerate(tail_arr):
        cand = candidates_by_c[int(c)]
        if len(cand) <= 1:
            continue
        cand_idx = [candidate_pos[p] for p in cand]
        cand_scores = score_np[row, cand_idx]
        current_p = int(mapping[int(c)])
        current_local = cand.index(current_p) if current_p in cand else 0
        best_local = int(np.argmax(cand_scores))
        gain = float(cand_scores[best_local] - cand_scores[current_local])
        occ = max(1.0, float(c_counts[int(c)]))
        if best_local == current_local or gain / occ < TAIL_REPAIR_MIN_GAIN_PER_OCC:
            continue
        repaired[int(c)] = cand[best_local]
        accepted += 1
        accepted_gain_per_occ.append(gain / occ)
    print(f"tail_repair_nodes={len(tail_arr)} candidates={len(candidate_p_arr)} accepted={accepted}", flush=True)
    if accepted_gain_per_occ:
        gain_arr = np.asarray(accepted_gain_per_occ, dtype=np.float32)
        print(f"tail_repair_gain_per_occ_median={float(np.median(gain_arr)):.6f}", flush=True)
        print(f"tail_repair_gain_per_occ_p10={float(np.percentile(gain_arr, 10)):.6f}", flush=True)
        print(f"tail_repair_gain_per_occ_p90={float(np.percentile(gain_arr, 90)):.6f}", flush=True)
    return repaired


def external_owner_repair(
    cipher_ids: np.ndarray,
    ref_ids: np.ndarray,
    mapping: np.ndarray,
    c_counts: np.ndarray,
    c_focus: np.ndarray,
    edges: list[tuple[float, int, int]],
    target_vocab_size: int,
) -> np.ndarray:
    if not EXTERNAL_OWNER_REPAIR or len(cipher_ids) > EXTERNAL_OWNER_MAX_TOKENS:
        return mapping
    c_nodes = c_focus[: min(len(c_focus), EXTERNAL_OWNER_NODES)]
    if len(c_nodes) < 64:
        return mapping

    p_nodes = set(map(int, mapping[c_nodes]))
    owner_of_p: dict[int, int] = {}
    for c in c_focus:
        c_int = int(c)
        p_int = int(mapping[c_int])
        if p_int < 0 or p_int >= target_vocab_size or p_int in owner_of_p:
            continue
        owner_of_p[p_int] = c_int

    c_node_set = set(map(int, c_nodes))
    candidates: list[tuple[int, int, int]] = []
    candidates_seen: set[tuple[int, int]] = set()
    per_c_counts: dict[int, int] = {}
    for _, c, p in edges:
        if c not in c_node_set or p in p_nodes:
            continue
        if c_counts[int(c)] < EXTERNAL_OWNER_MIN_COUNT:
            continue
        owner = owner_of_p.get(int(p))
        if owner is None or owner == c or c_counts[owner] < EXTERNAL_OWNER_MIN_COUNT:
            continue
        used = per_c_counts.get(int(c), 0)
        if used >= EXTERNAL_OWNER_CANDIDATES:
            continue
        key = (int(c), int(p))
        if key in candidates_seen:
            continue
        candidates_seen.add(key)
        per_c_counts[int(c)] = used + 1
        candidates.append((int(c), int(p), int(owner)))
    if not candidates:
        return mapping

    context_c: list[int] = []
    context_p: list[int] = []
    seen_p: set[int] = set()
    for c in c_focus[: max(EXTERNAL_OWNER_CONTEXTS * 2, EXTERNAL_OWNER_CONTEXTS)]:
        c_int = int(c)
        p_int = int(mapping[c_int])
        if p_int < 0 or p_int >= target_vocab_size or p_int in seen_p:
            continue
        context_c.append(c_int)
        context_p.append(p_int)
        seen_p.add(p_int)
        if len(context_c) >= EXTERNAL_OWNER_CONTEXTS:
            break
    if len(context_c) < 64:
        return mapping

    repair_nodes: list[int] = []
    repair_seen: set[int] = set()
    candidate_p: list[int] = []
    candidate_p_seen: set[int] = set()
    for c, p, owner in candidates:
        for node in (c, owner):
            if node not in repair_seen:
                repair_seen.add(node)
                repair_nodes.append(node)
        for p_int in (int(mapping[c]), int(mapping[owner]), p):
            if 0 <= p_int < target_vocab_size and p_int not in candidate_p_seen:
                candidate_p_seen.add(p_int)
                candidate_p.append(p_int)
    if len(repair_nodes) < 64 or len(candidate_p) < 64:
        return mapping

    repair_arr = np.asarray(repair_nodes, dtype=np.int64)
    context_c_arr = np.asarray(context_c, dtype=np.int64)
    context_p_arr = np.asarray(context_p, dtype=np.int64)
    candidate_p_arr = np.asarray(candidate_p, dtype=np.int64)
    c_right = dense_cross_bigram_counts(cipher_ids, repair_arr, context_c_arr, len(mapping))
    c_left = dense_cross_bigram_counts(cipher_ids, context_c_arr, repair_arr, len(mapping)).T
    p_right = dense_cross_bigram_counts(ref_ids, candidate_p_arr, context_p_arr, target_vocab_size)
    p_left = dense_cross_bigram_counts(ref_ids, context_p_arr, candidate_p_arr, target_vocab_size)
    p_right_log = np.log(
        (p_right + BIGRAM_REFINE_ALPHA)
        / (p_right.sum(axis=1, keepdims=True) + BIGRAM_REFINE_ALPHA * len(context_p_arr))
    ).astype(np.float32)
    p_left_log = np.log(
        (p_left + BIGRAM_REFINE_ALPHA)
        / (p_left.sum(axis=1, keepdims=True) + BIGRAM_REFINE_ALPHA * len(candidate_p_arr))
    ).astype(np.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        scores = torch.as_tensor(c_right, dtype=torch.float32, device=device) @ torch.as_tensor(
            p_right_log, dtype=torch.float32, device=device
        ).T
        scores.add_(
            torch.as_tensor(c_left, dtype=torch.float32, device=device)
            @ torch.as_tensor(p_left_log, dtype=torch.float32, device=device)
        )
        score_np = scores.cpu().numpy()

    node_pos = {int(c): i for i, c in enumerate(repair_nodes)}
    p_pos = {int(p): i for i, p in enumerate(candidate_p)}
    proposals: list[tuple[float, int, int, int, int]] = []
    for c, p, owner in candidates:
        old_c = int(mapping[c])
        old_owner = int(mapping[owner])
        if old_owner != p:
            continue
        ci = node_pos.get(c)
        oi = node_pos.get(owner)
        pi = p_pos.get(p)
        old_ci = p_pos.get(old_c)
        if ci is None or oi is None or pi is None or old_ci is None:
            continue
        old_score = float(score_np[ci, old_ci] + score_np[oi, pi])
        new_score = float(score_np[ci, pi] + score_np[oi, old_ci])
        occ = max(1.0, float(c_counts[c] + c_counts[owner]))
        gain_per_occ = (new_score - old_score) / occ
        if gain_per_occ < EXTERNAL_OWNER_MIN_GAIN_PER_OCC:
            continue
        proposals.append((gain_per_occ, c, p, owner, old_c))

    proposals.sort(reverse=True)
    repaired = mapping.copy()
    used_nodes: set[int] = set()
    used_targets: set[int] = set()
    accepted = 0
    for gain, c, p, owner, old_c in proposals:
        if c in used_nodes or owner in used_nodes or p in used_targets or old_c in used_targets:
            continue
        if int(repaired[c]) != old_c or int(repaired[owner]) != p:
            continue
        repaired[c] = p
        repaired[owner] = old_c
        used_nodes.update((c, owner))
        used_targets.update((p, old_c))
        accepted += 1
    print(f"external_owner_candidates={len(candidates)} proposals={len(proposals)} accepted={accepted}", flush=True)
    if proposals:
        gains = np.asarray([p[0] for p in proposals], dtype=np.float32)
        print(f"external_owner_gain_per_occ_median={float(np.median(gains)):.6f}", flush=True)
        print(f"external_owner_gain_per_occ_p90={float(np.percentile(gains, 90)):.6f}", flush=True)
    return repaired


def align_shuffled(
    cipher_ids: np.ndarray,
    ref_ids: np.ndarray,
    target_vocab_size: int,
) -> np.ndarray:
    global LAST_FINAL_EDGES, LAST_PRE_REPAIR_EDGES, LAST_POST_REPAIR_EDGES
    LAST_FINAL_EDGES = []
    LAST_PRE_REPAIR_EDGES = []
    LAST_POST_REPAIR_EDGES = []
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

    mapping = np.zeros(max(len(c_counts), target_vocab_size), dtype=np.int64)
    init = c_order_all[: len(p_order_all)]
    mapping[init] = p_order_all[: len(init)]

    c_log = np.log(np.maximum(c_counts, 1) / max(1, int(c_counts.sum())))
    p_log = np.log(np.maximum(p_counts, 1) / max(1, int(p_counts.sum())))
    p_rank = np.empty(target_vocab_size, dtype=np.int64)
    p_rank[p_order_all] = np.arange(target_vocab_size)
    c_focus_pos = {int(token_id): row for row, token_id in enumerate(c_focus)}
    anchor_rows = np.arange(min(ANCHORS, len(c_focus)), dtype=np.int64)

    for round_idx in range(rounds):
        print(f"round {round_idx + 1}/{rounds}: focus={len(c_focus)} anchors={len(anchor_rows)}", flush=True)
        edges = build_context_edges_for_mapping(
            cipher_ids,
            ref_ids,
            mapping,
            c_focus,
            p_focus,
            c_counts,
            p_counts,
            c_log,
            p_log,
            p_rank,
            anchor_rows,
            candidate_window,
            use_skip_context,
            device,
        )
        if round_idx == rounds - 1:
            LAST_PRE_REPAIR_EDGES = list(edges)
            LAST_FINAL_EDGES = list(edges)
        edges.sort(reverse=True)
        used_c: set[int] = set()
        used_p: set[int] = set()
        assigned_p_by_c: dict[int, int] = {}
        assigned_c_by_p: dict[int, int] = {}
        assigned_score_by_c: dict[int, float] = {}
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
            mapping = tail_unary_repair(
                cipher_ids,
                ref_ids,
                mapping,
                c_counts,
                p_counts,
                c_focus,
                edges,
                target_vocab_size,
            )
            mapping = external_owner_repair(
                cipher_ids,
                ref_ids,
                mapping,
                c_counts,
                c_focus,
                edges,
                target_vocab_size,
            )
            if POST_REPAIR_EDGE_DIAGNOSTIC or POST_REPAIR_REFRESH_REPAIR:
                refreshed_edges = build_context_edges_for_mapping(
                    cipher_ids,
                    ref_ids,
                    mapping,
                    c_focus,
                    p_focus,
                    c_counts,
                    p_counts,
                    c_log,
                    p_log,
                    p_rank,
                    anchor_rows,
                    candidate_window,
                    use_skip_context,
                    device,
                    label="post_repair_",
                )
                LAST_POST_REPAIR_EDGES = list(refreshed_edges)
                print(f"post_repair_edges={len(refreshed_edges)}", flush=True)
                if POST_REPAIR_REFRESH_REPAIR:
                    refreshed_edges.sort(reverse=True)
                    mapping = tail_unary_repair(
                        cipher_ids,
                        ref_ids,
                        mapping,
                        c_counts,
                        p_counts,
                        c_focus,
                        refreshed_edges,
                        target_vocab_size,
                    )
                    mapping = external_owner_repair(
                        cipher_ids,
                        ref_ids,
                        mapping,
                        c_counts,
                        c_focus,
                        refreshed_edges,
                        target_vocab_size,
                    )
                    LAST_FINAL_EDGES = list(refreshed_edges)
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


def error_breakdown(original: str, recovered: str, max_chars: int = 50_000) -> dict[str, int]:
    original = original[:max_chars]
    recovered = recovered[:max_chars]
    counts_by_class = {
        "whitespace": 0,
        "punctuation": 0,
        "alnum": 0,
        "replacement": 0,
        "other": 0,
    }

    def classify(ch: str) -> str:
        if ch == "\ufffd":
            return "replacement"
        if ch.isspace():
            return "whitespace"
        if ch.isalnum():
            return "alnum"
        if ch.isprintable():
            return "punctuation"
        return "other"

    for op in Levenshtein.editops(original, recovered):
        source_char = original[op.src_pos] if op.src_pos < len(original) else ""
        dest_char = recovered[op.dest_pos] if op.dest_pos < len(recovered) else ""
        label = classify(source_char or dest_char)
        if label == "alnum" and dest_char and classify(dest_char) != "alnum":
            label = classify(dest_char)
        counts_by_class[label] += 1
    return counts_by_class


def oracle_candidate_diagnostics(task, mapping: np.ndarray) -> dict[str, float]:
    edge_sets: list[tuple[str, list[tuple[float, int, int]]]] = []
    if LAST_PRE_REPAIR_EDGES:
        edge_sets.append(("pre", LAST_PRE_REPAIR_EDGES))
    elif LAST_FINAL_EDGES:
        edge_sets.append(("pre", LAST_FINAL_EDGES))
    if LAST_POST_REPAIR_EDGES:
        edge_sets.append(("post", LAST_POST_REPAIR_EDGES))
    if not ORACLE_CANDIDATE_DIAGNOSTICS or not edge_sets:
        return {}
    c_counts = counts(task.cipher_ids, int(max(len(mapping), int(task.cipher_ids.max(initial=0)) + 1)))
    focus = np.argsort(-c_counts)[:ORACLE_CANDIDATE_TOP_N]

    cipher_to_source: dict[int, int] = {}
    for cipher, source in zip(task.cipher_ids, task.secret_ids):
        c_int = int(cipher)
        if c_int not in cipher_to_source:
            cipher_to_source[c_int] = int(source)

    candidates_by_label: dict[str, dict[int, list[int]]] = {}
    for label, edges in edge_sets:
        candidates_by_c: dict[int, list[int]] = {}
        for score, c, p in sorted(edges, reverse=True):
            bucket = candidates_by_c.setdefault(int(c), [])
            if len(bucket) >= max(ORACLE_CANDIDATE_KS):
                continue
            if int(p) not in bucket:
                bucket.append(int(p))
        candidates_by_label[label] = candidates_by_c

    topk_weight_by_label = {
        label: {k: 0.0 for k in ORACLE_CANDIDATE_KS}
        for label, _ in edge_sets
    }
    pre_topk_weight = topk_weight_by_label[edge_sets[0][0]]
    post_topk_weight = topk_weight_by_label.get("post")
    total_weight = 0.0
    single_weight = 0.0
    mapped_weight = 0.0
    mean_emission_len_num = 0.0
    for c in focus:
        c_int = int(c)
        source_id = cipher_to_source.get(c_int)
        if source_id is None:
            continue
        weight = float(c_counts[c_int])
        if weight <= 0:
            continue
        total_weight += weight
        source_piece = task.source_adapter.decode([source_id])
        target_piece = task.target_adapter.encode(source_piece)
        mean_emission_len_num += weight * len(target_piece)
        if len(target_piece) != 1:
            continue
        single_weight += weight
        oracle_p = int(target_piece[0])
        if int(mapping[c_int]) == oracle_p:
            mapped_weight += weight
        for label, candidates_by_c in candidates_by_label.items():
            cand = candidates_by_c.get(c_int, [])
            for k in ORACLE_CANDIDATE_KS:
                if oracle_p in cand[:k]:
                    topk_weight_by_label[label][k] += weight

    if total_weight <= 0.0:
        return {}
    metrics: dict[str, float] = {
        "oracle_focus_mass": total_weight / max(1.0, float(len(task.cipher_ids))),
        "oracle_single_token_mass": single_weight / total_weight,
        "oracle_mean_target_emission_len": mean_emission_len_num / total_weight,
        "oracle_current_mapping_top1": mapped_weight / max(1.0, single_weight),
    }
    for k in ORACLE_CANDIDATE_KS:
        pre_value = pre_topk_weight[k] / max(1.0, single_weight)
        metrics[f"oracle_edge_top{k}"] = pre_value
        metrics[f"oracle_pre_edge_top{k}"] = pre_value
        if post_topk_weight is not None:
            post_value = post_topk_weight[k] / max(1.0, single_weight)
            metrics[f"oracle_post_edge_top{k}"] = post_value
            metrics[f"oracle_post_edge_top{k}_delta"] = post_value - pre_value
    print(f"oracle_candidate_diagnostics: {json.dumps(metrics, sort_keys=True)}", flush=True)
    return metrics


def token_piece_features(piece: str) -> list[float]:
    stripped = piece.strip()
    return [
        float(len(piece)),
        float(len(piece.encode("utf-8", errors="replace"))),
        float(piece.startswith(" ")),
        float(piece.isspace()),
        float("\n" in piece),
        float(bool(stripped) and stripped.isdigit()),
        float(any(ch.isalpha() for ch in piece)),
        float(any(ch.isalnum() for ch in piece)),
        float(bool(stripped) and all(not ch.isalnum() and not ch.isspace() for ch in stripped)),
        float(any((not ch.isalnum()) and (not ch.isspace()) for ch in piece)),
        float(piece[:1].isupper()),
        float(piece[:1].islower()),
    ]


def collect_oracle_rerank_groups(name: str, task, mapping: np.ndarray) -> list[dict]:
    c_counts = counts(task.cipher_ids, int(max(len(mapping), int(task.cipher_ids.max(initial=0)) + 1)))
    p_counts = counts(task.ref_ids, task.target_adapter.spec.vocab_size)
    c_focus = np.argsort(-c_counts)[:ORACLE_CANDIDATE_TOP_N].astype(np.int64)
    c_focus_set = set(map(int, c_focus))
    p_order = np.argsort(-p_counts)
    p_rank = np.empty(task.target_adapter.spec.vocab_size, dtype=np.int64)
    p_rank[p_order] = np.arange(task.target_adapter.spec.vocab_size)
    inv_perm = np.empty(len(task.perm), dtype=np.int64)
    inv_perm[task.perm.astype(np.int64)] = np.arange(len(task.perm), dtype=np.int64)

    candidates_by_c: dict[int, list[tuple[float, int]]] = {}
    seen_by_c: dict[int, set[int]] = {}
    for score, c, p in sorted(LAST_FINAL_EDGES, reverse=True):
        c_int = int(c)
        if c_int not in c_focus_set:
            continue
        bucket = candidates_by_c.setdefault(c_int, [])
        if len(bucket) >= ORACLE_RERANKER_TOP_K:
            continue
        seen = seen_by_c.setdefault(c_int, set())
        p_int = int(p)
        if p_int in seen:
            continue
        seen.add(p_int)
        bucket.append((float(score), p_int))

    owner_of_p: dict[int, int] = {}
    for c in c_focus:
        c_int = int(c)
        p_int = int(mapping[c_int])
        if 0 <= p_int < task.target_adapter.spec.vocab_size and p_int not in owner_of_p:
            owner_of_p[p_int] = c_int

    decode_cache: dict[int, str] = {}

    def target_piece(p: int) -> str:
        cached = decode_cache.get(p)
        if cached is None:
            cached = task.target_adapter.decode([p])
            decode_cache[p] = cached
        return cached

    groups: list[dict] = []
    total_single_mass = 0.0
    positive_mass = 0.0
    current_correct_mass = 0.0
    edge_top1_mass = 0.0
    for c in c_focus:
        c_int = int(c)
        cand = candidates_by_c.get(c_int)
        if not cand:
            continue
        source_id = int(inv_perm[c_int])
        source_piece = task.source_adapter.decode([source_id])
        oracle_ids = task.target_adapter.encode(source_piece)
        if len(oracle_ids) != 1:
            continue
        oracle_p = int(oracle_ids[0])
        weight = float(c_counts[c_int])
        total_single_mass += weight
        if int(mapping[c_int]) == oracle_p:
            current_correct_mass += weight
        if cand and cand[0][1] == oracle_p:
            edge_top1_mass += weight

        label_idx = -1
        best_score = cand[0][0]
        current_p = int(mapping[c_int])
        current_rank = int(p_rank[current_p]) if 0 <= current_p < len(p_rank) else len(p_rank)
        c_log = np.log(max(1.0, weight))
        feats: list[list[float]] = []
        candidate_ids: list[int] = []
        for rank, (score, p) in enumerate(cand):
            if p == oracle_p:
                label_idx = rank
            p_count = float(p_counts[p])
            p_log = np.log(max(1.0, p_count))
            rank_delta = abs(float(p_rank[p] - current_rank)) / max(1.0, float(len(p_rank)))
            owner = owner_of_p.get(p)
            piece = target_piece(p)
            feats.append(
                [
                    float(score),
                    float(score - best_score),
                    float(rank),
                    1.0 / float(rank + 1),
                    c_log,
                    p_log,
                    abs(c_log - p_log),
                    rank_delta,
                    float(p == current_p),
                    float(owner is not None and owner != c_int),
                    float(p_count / max(1.0, float(len(task.ref_ids)))),
                ]
                + token_piece_features(piece)
            )
            candidate_ids.append(p)
        if label_idx >= 0:
            positive_mass += weight
        groups.append(
            {
                "task": name,
                "cipher": c_int,
                "weight": weight,
                "features": np.asarray(feats, dtype=np.float32),
                "label": label_idx,
                "candidate_ids": candidate_ids,
                "oracle": oracle_p,
                "current_correct": int(current_p == oracle_p),
                "edge_top1_correct": int(cand[0][1] == oracle_p),
            }
        )
    print(
        f"oracle_rerank_collect task={name} groups={len(groups)} single_mass={total_single_mass:.0f} "
        f"top64_mass={positive_mass:.0f} current_top1={current_correct_mass / max(1.0, total_single_mass):.4f} "
        f"edge_top1={edge_top1_mass / max(1.0, total_single_mass):.4f}",
        flush=True,
    )
    return groups


def train_oracle_group_reranker(groups: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_groups = [g for g in groups if int(g["label"]) >= 0]
    if not train_groups:
        raise RuntimeError("no positive oracle reranker groups")
    dim = int(train_groups[0]["features"].shape[1])
    max_k = max(int(g["features"].shape[0]) for g in train_groups)
    x = np.zeros((len(train_groups), max_k, dim), dtype=np.float32)
    mask = np.zeros((len(train_groups), max_k), dtype=bool)
    labels = np.zeros(len(train_groups), dtype=np.int64)
    weights = np.zeros(len(train_groups), dtype=np.float32)
    for row, g in enumerate(train_groups):
        feat = g["features"]
        k = feat.shape[0]
        x[row, :k] = feat
        mask[row, :k] = True
        labels[row] = int(g["label"])
        weights[row] = min(float(np.sqrt(float(g["weight"]))), ORACLE_RERANKER_WEIGHT_CAP)
    flat = x[mask]
    mean = flat.mean(axis=0)
    std = flat.std(axis=0) + 1.0e-6
    x = (x - mean[None, None, :]) / std[None, None, :]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    xt = torch.as_tensor(x, dtype=torch.float32, device=device)
    mt = torch.as_tensor(mask, dtype=torch.bool, device=device)
    yt = torch.as_tensor(labels, dtype=torch.long, device=device)
    wt = torch.as_tensor(weights / max(1.0e-6, float(weights.mean())), dtype=torch.float32, device=device)
    coef = torch.zeros(dim, dtype=torch.float32, device=device, requires_grad=True)
    opt = torch.optim.Adam([coef], lr=ORACLE_RERANKER_LR)
    for _ in range(ORACLE_RERANKER_STEPS):
        opt.zero_grad(set_to_none=True)
        logits = xt @ coef
        logits = logits.masked_fill(~mt, -1.0e9)
        loss_by_group = torch.nn.functional.cross_entropy(logits, yt, reduction="none")
        loss = (loss_by_group * wt).mean() + ORACLE_RERANKER_L2 * torch.sum(coef * coef)
        loss.backward()
        opt.step()
    return coef.detach().cpu().numpy(), mean, std


def evaluate_oracle_group_reranker(groups: list[dict], coef: np.ndarray, mean: np.ndarray, std: np.ndarray) -> dict[str, float]:
    total = 0.0
    covered = 0.0
    current = 0.0
    edge = 0.0
    reranked = 0.0
    changed = 0.0
    for g in groups:
        weight = float(g["weight"])
        total += weight
        label = int(g["label"])
        current += weight * int(g["current_correct"])
        edge += weight * int(g["edge_top1_correct"])
        if label < 0:
            continue
        covered += weight
        feat = (g["features"] - mean[None, :]) / std[None, :]
        scores = feat @ coef
        pred = int(np.argmax(scores))
        reranked += weight * int(pred == label)
        changed += weight * int(pred != 0)
    return {
        "groups": float(len(groups)),
        "mass": total,
        "coverage": covered / max(1.0, total),
        "current_top1": current / max(1.0, total),
        "edge_top1": edge / max(1.0, total),
        "rerank_top1": reranked / max(1.0, total),
        "rerank_top1_conditional": reranked / max(1.0, covered),
        "changed_mass": changed / max(1.0, total),
    }


def oracle_reranker_diagnostic_main() -> None:
    pairs = [
        ("qwen", "qwen3_6_27b", "openai_o200k"),
        ("mistral", "mistral_medium_3_5", "openai_o200k"),
        ("gemma", "gemma4_31b", "openai_o200k"),
    ]
    collected: dict[str, list[dict]] = {}
    for name, source, target in pairs:
        print(f"oracle_reranker_task={name} source={source} target={target}", flush=True)
        task = load_task(source, target, target_tokens=100_000, reference_tokens=REFERENCE_TOKENS, seed=SEED)
        mapping = align_shuffled(task.cipher_ids, task.ref_ids, task.target_adapter.spec.vocab_size)
        collected[name] = collect_oracle_rerank_groups(name, task, mapping)

    folds: dict[str, dict[str, float]] = {}
    for test_name, _, _ in pairs:
        train_groups = [g for name, groups in collected.items() if name != test_name for g in groups]
        test_groups = collected[test_name]
        coef, mean, std = train_oracle_group_reranker(train_groups)
        metrics = evaluate_oracle_group_reranker(test_groups, coef, mean, std)
        folds[test_name] = metrics
        print(f"oracle_reranker_fold={test_name} {json.dumps(metrics, sort_keys=True)}", flush=True)

    total_mass = sum(m["mass"] for m in folds.values())
    aggregate = {
        key: sum(m[key] * m["mass"] for m in folds.values()) / max(1.0, total_mass)
        for key in ("coverage", "current_top1", "edge_top1", "rerank_top1", "changed_mass")
    }
    aggregate["mass"] = total_mass
    print(f"oracle_reranker_aggregate: {json.dumps(aggregate, sort_keys=True)}", flush=True)

    out_dir = CACHE_DIR / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "oracle_reranker_diagnostic.json").write_text(
        json.dumps({"folds": folds, "aggregate": aggregate}, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def synthetic_mode_probs(mode: str) -> tuple[float, float, float]:
    if mode.startswith("identity"):
        return 0.0, 0.0, 0.0
    if mode == "merge":
        return 0.28, 0.06, 0.06
    if mode == "split":
        return 0.08, 0.24, 0.10
    if mode == "shift":
        return 0.08, 0.10, 0.24
    return 0.14, 0.12, 0.12


def make_synthetic_cipher_task(
    base_ids: np.ndarray,
    target_adapter,
    mode: str,
    seed: int,
) -> tuple[np.ndarray, dict[int, list[int]], dict[int, str]]:
    rng = np.random.default_rng(seed)
    merge_p, split_p, shift_p = synthetic_mode_probs(mode)
    pieces_by_text: dict[str, int] = {}
    oracle_by_type: dict[int, list[int]] = {}
    text_by_type: dict[int, str] = {}
    type_sequence: list[int] = []

    def add_piece(piece: str) -> None:
        if piece == "":
            return
        token_type = pieces_by_text.get(piece)
        if token_type is None:
            token_type = len(pieces_by_text)
            pieces_by_text[piece] = token_type
            oracle_by_type[token_type] = target_adapter.encode(piece)
            text_by_type[token_type] = piece
        type_sequence.append(token_type)

    i = 0
    while i < len(base_ids):
        piece = target_adapter.decode([int(base_ids[i])])
        r = float(rng.random())
        if r < merge_p and i + 1 < len(base_ids):
            n = 3 if (mode == "merge" and rng.random() < 0.35 and i + 2 < len(base_ids)) else 2
            add_piece(target_adapter.decode([int(x) for x in base_ids[i : i + n]]))
            i += n
            continue
        if r < merge_p + split_p and len(piece) >= 4:
            split_min = 1
            split_max = len(piece) - 1
            if split_min < split_max:
                cut = int(rng.integers(split_min, split_max + 1))
                add_piece(piece[:cut])
                add_piece(piece[cut:])
                i += 1
                continue
        if r < merge_p + split_p + shift_p and piece.startswith(" ") and len(piece) > 1:
            if rng.random() < 0.5:
                add_piece(" ")
                add_piece(piece[1:])
            else:
                add_piece(piece[:2])
                add_piece(piece[2:])
            i += 1
            continue
        stripped = piece.strip()
        if r < merge_p + split_p + shift_p and stripped and all(
            (not ch.isalnum()) and (not ch.isspace()) for ch in stripped
        ) and len(piece) > 1:
            add_piece(piece[:-1])
            add_piece(piece[-1])
            i += 1
            continue
        add_piece(piece)
        i += 1

    type_count = len(pieces_by_text)
    if type_count == 0:
        return np.empty(0, dtype=np.uint32), {}, {}
    if type_count > target_adapter.spec.vocab_size:
        raise RuntimeError(f"synthetic type count {type_count} exceeds target vocab")
    perm = rng.permutation(target_adapter.spec.vocab_size)[:type_count].astype(np.int64)
    cipher_ids = perm[np.asarray(type_sequence, dtype=np.int64)].astype(np.uint32)
    oracle_by_cipher = {int(perm[token_type]): oracle for token_type, oracle in oracle_by_type.items()}
    text_by_cipher = {int(perm[token_type]): text for token_type, text in text_by_type.items()}
    return cipher_ids, oracle_by_cipher, text_by_cipher


def collect_synthetic_rerank_groups(
    name: str,
    cipher_ids: np.ndarray,
    oracle_by_cipher: dict[int, list[int]],
    ref_ids: np.ndarray,
    target_adapter,
    mapping: np.ndarray,
) -> list[dict]:
    c_counts = counts(cipher_ids, int(max(len(mapping), int(cipher_ids.max(initial=0)) + 1)))
    p_counts = counts(ref_ids, target_adapter.spec.vocab_size)
    c_focus = np.argsort(-c_counts)[:ORACLE_CANDIDATE_TOP_N].astype(np.int64)
    c_focus_set = set(map(int, c_focus))
    p_order = np.argsort(-p_counts)
    p_rank = np.empty(target_adapter.spec.vocab_size, dtype=np.int64)
    p_rank[p_order] = np.arange(target_adapter.spec.vocab_size)

    candidates_by_c: dict[int, list[tuple[float, int]]] = {}
    seen_by_c: dict[int, set[int]] = {}
    for score, c, p in sorted(LAST_FINAL_EDGES, reverse=True):
        c_int = int(c)
        if c_int not in c_focus_set:
            continue
        bucket = candidates_by_c.setdefault(c_int, [])
        if len(bucket) >= ORACLE_RERANKER_TOP_K:
            continue
        seen = seen_by_c.setdefault(c_int, set())
        p_int = int(p)
        if p_int in seen:
            continue
        seen.add(p_int)
        bucket.append((float(score), p_int))

    owner_of_p: dict[int, int] = {}
    for c in c_focus:
        c_int = int(c)
        p_int = int(mapping[c_int])
        if 0 <= p_int < target_adapter.spec.vocab_size and p_int not in owner_of_p:
            owner_of_p[p_int] = c_int

    decode_cache: dict[int, str] = {}

    def target_piece(p: int) -> str:
        cached = decode_cache.get(p)
        if cached is None:
            cached = target_adapter.decode([p])
            decode_cache[p] = cached
        return cached

    groups: list[dict] = []
    total_single_mass = 0.0
    positive_mass = 0.0
    current_correct_mass = 0.0
    edge_top1_mass = 0.0
    for c in c_focus:
        c_int = int(c)
        cand = candidates_by_c.get(c_int)
        oracle_ids = oracle_by_cipher.get(c_int)
        if not cand or oracle_ids is None or len(oracle_ids) != 1:
            continue
        oracle_p = int(oracle_ids[0])
        weight = float(c_counts[c_int])
        total_single_mass += weight
        if int(mapping[c_int]) == oracle_p:
            current_correct_mass += weight
        if cand and cand[0][1] == oracle_p:
            edge_top1_mass += weight

        label_idx = -1
        best_score = cand[0][0]
        current_p = int(mapping[c_int])
        current_rank = int(p_rank[current_p]) if 0 <= current_p < len(p_rank) else len(p_rank)
        c_log = np.log(max(1.0, weight))
        feats: list[list[float]] = []
        candidate_ids: list[int] = []
        for rank, (score, p) in enumerate(cand):
            if p == oracle_p:
                label_idx = rank
            p_count = float(p_counts[p])
            p_log = np.log(max(1.0, p_count))
            rank_delta = abs(float(p_rank[p] - current_rank)) / max(1.0, float(len(p_rank)))
            owner = owner_of_p.get(p)
            piece = target_piece(p)
            feats.append(
                [
                    float(score),
                    float(score - best_score),
                    float(rank),
                    1.0 / float(rank + 1),
                    c_log,
                    p_log,
                    abs(c_log - p_log),
                    rank_delta,
                    float(p == current_p),
                    float(owner is not None and owner != c_int),
                    float(p_count / max(1.0, float(len(ref_ids)))),
                ]
                + token_piece_features(piece)
            )
            candidate_ids.append(p)
        if label_idx >= 0:
            positive_mass += weight
        groups.append(
            {
                "task": name,
                "cipher": c_int,
                "weight": weight,
                "features": np.asarray(feats, dtype=np.float32),
                "label": label_idx,
                "candidate_ids": candidate_ids,
                "oracle": oracle_p,
                "current_correct": int(current_p == oracle_p),
                "edge_top1_correct": int(cand[0][1] == oracle_p),
            }
        )
    print(
        f"synthetic_rerank_collect task={name} groups={len(groups)} single_mass={total_single_mass:.0f} "
        f"top64_mass={positive_mass:.0f} current_top1={current_correct_mass / max(1.0, total_single_mass):.4f} "
        f"edge_top1={edge_top1_mass / max(1.0, total_single_mass):.4f}",
        flush=True,
    )
    return groups


def synthetic_reranker_diagnostic_main() -> None:
    task = load_task("qwen3_6_27b", TARGET_TOKENIZER, target_tokens=100_000, reference_tokens=REFERENCE_TOKENS, seed=SEED)
    target_adapter = task.target_adapter
    collected: dict[str, list[dict]] = {}
    for idx, mode in enumerate(SYNTHETIC_RERANKER_TASKS):
        base_start = idx * (SYNTHETIC_RERANKER_TOKENS + 50_000)
        ref_start = 1_000_000 + idx * (SYNTHETIC_RERANKER_REF_TOKENS + 50_000)
        base_ids = np.asarray(task.ref_ids[base_start : base_start + SYNTHETIC_RERANKER_TOKENS], dtype=np.uint32)
        ref_ids = np.asarray(task.ref_ids[ref_start : ref_start + SYNTHETIC_RERANKER_REF_TOKENS], dtype=np.uint32)
        if len(base_ids) < 10_000 or len(ref_ids) < 10_000:
            raise RuntimeError("not enough cached reference ids for synthetic reranker diagnostic")
        cipher_ids, oracle_by_cipher, _ = make_synthetic_cipher_task(
            base_ids,
            target_adapter,
            mode,
            seed=SEED + 1000 + idx,
        )
        print(
            f"synthetic_reranker_task={mode} cipher_tokens={len(cipher_ids)} "
            f"types={len(oracle_by_cipher)} ref_tokens={len(ref_ids)}",
            flush=True,
        )
        mapping = align_shuffled(cipher_ids, ref_ids, target_adapter.spec.vocab_size)
        collected[mode] = collect_synthetic_rerank_groups(mode, cipher_ids, oracle_by_cipher, ref_ids, target_adapter, mapping)

    folds: dict[str, dict[str, float]] = {}
    for test_name in SYNTHETIC_RERANKER_TASKS:
        train_groups = [g for name, groups in collected.items() if name != test_name for g in groups]
        test_groups = collected[test_name]
        coef, mean, std = train_oracle_group_reranker(train_groups)
        metrics = evaluate_oracle_group_reranker(test_groups, coef, mean, std)
        folds[test_name] = metrics
        print(f"synthetic_reranker_fold={test_name} {json.dumps(metrics, sort_keys=True)}", flush=True)

    total_mass = sum(m["mass"] for m in folds.values())
    aggregate = {
        key: sum(m[key] * m["mass"] for m in folds.values()) / max(1.0, total_mass)
        for key in ("coverage", "current_top1", "edge_top1", "rerank_top1", "changed_mass")
    }
    aggregate["mass"] = total_mass
    aggregate["rerank_minus_current"] = aggregate["rerank_top1"] - aggregate["current_top1"]
    aggregate["clears_threshold"] = float(aggregate["rerank_minus_current"] >= SYNTHETIC_RERANKER_MIN_TOP1_GAIN)
    print(f"synthetic_reranker_aggregate: {json.dumps(aggregate, sort_keys=True)}", flush=True)

    out_dir = CACHE_DIR / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "synthetic_reranker_diagnostic.json").write_text(
        json.dumps({"folds": folds, "aggregate": aggregate}, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def variable_emission_repair(
    cipher_ids: np.ndarray,
    ref_ids: np.ndarray,
    mapping: np.ndarray,
    target_vocab_size: int,
) -> dict[int, tuple[int, ...]]:
    if not VARIABLE_EMISSION_REPAIR or len(cipher_ids) > VARIABLE_EMISSION_MAX_TOKENS:
        return {}

    c_counts = counts(cipher_ids, int(max(len(mapping), int(cipher_ids.max(initial=0)) + 1)))
    p_counts = counts(ref_ids, target_vocab_size)
    c_focus = np.argsort(-c_counts)[: min(TOP_TOKENS, np.count_nonzero(c_counts))].astype(np.int64)
    p_order = np.argsort(-p_counts).astype(np.int64)
    repair_nodes = [int(c) for c in c_focus[:VARIABLE_EMISSION_NODES] if c_counts[int(c)] >= VARIABLE_EMISSION_MIN_COUNT]
    if len(repair_nodes) < 64:
        return {}

    context_c: list[int] = []
    context_p: list[int] = []
    seen_p: set[int] = set()
    for c in c_focus[: max(VARIABLE_EMISSION_CONTEXTS * 2, VARIABLE_EMISSION_CONTEXTS)]:
        c_int = int(c)
        p_int = int(mapping[c_int])
        if p_int < 0 or p_int >= target_vocab_size or p_int in seen_p:
            continue
        context_c.append(c_int)
        context_p.append(p_int)
        seen_p.add(p_int)
        if len(context_c) >= VARIABLE_EMISSION_CONTEXTS:
            break
    if len(context_c) < 64:
        return {}

    seed_p: list[int] = []
    seed_seen: set[int] = set()
    for c in repair_nodes:
        p_int = int(mapping[c])
        if 0 <= p_int < target_vocab_size and p_int not in seed_seen:
            seed_seen.add(p_int)
            seed_p.append(p_int)
    if len(seed_p) < 64:
        return {}

    pool = p_order[: min(VARIABLE_EMISSION_REF_POOL, len(p_order))]
    seed_arr = np.asarray(seed_p, dtype=np.int64)
    pool_arr = pool.astype(np.int64, copy=False)
    succ = dense_cross_bigram_counts(ref_ids, seed_arr, pool_arr, target_vocab_size)
    pred = dense_cross_bigram_counts(ref_ids, pool_arr, seed_arr, target_vocab_size).T
    pool_by_idx = [int(p) for p in pool_arr]
    pool_pos = {int(p): i for i, p in enumerate(pool_arr)}
    seed_pos = {p: i for i, p in enumerate(seed_p)}

    candidates_by_c: dict[int, list[tuple[int, int]]] = {}
    candidate_token_seen: set[int] = set()
    for c in repair_nodes:
        p_int = int(mapping[c])
        row = seed_pos.get(p_int)
        if row is None:
            continue
        pairs: list[tuple[int, int]] = []
        for counts_row, direction in ((succ[row], "right"), (pred[row], "left")):
            positive = np.flatnonzero(counts_row > 0)
            if positive.size == 0:
                continue
            take = min(VARIABLE_EMISSION_BIGRAM_CANDIDATES, int(positive.size))
            if positive.size > take:
                best_idx = positive[np.argpartition(counts_row[positive], -take)[-take:]]
            else:
                best_idx = positive
            best_idx = best_idx[np.argsort(-counts_row[best_idx])]
            for idx in best_idx:
                q_int = pool_by_idx[int(idx)]
                if q_int == p_int:
                    continue
                pair = (p_int, q_int) if direction == "right" else (q_int, p_int)
                if pair not in pairs:
                    pairs.append(pair)
                candidate_token_seen.update(pair)
        if pairs:
            candidates_by_c[c] = pairs
            candidate_token_seen.add(p_int)

    if not candidates_by_c or len(candidate_token_seen) < 64:
        return {}

    repair_arr = np.asarray(list(candidates_by_c.keys()), dtype=np.int64)
    context_c_arr = np.asarray(context_c, dtype=np.int64)
    context_p_arr = np.asarray(context_p, dtype=np.int64)
    candidate_arr = np.asarray(sorted(candidate_token_seen), dtype=np.int64)
    candidate_pos = {int(p): i for i, p in enumerate(candidate_arr)}

    c_right = dense_cross_bigram_counts(cipher_ids, repair_arr, context_c_arr, len(mapping))
    c_left = dense_cross_bigram_counts(cipher_ids, context_c_arr, repair_arr, len(mapping)).T
    p_context_to_candidate = dense_cross_bigram_counts(ref_ids, context_p_arr, candidate_arr, target_vocab_size)
    p_candidate_to_context = dense_cross_bigram_counts(ref_ids, candidate_arr, context_p_arr, target_vocab_size)
    p_candidate_bigram = (
        dense_bigram_counts(ref_ids, candidate_arr, target_vocab_size)
        if VARIABLE_EMISSION_FREE_LOCAL_PAIRS
        else None
    )

    total_ref = float(max(1, int(p_counts.sum())))
    alpha = BIGRAM_REFINE_ALPHA
    candidate_unigram = (p_counts[candidate_arr].astype(np.float32) + alpha) / (
        total_ref + alpha * target_vocab_size
    )
    context_unigram = (p_counts[context_p_arr].astype(np.float32) + alpha) / (
        total_ref + alpha * target_vocab_size
    )
    context_den = p_counts[context_p_arr].astype(np.float32)[:, None] + alpha
    candidate_den = p_counts[candidate_arr].astype(np.float32)[:, None] + alpha
    log_context_to_candidate = np.log(
        (p_context_to_candidate + alpha * candidate_unigram[None, :]) / context_den
    ).astype(np.float32)
    log_candidate_to_context = np.log(
        (p_candidate_to_context + alpha * context_unigram[None, :]) / candidate_den
    ).astype(np.float32)

    def internal_log(first: int, second: int) -> float:
        first_idx = candidate_pos.get(first)
        second_idx = candidate_pos.get(second)
        if p_candidate_bigram is not None and first_idx is not None and second_idx is not None:
            cnt = float(p_candidate_bigram[first_idx, second_idx])
        else:
            row = seed_pos.get(first)
            col = pool_pos.get(second)
            if row is not None and col is not None:
                cnt = float(succ[row, col])
            else:
                row2 = seed_pos.get(second)
                col2 = pool_pos.get(first)
                cnt = float(pred[row2, col2]) if row2 is not None and col2 is not None else 0.0
        uni = (float(p_counts[second]) + alpha) / (total_ref + alpha * target_vocab_size)
        return float(np.log((cnt + alpha * uni) / (float(p_counts[first]) + alpha)))

    emissions: dict[int, tuple[int, ...]] = {}
    proposals: list[tuple[float, int, tuple[int, int], float]] = []
    for row, c in enumerate(repair_arr):
        c_int = int(c)
        current = int(mapping[c_int])
        current_idx = candidate_pos.get(current)
        if current_idx is None:
            continue
        current_score = float(c_left[row] @ log_context_to_candidate[:, current_idx])
        current_score += float(c_right[row] @ log_candidate_to_context[current_idx, :])
        occ = max(1.0, float(c_counts[c_int]))
        best_pair: tuple[int, int] | None = None
        best_score = current_score
        local_pairs = candidates_by_c.get(c_int, [])
        if VARIABLE_EMISSION_FREE_LOCAL_PAIRS:
            local_tokens = sorted({tok for pair in local_pairs for tok in pair} | {current})
            expanded_pairs: list[tuple[int, int]] = []
            for first in local_tokens:
                first_idx = candidate_pos.get(first)
                if first_idx is None:
                    continue
                for second in local_tokens:
                    if first == second:
                        continue
                    second_idx = candidate_pos.get(second)
                    if second_idx is None:
                        continue
                    if p_candidate_bigram is not None and p_candidate_bigram[first_idx, second_idx] <= 0:
                        continue
                    expanded_pairs.append((first, second))
            local_pairs = expanded_pairs
        for first, second in local_pairs:
            first_idx = candidate_pos.get(first)
            second_idx = candidate_pos.get(second)
            if first_idx is None or second_idx is None:
                continue
            score = float(c_left[row] @ log_context_to_candidate[:, first_idx])
            score += occ * internal_log(first, second)
            score += float(c_right[row] @ log_candidate_to_context[second_idx, :])
            if score > best_score:
                best_score = score
                best_pair = (first, second)
        if best_pair is None:
            continue
        gain_per_occ = (best_score - current_score) / occ
        if gain_per_occ >= VARIABLE_EMISSION_MIN_GAIN_PER_OCC:
            proposals.append((gain_per_occ, c_int, best_pair, occ))

    proposals.sort(reverse=True)
    for gain, c, pair, _ in proposals[:VARIABLE_EMISSION_MAX_ACCEPTED]:
        emissions[c] = pair
    print(
        f"variable_emission_nodes={len(repair_arr)} candidates={sum(len(v) for v in candidates_by_c.values())} "
        f"proposals={len(proposals)} accepted={len(emissions)}",
        flush=True,
    )
    if proposals:
        gains = np.asarray([p[0] for p in proposals], dtype=np.float32)
        print(f"variable_emission_gain_per_occ_median={float(np.median(gains)):.6f}", flush=True)
        print(f"variable_emission_gain_per_occ_p90={float(np.percentile(gains, 90)):.6f}", flush=True)
    return emissions


def classify_piece(piece: str) -> str:
    if piece == "":
        return "empty"
    if "\ufffd" in piece:
        return "replacement"
    if piece in ("\n", "\r\n") or (piece and set(piece) <= {"\n", "\r"}):
        return "newline"
    if piece.isspace():
        return "whitespace"
    stripped = piece.strip()
    if stripped and all(not ch.isalnum() and not ch.isspace() for ch in stripped):
        return "punctuation"
    if stripped and stripped.isalnum():
        return "alnum"
    return "mixed"


def lm_total_bits(byte_lm, text: str) -> tuple[float, int]:
    data = text.encode("utf-8", errors="replace")
    if not data:
        return 0.0, 0
    return float(byte_lm.bits_per_byte(data) * len(data)), len(data)


def piece_for_cipher(
    cipher_id: int,
    mapping: np.ndarray,
    emissions: dict[int, tuple[int, ...]],
    target_adapter,
    cache: dict[int, str],
) -> str:
    cached = cache.get(cipher_id)
    if cached is not None:
        return cached
    emission = emissions.get(cipher_id)
    if emission is None:
        piece = target_adapter.decode([int(mapping[cipher_id])])
    else:
        piece = target_adapter.decode([int(p) for p in emission])
    cache[cipher_id] = piece
    return piece


def string_lexicon_repair(
    cipher_ids: np.ndarray,
    mapping: np.ndarray,
    emissions: dict[int, tuple[int, ...]],
    target_adapter,
    byte_lm,
) -> dict[int, str]:
    """Infer a tiny cipher-id -> string-piece override lexicon.

    This is intentionally conservative: it only tries split/boundary moves for
    frequent adjacent cipher-token pairs and scores the resulting global
    string-piece assignments on local byte-LM windows around all occurrences of
    the touched token types. The source tokenizer is not used.
    """

    if not STRING_LEXICON_REPAIR or len(cipher_ids) > STRING_LEXICON_MAX_TOKENS:
        return {}

    vocab_floor = int(max(len(mapping), int(cipher_ids.max(initial=0)) + 1))
    c_counts = counts(cipher_ids, vocab_floor)
    c_order = np.argsort(-c_counts)
    c_nodes = c_order[: min(STRING_LEXICON_NODES, np.count_nonzero(c_counts))].astype(np.int64)
    if len(c_nodes) < 64:
        return {}

    node_pos = {int(c): i for i, c in enumerate(c_nodes)}
    positions_by_c: dict[int, list[int]] = {int(c): [] for c in c_nodes}
    max_positions = max(STRING_LEXICON_MAX_CONTEXTS, 32)
    for pos, c in enumerate(cipher_ids):
        c_int = int(c)
        bucket = positions_by_c.get(c_int)
        if bucket is not None and len(bucket) < max_positions:
            bucket.append(pos)

    c_big = dense_bigram_counts(cipher_ids, c_nodes, vocab_floor)
    pair_rows, pair_cols = np.nonzero(c_big >= STRING_LEXICON_MIN_PAIR_COUNT)
    if len(pair_rows) == 0:
        return {}
    pair_counts = c_big[pair_rows, pair_cols]
    order = np.argsort(-pair_counts)[:STRING_LEXICON_BIGRAMS]
    piece_cache: dict[int, str] = {}
    proposals: list[tuple[float, int, int, str, str, str, str, int, float]] = []

    def render_window(start: int, stop: int, overrides: dict[int, str]) -> str:
        parts: list[str] = []
        for tok in cipher_ids[start:stop]:
            tok_int = int(tok)
            override = overrides.get(tok_int)
            if override is not None:
                parts.append(override)
            else:
                parts.append(piece_for_cipher(tok_int, mapping, emissions, target_adapter, piece_cache))
        return "".join(parts)

    def score_pair_overrides(a: int, b: int, cand_a: str, cand_b: str) -> tuple[float, int]:
        base_overrides: dict[int, str] = {}
        cand_overrides = {a: cand_a, b: cand_b}
        windows: set[tuple[int, int]] = set()
        for token in (a, b):
            for pos in positions_by_c.get(token, []):
                start = max(0, pos - STRING_LEXICON_CONTEXT_RADIUS)
                stop = min(len(cipher_ids), pos + STRING_LEXICON_CONTEXT_RADIUS + 1)
                windows.add((start, stop))
        old_bits = 0.0
        new_bits = 0.0
        total_bytes = 0
        for start, stop in windows:
            old_text = render_window(start, stop, base_overrides)
            new_text = render_window(start, stop, cand_overrides)
            old_chunk_bits, _ = lm_total_bits(byte_lm, old_text)
            new_chunk_bits, chunk_bytes = lm_total_bits(byte_lm, new_text)
            old_bits += old_chunk_bits
            new_bits += new_chunk_bits
            total_bytes += chunk_bytes
        return old_bits - new_bits, total_bytes

    for idx in order:
        row = int(pair_rows[int(idx)])
        col = int(pair_cols[int(idx)])
        if row == col:
            continue
        a = int(c_nodes[row])
        b = int(c_nodes[col])
        pair_count = int(c_big[row, col])
        min_count = max(1, min(int(c_counts[a]), int(c_counts[b])))
        coverage = pair_count / min_count
        if coverage < STRING_LEXICON_MIN_PAIR_COVERAGE:
            continue
        left_piece = piece_for_cipher(a, mapping, emissions, target_adapter, piece_cache)
        right_piece = piece_for_cipher(b, mapping, emissions, target_adapter, piece_cache)
        if "\ufffd" in left_piece or "\ufffd" in right_piece:
            continue
        combo = left_piece + right_piece
        if len(combo) > STRING_LEXICON_MAX_COMBO_CHARS or len(combo) < 2:
            continue
        boundary = len(left_piece)
        split_points: set[int] = set(range(max(1, boundary - STRING_LEXICON_MAX_SHIFT), min(len(combo), boundary + STRING_LEXICON_MAX_SHIFT) + 1))
        for split in range(1, len(combo)):
            prev_ch = combo[split - 1]
            next_ch = combo[split] if split < len(combo) else ""
            if prev_ch.isspace() or next_ch.isspace() or prev_ch in "'\".,:;!?)]}" or next_ch in "'\"([{" :
                split_points.add(split)
        if STRING_LEXICON_ALLOW_EMPTY_SPLITS and coverage >= 0.75:
            split_points.update((0, len(combo)))
        for split in sorted(split_points):
            if split == boundary:
                continue
            if not STRING_LEXICON_ALLOW_EMPTY_SPLITS and (split == 0 or split == len(combo)):
                continue
            cand_left = combo[:split]
            cand_right = combo[split:]
            if cand_left == left_piece and cand_right == right_piece:
                continue
            if STRING_LEXICON_FORMAT_ONLY:
                labels = (classify_piece(cand_left), classify_piece(cand_right))
                if any(label in {"alnum", "empty", "replacement"} for label in labels):
                    continue
            gain, scored_bytes = score_pair_overrides(a, b, cand_left, cand_right)
            if scored_bytes <= 0:
                continue
            gain_per_byte = gain / scored_bytes
            if gain_per_byte >= STRING_LEXICON_MIN_GAIN_PER_BYTE:
                proposals.append(
                    (
                        gain_per_byte,
                        a,
                        b,
                        cand_left,
                        cand_right,
                        left_piece,
                        right_piece,
                        pair_count,
                        coverage,
                    )
                )

    proposals.sort(reverse=True)
    overrides: dict[int, str] = {}
    used: set[int] = set()
    accepted = 0
    for gain, a, b, cand_left, cand_right, _, _, _, _ in proposals:
        if a in used or b in used:
            continue
        overrides[a] = cand_left
        overrides[b] = cand_right
        used.update((a, b))
        accepted += 1
        if accepted >= STRING_LEXICON_MAX_ACCEPTED:
            break

    print(
        f"string_lexicon_nodes={len(c_nodes)} pair_candidates={len(order)} proposals={len(proposals)} accepted_pairs={accepted}",
        flush=True,
    )
    if proposals:
        gains = np.asarray([p[0] for p in proposals], dtype=np.float32)
        print(f"string_lexicon_gain_per_byte_median={float(np.median(gains)):.6f}", flush=True)
        print(f"string_lexicon_gain_per_byte_p90={float(np.percentile(gains, 90)):.6f}", flush=True)
    if overrides:
        classes: dict[str, int] = {}
        empty = 0
        for piece in overrides.values():
            label = classify_piece(piece)
            classes[label] = classes.get(label, 0) + 1
            empty += int(piece == "")
        print(f"string_lexicon_classes={json.dumps(classes, sort_keys=True)} empty={empty}", flush=True)
    return overrides


def valid_piece_candidate(piece: str) -> bool:
    if piece == "" or len(piece) > STRING_CANDIDATE_MAX_CHARS:
        return False
    if "\ufffd" in piece or "\x00" in piece:
        return False
    return all(ch.isprintable() or ch in "\n\r\t" for ch in piece)


def candidate_class_compatible(base_label: str, cand_label: str) -> bool:
    if cand_label in {"empty", "replacement"}:
        return False
    if base_label in {"newline", "whitespace", "punctuation"}:
        return cand_label == base_label
    if base_label == "alnum":
        return cand_label in {"alnum", "mixed"}
    if base_label == "mixed":
        return cand_label in {"alnum", "mixed", "punctuation"}
    return cand_label == base_label


def build_string_candidate_inventory(
    ref_ids: np.ndarray,
    target_adapter,
    reference_text: Path,
) -> dict[str, list[str]]:
    p_counts = counts(ref_ids, target_adapter.spec.vocab_size)
    p_order = np.argsort(-p_counts)
    raw_counts: dict[str, int] = {}

    def add_piece(piece: str, weight: int = 1) -> None:
        if not valid_piece_candidate(piece):
            return
        raw_counts[piece] = raw_counts.get(piece, 0) + weight

    for p in p_order[:STRING_CANDIDATE_REF_TOKENS]:
        piece = target_adapter.decode([int(p)])
        add_piece(piece, int(p_counts[int(p)]))
        if 2 <= len(piece) <= STRING_CANDIDATE_MAX_CHARS:
            for width in range(2, min(12, len(piece)) + 1):
                add_piece(piece[:width])
                add_piece(piece[-width:])
            stripped = piece.strip()
            if stripped and stripped != piece:
                add_piece(stripped)
                add_piece(" " + stripped)

    raw = reference_text.read_bytes()[:2_000_000].decode("utf-8", errors="ignore")
    span_re = re.compile(r"\n+|\s+[A-Za-z]{1,20}|[A-Za-z]{2,20}|\s+\d{1,8}|\d{1,8}|\s*[.,;:!?()\[\]{}\"'`-]+")
    for match in span_re.finditer(raw):
        add_piece(match.group(0))

    ranked = sorted(raw_counts.items(), key=lambda kv: (-kv[1], len(kv[0]), kv[0]))
    inventory: dict[str, list[str]] = {}
    substring_budget = 0
    raw_budget = 0
    for piece, _ in ranked:
        label = classify_piece(piece)
        if piece in raw:
            raw_budget += 1
            if raw_budget > STRING_CANDIDATE_RAW_SPANS + STRING_CANDIDATE_SUBSTRINGS:
                continue
        else:
            substring_budget += 1
            if substring_budget > STRING_CANDIDATE_SUBSTRINGS:
                continue
        bucket = inventory.setdefault(label, [])
        if len(bucket) < STRING_CANDIDATE_RAW_SPANS:
            bucket.append(piece)
    return inventory


def string_candidate_repair(
    cipher_ids: np.ndarray,
    ref_ids: np.ndarray,
    mapping: np.ndarray,
    emissions: dict[int, tuple[int, ...]],
    target_adapter,
    byte_lm,
    reference_text: Path,
) -> dict[int, str]:
    if not STRING_CANDIDATE_REPAIR or len(cipher_ids) > STRING_CANDIDATE_MAX_TOKENS:
        return {}

    vocab_floor = int(max(len(mapping), int(cipher_ids.max(initial=0)) + 1))
    c_counts = counts(cipher_ids, vocab_floor)
    c_order = np.argsort(-c_counts)
    repair_nodes = [int(c) for c in c_order[:STRING_CANDIDATE_NODES] if c_counts[int(c)] > 0]
    if len(repair_nodes) < 32:
        return {}

    inventory = build_string_candidate_inventory(ref_ids, target_adapter, reference_text)
    piece_cache: dict[int, str] = {}
    positions_by_c: dict[int, list[int]] = {int(c): [] for c in repair_nodes}
    for pos, c in enumerate(cipher_ids):
        c_int = int(c)
        bucket = positions_by_c.get(c_int)
        if bucket is not None and len(bucket) < STRING_CANDIDATE_MAX_CONTEXTS:
            bucket.append(pos)

    def render_window(start: int, stop: int, override_c: int | None = None, override_piece: str | None = None) -> str:
        parts: list[str] = []
        for tok in cipher_ids[start:stop]:
            tok_int = int(tok)
            if override_c is not None and tok_int == override_c and override_piece is not None:
                parts.append(override_piece)
            else:
                parts.append(piece_for_cipher(tok_int, mapping, emissions, target_adapter, piece_cache))
        return "".join(parts)

    proposals: list[tuple[float, int, str, str, int]] = []
    candidate_count = 0
    for c in repair_nodes:
        base_piece = piece_for_cipher(c, mapping, emissions, target_adapter, piece_cache)
        base_label = classify_piece(base_piece)
        local_candidates: list[str] = []
        seen: set[str] = {base_piece}
        for label, pieces in inventory.items():
            if not candidate_class_compatible(base_label, label):
                continue
            if STRING_CANDIDATE_FORMAT_ONLY and label not in {"newline", "whitespace", "punctuation"}:
                continue
            for piece in pieces:
                if piece in seen:
                    continue
                if abs(len(piece) - len(base_piece)) > 8:
                    continue
                seen.add(piece)
                local_candidates.append(piece)
                if len(local_candidates) >= STRING_CANDIDATE_MAX_PER_TOKEN:
                    break
            if len(local_candidates) >= STRING_CANDIDATE_MAX_PER_TOKEN:
                break
        if not local_candidates:
            continue
        candidate_count += len(local_candidates)

        windows: set[tuple[int, int]] = set()
        for pos in positions_by_c.get(c, []):
            start = max(0, pos - STRING_CANDIDATE_CONTEXT_RADIUS)
            stop = min(len(cipher_ids), pos + STRING_CANDIDATE_CONTEXT_RADIUS + 1)
            windows.add((start, stop))
        if not windows:
            continue
        old_bits = 0.0
        old_bytes = 0
        rendered_old: dict[tuple[int, int], str] = {}
        for start, stop in windows:
            text = render_window(start, stop)
            rendered_old[(start, stop)] = text
            bits, nbytes = lm_total_bits(byte_lm, text)
            old_bits += bits
            old_bytes += nbytes
        if old_bytes <= 0:
            continue

        best_piece: str | None = None
        best_gain = 0.0
        for candidate in local_candidates:
            new_bits = 0.0
            new_bytes = 0
            for start, stop in windows:
                text = render_window(start, stop, c, candidate)
                bits, nbytes = lm_total_bits(byte_lm, text)
                new_bits += bits
                new_bytes += nbytes
            if new_bytes <= 0:
                continue
            # Normalize by the new byte count so deletions/insertions must still
            # improve the actual byte-level objective, not just shorten text.
            gain_per_byte = (old_bits - new_bits) / new_bytes
            if gain_per_byte > best_gain:
                best_gain = gain_per_byte
                best_piece = candidate
        if best_piece is not None and best_gain >= STRING_CANDIDATE_MIN_GAIN_PER_BYTE:
            proposals.append((best_gain, c, best_piece, base_piece, int(c_counts[c])))

    proposals.sort(reverse=True)
    overrides: dict[int, str] = {}
    for gain, c, piece, _, _ in proposals[:STRING_CANDIDATE_MAX_ACCEPTED]:
        overrides[c] = piece

    print(
        f"string_candidate_nodes={len(repair_nodes)} inventory={sum(len(v) for v in inventory.values())} "
        f"candidates={candidate_count} proposals={len(proposals)} accepted={len(overrides)}",
        flush=True,
    )
    if proposals:
        gains = np.asarray([p[0] for p in proposals], dtype=np.float32)
        print(f"string_candidate_gain_per_byte_median={float(np.median(gains)):.6f}", flush=True)
        print(f"string_candidate_gain_per_byte_p90={float(np.percentile(gains, 90)):.6f}", flush=True)
    if overrides:
        classes: dict[str, int] = {}
        duplicates = len(overrides) - len(set(overrides.values()))
        for piece in overrides.values():
            label = classify_piece(piece)
            classes[label] = classes.get(label, 0) + 1
        print(
            f"string_candidate_classes={json.dumps(classes, sort_keys=True)} duplicate_pieces={duplicates}",
            flush=True,
        )
    return overrides


def decode_with_variable_emissions(
    cipher_ids: np.ndarray,
    mapping: np.ndarray,
    emissions: dict[int, tuple[int, ...]],
    target_adapter,
) -> str:
    if not emissions:
        return target_adapter.decode(mapping[cipher_ids].astype(int).tolist())
    out: list[int] = []
    for c in cipher_ids:
        c_int = int(c)
        emission = emissions.get(c_int)
        if emission is None:
            out.append(int(mapping[c_int]))
        else:
            out.extend(int(p) for p in emission)
    return target_adapter.decode(out)


def decode_with_string_lexicon(
    cipher_ids: np.ndarray,
    mapping: np.ndarray,
    emissions: dict[int, tuple[int, ...]],
    string_overrides: dict[int, str],
    target_adapter,
) -> str:
    if not string_overrides:
        return decode_with_variable_emissions(cipher_ids, mapping, emissions, target_adapter)
    cache: dict[int, str] = {}
    parts: list[str] = []
    for c in cipher_ids:
        c_int = int(c)
        override = string_overrides.get(c_int)
        if override is not None:
            parts.append(override)
        else:
            parts.append(piece_for_cipher(c_int, mapping, emissions, target_adapter, cache))
    return "".join(parts)


def main() -> None:
    if ORACLE_RERANKER_DIAGNOSTIC:
        oracle_reranker_diagnostic_main()
        return
    if SYNTHETIC_RERANKER_DIAGNOSTIC:
        synthetic_reranker_diagnostic_main()
        return

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
    print(f"cipher_tokens: {len(task.cipher_ids):,}")
    print(f"reference_tokens: {len(task.ref_ids):,}")

    mapping = align_shuffled(task.cipher_ids, task.ref_ids, task.target_adapter.spec.vocab_size)
    emissions = variable_emission_repair(task.cipher_ids, task.ref_ids, mapping, task.target_adapter.spec.vocab_size)
    string_overrides = string_lexicon_repair(
        task.cipher_ids,
        mapping,
        emissions,
        task.target_adapter,
        task.byte_lm,
    )
    candidate_overrides = string_candidate_repair(
        task.cipher_ids,
        task.ref_ids,
        mapping,
        emissions,
        task.target_adapter,
        task.byte_lm,
        task.reference_text,
    )
    string_overrides.update(candidate_overrides)
    recovered_sample = decode_with_string_lexicon(
        task.cipher_ids[:SAMPLE_TOKENS],
        mapping,
        emissions,
        string_overrides,
        task.target_adapter,
    )
    metrics = evaluate_recovery(task, recovered_sample, SAMPLE_TOKENS)
    diagnostics: dict[str, int] = {}
    if ENABLE_DIAGNOSTICS:
        original_sample = task.source_adapter.decode(task.secret_ids[:SAMPLE_TOKENS].astype(int).tolist())
        diagnostics = error_breakdown(original_sample, recovered_sample)
        print(f"error_breakdown: {json.dumps(diagnostics, sort_keys=True)}", flush=True)
        oracle_metrics = oracle_candidate_diagnostics(task, mapping)
        diagnostics.update({k: int(v * 1_000_000) for k, v in oracle_metrics.items()})

    out_dir = CACHE_DIR / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "last_recovered.txt").write_text(recovered_sample, encoding="utf-8", errors="ignore")
    report = {
        "source_tokenizer": task.source_adapter.spec.name,
        "target_tokenizer": task.target_adapter.spec.name,
        "target_tokens": int(len(task.cipher_ids)),
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
        "variable_emission_repair": VARIABLE_EMISSION_REPAIR,
        "variable_emissions": len(emissions),
        "string_lexicon_repair": STRING_LEXICON_REPAIR,
        "string_lexicon_overrides": len(string_overrides),
        "string_candidate_repair": STRING_CANDIDATE_REPAIR,
        "string_candidate_overrides": len(candidate_overrides),
        "string_lexicon_classes": {
            label: sum(1 for piece in string_overrides.values() if classify_piece(piece) == label)
            for label in sorted({classify_piece(piece) for piece in string_overrides.values()})
        },
        "diagnostics": diagnostics,
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
    print(f"target_tokens_M:  {len(task.cipher_ids) / 1e6:.3f}")
    print(f"reference_tokens_M: {len(task.ref_ids) / 1e6:.3f}")
    print(f"top_tokens:       {TOP_TOKENS}")
    print(f"anchors:          {ANCHORS}")
    print(f"rounds:           {effective_rounds(len(task.cipher_ids))}")


if __name__ == "__main__":
    main()
