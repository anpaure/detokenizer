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
ORACLE_AUDIT = os.environ.get("DETOK_ORACLE_AUDIT", "0") == "1"
ORACLE_AUDIT_TYPES = int(os.environ.get("DETOK_ORACLE_AUDIT_TYPES", "8192"))
RETOK_OBJECTIVE_AUDIT = os.environ.get("DETOK_RETOK_OBJECTIVE_AUDIT", "0") == "1"
RETOK_MOVE_AUDIT_TYPES = int(os.environ.get("DETOK_RETOK_MOVE_AUDIT_TYPES", "2048"))
RETOK_MOVE_AUDIT_TOPK = int(os.environ.get("DETOK_RETOK_MOVE_AUDIT_TOPK", "16"))
RETOK_WINDOW_RADIUS = int(os.environ.get("DETOK_RETOK_WINDOW_RADIUS", "12"))
RETOK_MAX_OCCURRENCES = int(os.environ.get("DETOK_RETOK_MAX_OCCURRENCES", "256"))
RETOK_MOVE_AUDIT_MAX_WINDOWS = int(os.environ.get("DETOK_RETOK_MOVE_AUDIT_MAX_WINDOWS", "200000"))
RETOK_BATCH_AUDIT = os.environ.get("DETOK_RETOK_BATCH_AUDIT", "0") == "1"
RETOK_BATCH_TYPES = int(os.environ.get("DETOK_RETOK_BATCH_TYPES", "2048"))
RETOK_BATCH_TOPK = int(os.environ.get("DETOK_RETOK_BATCH_TOPK", "16"))
RETOK_BATCH_REPEATS = int(os.environ.get("DETOK_RETOK_BATCH_REPEATS", "8"))
RETOK_BATCH_SIZES = tuple(int(x) for x in os.environ.get("DETOK_RETOK_BATCH_SIZES", "8,16,32,64,128").split(",") if x)
RETOK_BATCH_GOOD_FRACTIONS = tuple(
    float(x) for x in os.environ.get("DETOK_RETOK_BATCH_GOOD_FRACTIONS", "0,0.1,0.25,0.5,0.75,1").split(",") if x
)
RETOK_CEM_SEARCH = os.environ.get("DETOK_RETOK_CEM_SEARCH", "0") == "1"
RETOK_CEM_TYPES = int(os.environ.get("DETOK_RETOK_CEM_TYPES", "1024"))
RETOK_CEM_TOPK = int(os.environ.get("DETOK_RETOK_CEM_TOPK", "16"))
RETOK_CEM_MOVE_POOL = int(os.environ.get("DETOK_RETOK_CEM_MOVE_POOL", "512"))
RETOK_CEM_PRIOR_MAX_WINDOWS = int(os.environ.get("DETOK_RETOK_CEM_PRIOR_MAX_WINDOWS", "250000"))
RETOK_CEM_PRIOR_MAX_OCC = int(os.environ.get("DETOK_RETOK_CEM_PRIOR_MAX_OCC", "64"))
RETOK_CEM_SAMPLES = int(os.environ.get("DETOK_RETOK_CEM_SAMPLES", "64"))
RETOK_CEM_ROUNDS = int(os.environ.get("DETOK_RETOK_CEM_ROUNDS", "8"))
RETOK_CEM_ELITE_FRAC = float(os.environ.get("DETOK_RETOK_CEM_ELITE_FRAC", "0.10"))
RETOK_CEM_BATCH_MOVES = int(os.environ.get("DETOK_RETOK_CEM_BATCH_MOVES", "32"))
RETOK_CEM_UPDATE_RATE = float(os.environ.get("DETOK_RETOK_CEM_UPDATE_RATE", "0.70"))
RETOK_CEM_ENTROPY_FLOOR = float(os.environ.get("DETOK_RETOK_CEM_ENTROPY_FLOOR", "0.02"))
RETOK_CEM_GRAPH_RANK_PENALTY = float(os.environ.get("DETOK_RETOK_CEM_GRAPH_RANK_PENALTY", "0.0005"))
RETOK_CEM_DUP_PENALTY = float(os.environ.get("DETOK_RETOK_CEM_DUP_PENALTY", "0.0"))
RETOK_CEM_LEN_PENALTY = float(os.environ.get("DETOK_RETOK_CEM_LEN_PENALTY", "0.01"))
RETOK_ELITE_FILTER = os.environ.get("DETOK_RETOK_ELITE_FILTER", "0") == "1"
RETOK_ELITE_GLOBAL_FRAC = float(os.environ.get("DETOK_RETOK_ELITE_GLOBAL_FRAC", "0.10"))
RETOK_ELITE_TOP_PER_ROUND = int(os.environ.get("DETOK_RETOK_ELITE_TOP_PER_ROUND", "4"))
RETOK_ELITE_ABLATION_MOVES = int(os.environ.get("DETOK_RETOK_ELITE_ABLATION_MOVES", "96"))
RETOK_ELITE_ABLATION_SAMPLES = int(os.environ.get("DETOK_RETOK_ELITE_ABLATION_SAMPLES", "6"))
RETOK_ELITE_PREFIX_MOVES = int(os.environ.get("DETOK_RETOK_ELITE_PREFIX_MOVES", "128"))
RETOK_ELITE_ADDBACK_MOVES = int(os.environ.get("DETOK_RETOK_ELITE_ADDBACK_MOVES", "64"))
WORD_CHAR_OBJECTIVE_AUDIT = os.environ.get("DETOK_WORD_CHAR_OBJECTIVE_AUDIT", "0") == "1"
WORD_CHAR_REF_BYTES = int(os.environ.get("DETOK_WORD_CHAR_REF_BYTES", "16000000"))
WORD_CHAR_MAX_WORDS = int(os.environ.get("DETOK_WORD_CHAR_MAX_WORDS", "100000"))
WORD_CHAR_TOP_MOVE_AUDIT = int(os.environ.get("DETOK_WORD_CHAR_TOP_MOVE_AUDIT", "512"))
WORD_CHAR_MOVE_WINDOWS = int(os.environ.get("DETOK_WORD_CHAR_MOVE_WINDOWS", "32"))
WORD_CHAR_WINDOW_RADIUS = int(os.environ.get("DETOK_WORD_CHAR_WINDOW_RADIUS", "16"))
WORD_CHAR_SAMPLE_AUDIT = int(os.environ.get("DETOK_WORD_CHAR_SAMPLE_AUDIT", "96"))

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

LAST_FINAL_EDGES: list[tuple[float, int, int]] = []
LAST_C_FOCUS: np.ndarray = np.empty(0, dtype=np.int64)
LAST_BIGRAM_CANDIDATES: dict[int, set[int]] = {}
LAST_TAIL_CANDIDATES: dict[int, set[int]] = {}
LAST_OWNER_CANDIDATES: dict[int, set[int]] = {}


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


def refine_with_bigram_objective(
    cipher_ids: np.ndarray,
    ref_ids: np.ndarray,
    mapping: np.ndarray,
    c_focus: np.ndarray,
    edges: list[tuple[float, int, int]],
    target_vocab_size: int,
    p_counts: np.ndarray,
) -> np.ndarray:
    global LAST_BIGRAM_CANDIDATES
    LAST_BIGRAM_CANDIDATES = {}
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
        LAST_BIGRAM_CANDIDATES.setdefault(int(c), set()).add(int(p))
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
    global LAST_TAIL_CANDIDATES
    LAST_TAIL_CANDIDATES = {}
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
    LAST_TAIL_CANDIDATES = {int(c): set(map(int, cand)) for c, cand in candidates_by_c.items()}

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
    global LAST_OWNER_CANDIDATES
    LAST_OWNER_CANDIDATES = {}
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
    for c, p, owner in candidates:
        LAST_OWNER_CANDIDATES.setdefault(int(c), set()).add(int(p))
        LAST_OWNER_CANDIDATES.setdefault(int(owner), set()).add(int(mapping[int(c)]))

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
    global LAST_C_FOCUS, LAST_FINAL_EDGES
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
    LAST_C_FOCUS = c_focus.copy()

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
        c_anchors = c_focus[anchor_rows]
        p_anchors = mapping[c_anchors]
        print(f"round {round_idx + 1}/{rounds}: focus={len(c_focus)} anchors={len(c_anchors)}", flush=True)
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

        edges.sort(reverse=True)
        if round_idx == rounds - 1:
            LAST_FINAL_EDGES = list(edges)
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


def inverse_permutation(perm: np.ndarray) -> np.ndarray:
    inv = np.full(int(perm.max(initial=0)) + 1, -1, dtype=np.int64)
    inv[perm.astype(np.int64, copy=False)] = np.arange(len(perm), dtype=np.int64)
    return inv


def singleton_truth_for_cipher(task, cipher_id: int, inv_perm: np.ndarray) -> tuple[int | None, str, str]:
    if cipher_id >= len(inv_perm):
        return None, "", "other"
    source_id = int(inv_perm[int(cipher_id)])
    if source_id < 0:
        return None, "", "other"
    piece = task.source_adapter.decode([source_id])
    target_ids = task.target_adapter.encode(piece)
    if len(target_ids) != 1:
        return None, piece, token_class(piece)
    return int(target_ids[0]), piece, token_class(piece)


def token_class(piece: str) -> str:
    if not piece:
        return "other"
    if piece.isspace():
        return "whitespace"
    stripped = piece.strip()
    if stripped and all(ch.isdigit() for ch in stripped):
        return "digit"
    if any(ch.isalnum() for ch in piece):
        return "alnum"
    if all((ch.isprintable() or ch.isspace()) for ch in piece):
        return "punctuation"
    return "other"


def decode_with_singleton_overrides(
    cipher_ids: np.ndarray,
    mapping: np.ndarray,
    emissions: dict[int, tuple[int, ...]],
    overrides: dict[int, int],
    target_adapter,
) -> str:
    out: list[int] = []
    for c in cipher_ids:
        c_int = int(c)
        override = overrides.get(c_int)
        if override is not None:
            out.append(int(override))
            continue
        emission = emissions.get(c_int)
        if emission is None:
            out.append(int(mapping[c_int]))
        else:
            out.extend(int(p) for p in emission)
    return target_adapter.decode(out)


def token_ids_with_singleton_overrides(
    cipher_ids: np.ndarray,
    mapping: np.ndarray,
    emissions: dict[int, tuple[int, ...]],
    overrides: dict[int, int],
) -> list[int]:
    out: list[int] = []
    for c in cipher_ids:
        c_int = int(c)
        override = overrides.get(c_int)
        if override is not None:
            out.append(int(override))
            continue
        emission = emissions.get(c_int)
        if emission is None:
            out.append(int(mapping[c_int]))
        else:
            out.extend(int(p) for p in emission)
    return out


def graph_candidate_pool(edges: list[tuple[float, int, int]], per_token: int = 64) -> dict[int, set[int]]:
    pool: dict[int, set[int]] = {}
    for _score, c, p in edges:
        bucket = pool.setdefault(int(c), set())
        if len(bucket) >= per_token:
            continue
        bucket.add(int(p))
    return pool


def graph_score_pool(edges: list[tuple[float, int, int]], per_token: int = 64) -> dict[int, list[tuple[int, float]]]:
    pool: dict[int, list[tuple[int, float]]] = {}
    seen: dict[int, set[int]] = {}
    for score, c, p in edges:
        c_int = int(c)
        p_int = int(p)
        bucket = pool.setdefault(c_int, [])
        if len(bucket) >= per_token:
            continue
        seen_bucket = seen.setdefault(c_int, set())
        if p_int in seen_bucket:
            continue
        bucket.append((p_int, float(score)))
        seen_bucket.add(p_int)
    return pool


def union_pools(*pools: dict[int, set[int]]) -> dict[int, set[int]]:
    merged: dict[int, set[int]] = {}
    for pool in pools:
        for c, values in pool.items():
            merged.setdefault(int(c), set()).update(map(int, values))
    return merged


class SparseTokenBigramLM:
    def __init__(self, ref_ids: np.ndarray, vocab_size: int, alpha: float = BIGRAM_REFINE_ALPHA):
        self.vocab_size = int(vocab_size)
        self.alpha = float(alpha)
        ref = ref_ids.astype(np.int64, copy=False)
        self.unigram = counts(ref, self.vocab_size).astype(np.float64)
        self.row_counts = np.bincount(ref[:-1], minlength=self.vocab_size).astype(np.float64)
        self.total_tokens = float(len(ref))
        print("building_sparse_retok_lm_pairs", flush=True)
        keys = ref[:-1].astype(np.int64, copy=False) * self.vocab_size + ref[1:].astype(np.int64, copy=False)
        self.keys, pair_counts = np.unique(keys, return_counts=True)
        self.pair_counts = pair_counts.astype(np.float64, copy=False)
        print(f"sparse_retok_lm_pairs={len(self.keys)}", flush=True)

    def _pair_counts(self, keys: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(self.keys, keys)
        valid = (idx < len(self.keys)) & (self.keys[idx.clip(max=max(0, len(self.keys) - 1))] == keys)
        out = np.zeros(len(keys), dtype=np.float64)
        if bool(valid.any()):
            out[valid] = self.pair_counts[idx[valid]]
        return out

    def score_ids(self, ids: list[int] | np.ndarray) -> dict[str, float]:
        arr = np.asarray(ids, dtype=np.int64)
        if len(arr) < 2:
            return {
                "bigram": 0.0,
                "backoff": 0.0,
                "bigram_nll": 0.0,
                "backoff_nll": 0.0,
                "pairs": 0.0,
            }
        prev = arr[:-1]
        nxt = arr[1:]
        mask = (prev >= 0) & (prev < self.vocab_size) & (nxt >= 0) & (nxt < self.vocab_size)
        if not bool(mask.any()):
            return {
                "bigram": 0.0,
                "backoff": 0.0,
                "bigram_nll": 0.0,
                "backoff_nll": 0.0,
                "pairs": 0.0,
            }
        prev = prev[mask]
        nxt = nxt[mask]
        keys = prev * self.vocab_size + nxt
        pair_counts = self._pair_counts(keys)
        denom = self.row_counts[prev] + self.alpha * self.vocab_size
        bigram_prob = (pair_counts + self.alpha) / np.maximum(denom, 1.0)
        bigram_score = float(np.log(bigram_prob).sum())

        unigram_prob = (self.unigram[nxt] + self.alpha) / (self.total_tokens + self.alpha * self.vocab_size)
        lam = self.row_counts[prev] / (self.row_counts[prev] + BIGRAM_UNIGRAM_BACKOFF_TAU)
        backoff_prob = lam * bigram_prob + (1.0 - lam) * unigram_prob
        backoff_score = float(np.log(np.maximum(backoff_prob, 1.0e-30)).sum())
        pairs = float(len(prev))
        return {
            "bigram": bigram_score,
            "backoff": backoff_score,
            "bigram_nll": -bigram_score / max(1.0, pairs),
            "backoff_nll": -backoff_score / max(1.0, pairs),
            "pairs": pairs,
        }


def auc_from_scores(labels: list[int], scores: list[float]) -> float:
    if not labels or len(labels) != len(scores):
        return 0.5
    label_arr = np.asarray(labels, dtype=np.int32)
    score_arr = np.asarray(scores, dtype=np.float64)
    pos = int(label_arr.sum())
    neg = int(len(label_arr) - pos)
    if pos == 0 or neg == 0:
        return 0.5
    order = np.argsort(score_arr)
    ranks = np.empty(len(score_arr), dtype=np.float64)
    sorted_scores = score_arr[order]
    start = 0
    while start < len(score_arr):
        stop = start + 1
        while stop < len(score_arr) and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        avg_rank = (start + stop + 1) / 2.0
        ranks[order[start:stop]] = avg_rank
        start = stop
    pos_rank_sum = float(ranks[label_arr == 1].sum())
    return (pos_rank_sum - pos * (pos + 1) / 2.0) / max(1.0, float(pos * neg))


def top_precision(labels: list[int], scores: list[float], k: int) -> float:
    if not labels or k <= 0:
        return 0.0
    label_arr = np.asarray(labels, dtype=np.int32)
    score_arr = np.asarray(scores, dtype=np.float64)
    take = min(k, len(label_arr))
    order = np.argsort(-score_arr)[:take]
    return float(label_arr[order].mean()) if take else 0.0


def current_piece_for_cipher(
    cipher_id: int,
    mapping: np.ndarray,
    emissions: dict[int, tuple[int, ...]],
    target_adapter,
    cache: dict[int, str],
) -> str:
    c_int = int(cipher_id)
    cached = cache.get(c_int)
    if cached is not None:
        return cached
    emission = emissions.get(c_int)
    if emission is None:
        piece = target_adapter.decode([int(mapping[c_int])])
    else:
        piece = target_adapter.decode([int(p) for p in emission])
    cache[c_int] = piece
    return piece


def target_piece(target_id: int, target_adapter, cache: dict[int, str]) -> str:
    p_int = int(target_id)
    cached = cache.get(p_int)
    if cached is not None:
        return cached
    piece = target_adapter.decode([p_int])
    cache[p_int] = piece
    return piece


def retokenized_mapping_rows(
    task,
    mapping: np.ndarray,
    emissions: dict[int, tuple[int, ...]],
    truth: dict[int, int],
    graph_pool: dict[int, set[int]],
    focus: np.ndarray,
    lm: SparseTokenBigramLM,
) -> list[dict[str, float | int | str]]:
    sample_cipher = task.cipher_ids[:SAMPLE_TOKENS]
    focus_set = set(map(int, focus))
    graph_overrides = {
        c: true_p
        for c, true_p in truth.items()
        if c in focus_set and true_p in graph_pool.get(c, set())
    }
    all_overrides = {c: true_p for c, true_p in truth.items() if c in focus_set}
    specs = [
        ("current_baseline", decode_with_variable_emissions(sample_cipher, mapping, emissions, task.target_adapter)),
        (
            "graph_top64_oracle",
            decode_with_singleton_overrides(sample_cipher, mapping, emissions, graph_overrides, task.target_adapter),
        ),
        (
            f"true_singleton_top{len(focus)}",
            decode_with_singleton_overrides(sample_cipher, mapping, emissions, all_overrides, task.target_adapter),
        ),
    ]
    rows: list[dict[str, float | int | str]] = []
    for name, text in specs:
        retok_ids = task.target_adapter.encode(text)
        metrics = evaluate_recovery(task, text, SAMPLE_TOKENS)
        score = lm.score_ids(retok_ids)
        rows.append(
            {
                "name": name,
                "cer50k": float(metrics["cer50k"]),
                "byte_lm_bpb": float(metrics["byte_lm_bpb"]),
                "retok_tokens": int(len(retok_ids)),
                "retok_bigram": float(score["bigram"]),
                "retok_backoff": float(score["backoff"]),
                "retok_bigram_nll": float(score["bigram_nll"]),
                "retok_backoff_nll": float(score["backoff_nll"]),
            }
        )
    return rows


def choose_occurrence_sample(positions: list[int], limit: int) -> list[int]:
    if len(positions) <= limit:
        return positions
    idx = np.linspace(0, len(positions) - 1, limit, dtype=np.int64)
    return [positions[int(i)] for i in idx]


def retok_move_level_audit(
    task,
    mapping: np.ndarray,
    emissions: dict[int, tuple[int, ...]],
    truth: dict[int, int],
    graph_scores: dict[int, list[tuple[int, float]]],
    lm: SparseTokenBigramLM,
) -> dict[str, object]:
    sample_cipher = task.cipher_ids[:SAMPLE_TOKENS]
    focus = [int(c) for c in LAST_C_FOCUS[: min(RETOK_MOVE_AUDIT_TYPES, len(LAST_C_FOCUS))]]
    focus_set = set(focus)
    positions_by_c: dict[int, list[int]] = {c: [] for c in focus}
    for pos, c in enumerate(sample_cipher):
        c_int = int(c)
        if c_int in focus_set:
            positions_by_c[c_int].append(pos)

    moves: list[tuple[int, int, int, float, int]] = []
    for c in focus:
        true_p = truth.get(c)
        if true_p is None:
            continue
        current_p = int(mapping[c])
        for rank, (p, score) in enumerate(graph_scores.get(c, [])[:RETOK_MOVE_AUDIT_TOPK], start=1):
            if int(p) == current_p:
                continue
            moves.append((c, int(p), rank, float(score), 1 if int(p) == true_p else 0))
    if not moves:
        return {}

    per_move_occ = max(1, min(RETOK_MAX_OCCURRENCES, RETOK_MOVE_AUDIT_MAX_WINDOWS // max(1, len(moves))))
    current_piece_cache: dict[int, str] = {}
    target_piece_cache: dict[int, str] = {}
    current_graph_score = {c: {p: score for p, score in candidates} for c, candidates in graph_scores.items()}
    labels: list[int] = []
    feature_scores: dict[str, list[float]] = {
        "retok_bigram_delta": [],
        "retok_backoff_delta": [],
        "retok_bigram_nll_delta": [],
        "retok_backoff_nll_delta": [],
        "byte_bpb_delta": [],
        "positive_fraction": [],
        "graph_delta": [],
        "combined": [],
    }

    for c, p, rank, graph_score, label in moves:
        positions = positions_by_c.get(c, [])
        if not positions:
            continue
        sampled_positions = choose_occurrence_sample(positions, per_move_occ)
        candidate_piece = target_piece(p, task.target_adapter, target_piece_cache)
        old_texts: list[str] = []
        new_texts: list[str] = []
        for pos in sampled_positions:
            start = max(0, pos - RETOK_WINDOW_RADIUS)
            stop = min(len(sample_cipher), pos + RETOK_WINDOW_RADIUS + 1)
            old_parts: list[str] = []
            new_parts: list[str] = []
            for j in range(start, stop):
                c_j = int(sample_cipher[j])
                old_piece = current_piece_for_cipher(c_j, mapping, emissions, task.target_adapter, current_piece_cache)
                old_parts.append(old_piece)
                new_parts.append(candidate_piece if j == pos else old_piece)
            old_texts.append("".join(old_parts))
            new_texts.append("".join(new_parts))

        old_ids_batch = task.target_adapter.encode_batch(old_texts)
        new_ids_batch = task.target_adapter.encode_batch(new_texts)
        bigram_deltas: list[float] = []
        backoff_deltas: list[float] = []
        bigram_nll_deltas: list[float] = []
        backoff_nll_deltas: list[float] = []
        byte_deltas: list[float] = []
        for old_text, new_text, old_ids, new_ids in zip(old_texts, new_texts, old_ids_batch, new_ids_batch):
            old_score = lm.score_ids(old_ids)
            new_score = lm.score_ids(new_ids)
            bigram_deltas.append(float(new_score["bigram"] - old_score["bigram"]))
            backoff_deltas.append(float(new_score["backoff"] - old_score["backoff"]))
            bigram_nll_deltas.append(float(old_score["bigram_nll"] - new_score["bigram_nll"]))
            backoff_nll_deltas.append(float(old_score["backoff_nll"] - new_score["backoff_nll"]))
            old_bpb = task.byte_lm.bits_per_byte(old_text.encode("utf-8", errors="replace"))
            new_bpb = task.byte_lm.bits_per_byte(new_text.encode("utf-8", errors="replace"))
            byte_deltas.append(float(old_bpb - new_bpb))

        median_bigram = float(np.median(np.asarray(bigram_deltas, dtype=np.float64)))
        median_backoff = float(np.median(np.asarray(backoff_deltas, dtype=np.float64)))
        median_bigram_nll = float(np.median(np.asarray(bigram_nll_deltas, dtype=np.float64)))
        median_backoff_nll = float(np.median(np.asarray(backoff_nll_deltas, dtype=np.float64)))
        median_byte = float(np.median(np.asarray(byte_deltas, dtype=np.float64)))
        positive_fraction = float(np.mean(np.asarray(backoff_nll_deltas, dtype=np.float64) > 0.0))
        current_score = current_graph_score.get(c, {}).get(int(mapping[c]), 0.0)
        graph_delta = float(graph_score - current_score)
        combined = median_backoff_nll + 0.25 * median_bigram_nll + 0.10 * median_byte + 0.01 * graph_delta

        labels.append(label)
        feature_scores["retok_bigram_delta"].append(median_bigram)
        feature_scores["retok_backoff_delta"].append(median_backoff)
        feature_scores["retok_bigram_nll_delta"].append(median_bigram_nll)
        feature_scores["retok_backoff_nll_delta"].append(median_backoff_nll)
        feature_scores["byte_bpb_delta"].append(median_byte)
        feature_scores["positive_fraction"].append(positive_fraction)
        feature_scores["graph_delta"].append(graph_delta)
        feature_scores["combined"].append(combined)

    feature_report: dict[str, dict[str, float]] = {}
    for name, values in feature_scores.items():
        feature_report[name] = {
            "auc": auc_from_scores(labels, values),
            "top100_precision": top_precision(labels, values, 100),
            "top500_precision": top_precision(labels, values, 500),
            "top1000_precision": top_precision(labels, values, 1000),
        }
    report = {
        "moves": len(labels),
        "positive_moves": int(sum(labels)),
        "per_move_occurrences": int(per_move_occ),
        "max_windows": int(RETOK_MOVE_AUDIT_MAX_WINDOWS),
        "features": feature_report,
    }
    print("retok_move_audit:", json.dumps(report, sort_keys=True), flush=True)
    return report


def retokenized_objective_audit(
    task,
    mapping: np.ndarray,
    emissions: dict[int, tuple[int, ...]],
) -> dict[str, object]:
    focus = LAST_C_FOCUS[: min(ORACLE_AUDIT_TYPES, len(LAST_C_FOCUS))]
    if len(focus) == 0:
        return {}
    inv = inverse_permutation(task.perm)
    truth: dict[int, int] = {}
    for c in focus:
        true_p, _piece, _cls = singleton_truth_for_cipher(task, int(c), inv)
        if true_p is not None:
            truth[int(c)] = int(true_p)

    graph_pool = graph_candidate_pool(LAST_FINAL_EDGES, TORCH_TOPK)
    graph_scores = graph_score_pool(LAST_FINAL_EDGES, TORCH_TOPK)
    lm = SparseTokenBigramLM(task.ref_ids, task.target_adapter.spec.vocab_size)
    rows = retokenized_mapping_rows(task, mapping, emissions, truth, graph_pool, focus, lm)
    print("retok_objective_rows:", flush=True)
    for row in rows:
        print(
            f"  {row['name']}: cer={row['cer50k']:.6f} bpb={row['byte_lm_bpb']:.6f} "
            f"retok_tokens={row['retok_tokens']} retok_bigram={row['retok_bigram']:.1f} "
            f"retok_backoff={row['retok_backoff']:.1f} "
            f"bigram_nll={row['retok_bigram_nll']:.6f} backoff_nll={row['retok_backoff_nll']:.6f}",
            flush=True,
        )
    move_report = retok_move_level_audit(task, mapping, emissions, truth, graph_scores, lm)
    return {
        "focus_types": int(len(focus)),
        "singleton_truth_types": int(len(truth)),
        "rows": rows,
        "move_audit": move_report,
    }


def sample_unique_moves(
    moves: list[tuple[int, int]],
    n: int,
    rng: np.random.Generator,
    used_c: set[int],
    used_p: set[int],
) -> list[tuple[int, int]]:
    if n <= 0:
        return []
    order = rng.permutation(len(moves))
    chosen: list[tuple[int, int]] = []
    for idx in order:
        c, p = moves[int(idx)]
        if c in used_c or p in used_p:
            continue
        chosen.append((c, p))
        used_c.add(c)
        used_p.add(p)
        if len(chosen) >= n:
            break
    return chosen


def pearson_corr(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    x -= float(x.mean())
    y -= float(y.mean())
    denom = float(np.sqrt((x * x).sum() * (y * y).sum()))
    if denom <= 0.0:
        return 0.0
    return float((x * y).sum() / denom)


def retok_batch_signal_audit(
    task,
    mapping: np.ndarray,
    emissions: dict[int, tuple[int, ...]],
) -> dict[str, object]:
    focus = [int(c) for c in LAST_C_FOCUS[: min(RETOK_BATCH_TYPES, len(LAST_C_FOCUS))]]
    if not focus:
        return {}
    inv = inverse_permutation(task.perm)
    truth: dict[int, int] = {}
    for c in focus:
        true_p, _piece, _cls = singleton_truth_for_cipher(task, int(c), inv)
        if true_p is not None:
            truth[int(c)] = int(true_p)

    graph_scores = graph_score_pool(LAST_FINAL_EDGES, TORCH_TOPK)
    good_moves: list[tuple[int, int]] = []
    bad_moves: list[tuple[int, int]] = []
    for c in focus:
        true_p = truth.get(c)
        if true_p is None:
            continue
        current_p = int(mapping[c])
        for p, _score in graph_scores.get(c, [])[:RETOK_BATCH_TOPK]:
            p_int = int(p)
            if p_int == current_p:
                continue
            if p_int == true_p:
                good_moves.append((c, p_int))
            else:
                bad_moves.append((c, p_int))
    if not good_moves or not bad_moves:
        return {}

    sample_cipher = task.cipher_ids[:SAMPLE_TOKENS]
    baseline_text = decode_with_variable_emissions(sample_cipher, mapping, emissions, task.target_adapter)
    baseline_metrics = evaluate_recovery(task, baseline_text, SAMPLE_TOKENS)
    baseline_cer = float(baseline_metrics["cer50k"])
    lm = SparseTokenBigramLM(task.ref_ids, task.target_adapter.spec.vocab_size)
    baseline_retok = task.target_adapter.encode(baseline_text)
    baseline_retok_score = lm.score_ids(baseline_retok)
    baseline_assigned_score = lm.score_ids(token_ids_with_singleton_overrides(sample_cipher, mapping, emissions, {}))
    baseline_bpb = task.byte_lm.bits_per_byte(baseline_text.encode("utf-8", errors="replace"))
    baseline_chars = len(baseline_text)

    owner_of_target: dict[int, int] = {}
    for c in focus:
        p = int(mapping[c])
        if p not in owner_of_target:
            owner_of_target[p] = c

    rng = np.random.default_rng(SEED + 991)
    rows: list[dict[str, float | int]] = []
    for batch_size in RETOK_BATCH_SIZES:
        if batch_size <= 0:
            continue
        for requested_fraction in RETOK_BATCH_GOOD_FRACTIONS:
            target_good = int(round(batch_size * requested_fraction))
            target_good = min(target_good, batch_size)
            for _ in range(RETOK_BATCH_REPEATS):
                used_c: set[int] = set()
                used_p: set[int] = set()
                chosen_good = sample_unique_moves(good_moves, target_good, rng, used_c, used_p)
                chosen_bad = sample_unique_moves(bad_moves, batch_size - len(chosen_good), rng, used_c, used_p)
                if len(chosen_good) + len(chosen_bad) == 0:
                    continue
                moves = chosen_good + chosen_bad
                overrides = {c: p for c, p in moves}
                text = decode_with_singleton_overrides(sample_cipher, mapping, emissions, overrides, task.target_adapter)
                retok_ids = task.target_adapter.encode(text)
                retok_score = lm.score_ids(retok_ids)
                assigned_score = lm.score_ids(token_ids_with_singleton_overrides(sample_cipher, mapping, emissions, overrides))
                metrics = evaluate_recovery(task, text, SAMPLE_TOKENS)
                bpb = task.byte_lm.bits_per_byte(text.encode("utf-8", errors="replace"))
                conflicts = sum(
                    1
                    for c, p in overrides.items()
                    if owner_of_target.get(int(p), int(c)) != int(c)
                )
                actual_good_fraction = len(chosen_good) / max(1, len(moves))
                rows.append(
                    {
                        "batch_size": int(batch_size),
                        "requested_good_fraction": float(requested_fraction),
                        "actual_good_fraction": float(actual_good_fraction),
                        "moves": int(len(moves)),
                        "target_conflicts": int(conflicts),
                        "length_delta": int(len(text) - baseline_chars),
                        "retok_backoff_nll_delta": float(
                            baseline_retok_score["backoff_nll"] - retok_score["backoff_nll"]
                        ),
                        "retok_bigram_nll_delta": float(
                            baseline_retok_score["bigram_nll"] - retok_score["bigram_nll"]
                        ),
                        "retok_backoff_delta": float(retok_score["backoff"] - baseline_retok_score["backoff"]),
                        "retok_bigram_delta": float(retok_score["bigram"] - baseline_retok_score["bigram"]),
                        "assigned_bigram_delta": float(assigned_score["bigram"] - baseline_assigned_score["bigram"]),
                        "byte_bpb_delta": float(baseline_bpb - bpb),
                        "oracle_cer_delta": float(baseline_cer - float(metrics["cer50k"])),
                    }
                )

    if not rows:
        return {}
    retok_scores = [float(row["retok_backoff_nll_delta"]) for row in rows]
    cer_deltas = [float(row["oracle_cer_delta"]) for row in rows]
    good_fracs = [float(row["actual_good_fraction"]) for row in rows]
    high_good_labels = [1 if frac >= 0.5 else 0 for frac in good_fracs]
    order = np.argsort(-np.asarray(retok_scores, dtype=np.float64))
    top_decile_n = max(1, int(np.ceil(0.10 * len(order))))
    top_decile = order[:top_decile_n]
    top_decile_good_fraction = float(np.asarray(good_fracs, dtype=np.float64)[top_decile].mean())
    best_idx = int(order[0])
    by_cell: dict[tuple[int, float], list[dict[str, float | int]]] = {}
    for row in rows:
        by_cell.setdefault((int(row["batch_size"]), float(row["requested_good_fraction"])), []).append(row)
    cell_rows: list[dict[str, float | int]] = []
    for (batch_size, frac), cell in sorted(by_cell.items()):
        cell_rows.append(
            {
                "batch_size": batch_size,
                "requested_good_fraction": frac,
                "n": len(cell),
                "actual_good_fraction": float(np.mean([float(r["actual_good_fraction"]) for r in cell])),
                "retok_backoff_nll_delta": float(np.mean([float(r["retok_backoff_nll_delta"]) for r in cell])),
                "oracle_cer_delta": float(np.mean([float(r["oracle_cer_delta"]) for r in cell])),
                "byte_bpb_delta": float(np.mean([float(r["byte_bpb_delta"]) for r in cell])),
                "target_conflicts": float(np.mean([float(r["target_conflicts"]) for r in cell])),
            }
        )

    summary = {
        "samples": len(rows),
        "good_moves": len(good_moves),
        "bad_moves": len(bad_moves),
        "baseline_cer50k": baseline_cer,
        "baseline_retok_backoff_nll": float(baseline_retok_score["backoff_nll"]),
        "pearson_retok_vs_cer_delta": pearson_corr(retok_scores, cer_deltas),
        "auc_high_good_fraction": auc_from_scores(high_good_labels, retok_scores),
        "top_decile_good_fraction": top_decile_good_fraction,
        "best_by_retok": rows[best_idx],
        "cells": cell_rows,
    }
    print("retok_batch_signal_summary:", json.dumps({k: v for k, v in summary.items() if k != "cells"}, sort_keys=True), flush=True)
    print("retok_batch_signal_cells:", flush=True)
    for row in cell_rows:
        print(
            f"  size={row['batch_size']} good={row['requested_good_fraction']:.2f} n={row['n']} "
            f"actual_good={row['actual_good_fraction']:.3f} "
            f"retok_nll_delta={row['retok_backoff_nll_delta']:.6f} "
            f"cer_delta={row['oracle_cer_delta']:.6f} bpb_delta={row['byte_bpb_delta']:.6f} "
            f"conflicts={row['target_conflicts']:.1f}",
            flush=True,
        )
    return summary


class ByteBackoffNgramLM:
    def __init__(self, order: int = 5, alpha: float = 0.05):
        self.order = int(order)
        self.alpha = float(alpha)
        self.context_counts: list[dict[tuple[int, ...], int]] = [dict() for _ in range(order)]
        self.next_counts: list[dict[tuple[int, ...], dict[int, int]]] = [dict() for _ in range(order)]

    def train(self, data: bytes) -> None:
        padded = bytes([0]) * (self.order - 1) + data
        for pos in range(self.order - 1, len(padded)):
            nxt = int(padded[pos])
            for n in range(self.order):
                ctx = tuple(padded[pos - n : pos]) if n else ()
                self.context_counts[n][ctx] = self.context_counts[n].get(ctx, 0) + 1
                bucket = self.next_counts[n].setdefault(ctx, {})
                bucket[nxt] = bucket.get(nxt, 0) + 1

    def nll_per_byte(self, data: bytes) -> float:
        if not data:
            return 99.0
        padded = bytes([0]) * (self.order - 1) + data
        nll = 0.0
        for pos in range(self.order - 1, len(padded)):
            nxt = int(padded[pos])
            for n in range(self.order - 1, -1, -1):
                ctx = tuple(padded[pos - n : pos]) if n else ()
                total = self.context_counts[n].get(ctx, 0)
                if total:
                    count = self.next_counts[n].get(ctx, {}).get(nxt, 0)
                    nll -= math.log((count + self.alpha) / (total + self.alpha * 256))
                    break
            else:
                nll += math.log(256)
        return nll / len(data)


class WordCharSourceModel:
    def __init__(self, reference_path: Path, ref_bytes: int = WORD_CHAR_REF_BYTES):
        raw = reference_path.read_bytes()[:ref_bytes]
        self.char_lm = ByteBackoffNgramLM(order=5, alpha=0.05)
        self.char_lm.train(raw)
        text = raw.decode("utf-8", errors="ignore").lower()
        word_counts: dict[str, int] = {}
        for word in re.findall(r"[a-z]+(?:'[a-z]+)?", text):
            word_counts[word] = word_counts.get(word, 0) + 1
        top_words = sorted(word_counts.items(), key=lambda item: item[1], reverse=True)[:WORD_CHAR_MAX_WORDS]
        self.word_counts = dict(top_words)
        self.word_total = float(sum(self.word_counts.values()))
        self.word_vocab = max(1, len(self.word_counts))
        self.alpha = 0.1
        self.malformed_penalty = 8.0
        print(
            f"word_char_model_ref_bytes={len(raw)} word_vocab={self.word_vocab} word_total={int(self.word_total)}",
            flush=True,
        )

    def score_text(self, text: str) -> dict[str, float]:
        data = text.encode("utf-8", errors="replace")
        char_nll = self.char_lm.nll_per_byte(data)
        tokens = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[A-Za-z0-9]+", text)
        word_nll = 0.0
        word_like = 0
        unknown = 0
        malformed = 0
        denom = self.word_total + self.alpha * (self.word_vocab + 1)
        for token in tokens:
            lower = token.lower()
            if re.fullmatch(r"[a-z]+(?:'[a-z]+)?", lower):
                word_like += 1
                count = self.word_counts.get(lower, 0)
                if count == 0:
                    unknown += 1
                word_nll -= math.log((count + self.alpha) / denom)
            elif any(ch.isalpha() for ch in lower):
                malformed += 1
                word_like += 1
                word_nll += self.malformed_penalty + self.char_lm.nll_per_byte(lower.encode("utf-8", errors="replace"))
        word_nll_per_word = word_nll / max(1, word_like)
        combined = 0.8 * word_nll_per_word + 0.2 * char_nll
        return {
            "char_nll": float(char_nll),
            "word_nll": float(word_nll_per_word),
            "word_char": float(combined),
            "word_count": int(word_like),
            "unknown_word_rate": float(unknown / max(1, word_like)),
            "malformed_rate": float(malformed / max(1, word_like)),
        }


def score_word_char_row(
    name: str,
    text: str,
    task,
    retok_lm: SparseTokenBigramLM,
    word_char_model: WordCharSourceModel,
) -> dict[str, float | int | str]:
    metrics = evaluate_recovery(task, text, SAMPLE_TOKENS)
    retok_ids = task.target_adapter.encode(text)
    retok_score = retok_lm.score_ids(retok_ids)
    word_char = word_char_model.score_text(text)
    row: dict[str, float | int | str] = {
        "name": name,
        "cer50k": float(metrics["cer50k"]),
        "byte_lm_bpb": float(metrics["byte_lm_bpb"]),
        "retok_backoff_nll": float(retok_score["backoff_nll"]),
        "retok_bigram_nll": float(retok_score["bigram_nll"]),
    }
    row.update(word_char)
    return row


def word_char_source_audit(
    task,
    sample_cipher: np.ndarray,
    mapping: np.ndarray,
    emissions: dict[int, tuple[int, ...]],
    retok_lm: SparseTokenBigramLM,
    baseline_cer: float,
    cem_overrides: dict[int, int] | None = None,
    elite_overrides: dict[int, int] | None = None,
    pool: list[dict[str, float | int]] | None = None,
    all_samples: list[tuple[float, dict[int, int], list[int], dict[str, float | int], int]] | None = None,
) -> dict[str, object]:
    word_char_model = WordCharSourceModel(task.reference_text)
    focus = LAST_C_FOCUS[: min(ORACLE_AUDIT_TYPES, len(LAST_C_FOCUS))]
    inv = inverse_permutation(task.perm)
    truth: dict[int, int] = {}
    for c in focus:
        true_p, _piece, _cls = singleton_truth_for_cipher(task, int(c), inv)
        if true_p is not None:
            truth[int(c)] = int(true_p)
    graph_pool = graph_candidate_pool(LAST_FINAL_EDGES, TORCH_TOPK)
    graph_overrides = {c: p for c, p in truth.items() if p in graph_pool.get(c, set())}
    all_truth_overrides = dict(truth)

    rows: list[dict[str, float | int | str]] = []
    baseline_text = decode_with_variable_emissions(sample_cipher, mapping, emissions, task.target_adapter)
    rows.append(score_word_char_row("baseline", baseline_text, task, retok_lm, word_char_model))
    rows.append(
        score_word_char_row(
            "graph_top64_oracle",
            decode_with_singleton_overrides(sample_cipher, mapping, emissions, graph_overrides, task.target_adapter),
            task,
            retok_lm,
            word_char_model,
        )
    )
    rows.append(
        score_word_char_row(
            f"true_singleton_top{len(focus)}",
            decode_with_singleton_overrides(sample_cipher, mapping, emissions, all_truth_overrides, task.target_adapter),
            task,
            retok_lm,
            word_char_model,
        )
    )
    if cem_overrides:
        rows.append(
            score_word_char_row(
                "cem_selected",
                decode_with_singleton_overrides(sample_cipher, mapping, emissions, cem_overrides, task.target_adapter),
                task,
                retok_lm,
                word_char_model,
            )
        )
    if elite_overrides:
        rows.append(
            score_word_char_row(
                "elite_filter_selected",
                decode_with_singleton_overrides(sample_cipher, mapping, emissions, elite_overrides, task.target_adapter),
                task,
                retok_lm,
                word_char_model,
            )
        )

    pool_good_rows: list[dict[str, float | int]] = []
    move_scores: dict[tuple[int, int], dict[str, float | int]] = {}
    if pool:
        pool_good: list[tuple[float, int, int]] = []
        for move in pool:
            c = int(move["c"])
            p = int(move["p"])
            if truth.get(c) == p:
                pool_good.append((float(move.get("prior_score", 0.0)), c, p))
        pool_good.sort(reverse=True)
        for n in (8, 16, 32, 64, 128):
            overrides: dict[int, int] = {}
            used_p: set[int] = set()
            for _prior, c, p in pool_good:
                if c in overrides or p in used_p:
                    continue
                overrides[c] = p
                used_p.add(p)
                if len(overrides) >= n:
                    break
            if not overrides:
                continue
            row = score_word_char_row(
                f"cem_pool_oracle_good_top{n}",
                decode_with_singleton_overrides(sample_cipher, mapping, emissions, overrides, task.target_adapter),
                task,
                retok_lm,
                word_char_model,
            )
            rows.append(row)
            pool_good_rows.append(
                {
                    "n": int(n),
                    "moves": int(len(overrides)),
                    "cer50k": float(row["cer50k"]),
                    "word_char": float(row["word_char"]),
                    "retok_backoff_nll": float(row["retok_backoff_nll"]),
                }
            )

        labels: list[int] = []
        scores: dict[str, list[float]] = {
            "word_char_delta": [],
            "char_nll_delta": [],
            "word_nll_delta": [],
            "retok_backoff_delta": [],
        }
        candidate_pool = pool[: min(WORD_CHAR_TOP_MOVE_AUDIT, len(pool))]
        focus_move_cs = {int(move["c"]) for move in candidate_pool}
        positions_by_c: dict[int, list[int]] = {c: [] for c in focus_move_cs}
        for pos, c in enumerate(sample_cipher):
            c_int = int(c)
            if c_int in positions_by_c:
                positions_by_c[c_int].append(pos)
        current_piece_cache: dict[int, str] = {}
        target_piece_cache: dict[int, str] = {}
        for move in candidate_pool:
            c = int(move["c"])
            p = int(move["p"])
            positions = positions_by_c.get(c, [])
            sampled_positions = choose_occurrence_sample(positions, WORD_CHAR_MOVE_WINDOWS)
            candidate_piece = target_piece(p, task.target_adapter, target_piece_cache)
            wc_deltas: list[float] = []
            char_deltas: list[float] = []
            word_deltas: list[float] = []
            retok_deltas: list[float] = []
            for pos in sampled_positions:
                start = max(0, pos - WORD_CHAR_WINDOW_RADIUS)
                stop = min(len(sample_cipher), pos + WORD_CHAR_WINDOW_RADIUS + 1)
                old_parts: list[str] = []
                new_parts: list[str] = []
                for j in range(start, stop):
                    c_j = int(sample_cipher[j])
                    old_piece = current_piece_for_cipher(c_j, mapping, emissions, task.target_adapter, current_piece_cache)
                    old_parts.append(old_piece)
                    new_parts.append(candidate_piece if j == pos else old_piece)
                old_text = "".join(old_parts)
                new_text = "".join(new_parts)
                old_wc = word_char_model.score_text(old_text)
                new_wc = word_char_model.score_text(new_text)
                old_retok = retok_lm.score_ids(task.target_adapter.encode(old_text))
                new_retok = retok_lm.score_ids(task.target_adapter.encode(new_text))
                wc_deltas.append(float(old_wc["word_char"] - new_wc["word_char"]))
                char_deltas.append(float(old_wc["char_nll"] - new_wc["char_nll"]))
                word_deltas.append(float(old_wc["word_nll"] - new_wc["word_nll"]))
                retok_deltas.append(float(old_retok["backoff_nll"] - new_retok["backoff_nll"]))
            if not wc_deltas:
                continue
            label = 1 if truth.get(c) == p else 0
            labels.append(label)
            scores["word_char_delta"].append(float(np.median(np.asarray(wc_deltas, dtype=np.float64))))
            scores["char_nll_delta"].append(float(np.median(np.asarray(char_deltas, dtype=np.float64))))
            scores["word_nll_delta"].append(float(np.median(np.asarray(word_deltas, dtype=np.float64))))
            scores["retok_backoff_delta"].append(float(np.median(np.asarray(retok_deltas, dtype=np.float64))))
            move_scores[(c, p)] = {
                "label": label,
                "word_char_delta": scores["word_char_delta"][-1],
                "char_nll_delta": scores["char_nll_delta"][-1],
                "word_nll_delta": scores["word_nll_delta"][-1],
                "retok_backoff_delta": scores["retok_backoff_delta"][-1],
            }
        move_audit = {
            name: {
                "auc": auc_from_scores(labels, values),
                "top100_precision": top_precision(labels, values, 100),
                "top500_precision": top_precision(labels, values, 500),
            }
            for name, values in scores.items()
        }
    else:
        move_audit = {}

    sample_rows: list[dict[str, float | int]] = []
    if all_samples:
        sample_wc_scores: list[float] = []
        sample_cer_deltas: list[float] = []
        sample_retok_scores: list[float] = []
        if len(all_samples) > WORD_CHAR_SAMPLE_AUDIT:
            sample_indices = np.linspace(0, len(all_samples) - 1, WORD_CHAR_SAMPLE_AUDIT, dtype=np.int64)
            scored_samples = [all_samples[int(i)] for i in sample_indices]
        else:
            scored_samples = all_samples
        for objective, overrides, _indices, _details, round_idx in scored_samples:
            text = decode_with_singleton_overrides(sample_cipher, mapping, emissions, overrides, task.target_adapter)
            wc = word_char_model.score_text(text)
            metrics = evaluate_recovery(task, text, SAMPLE_TOKENS)
            sample_wc_scores.append(-float(wc["word_char"]))
            sample_retok_scores.append(-float(objective))
            sample_cer_deltas.append(float(baseline_cer - float(metrics["cer50k"])))
            sample_rows.append(
                {
                    "round": int(round_idx),
                    "moves": int(len(overrides)),
                    "word_char": float(wc["word_char"]),
                    "cer_delta": float(sample_cer_deltas[-1]),
                    "retok_objective": float(objective),
                }
            )
        batch_audit = {
            "samples": int(len(scored_samples)),
            "total_samples": int(len(all_samples)),
            "word_char_vs_cer_corr": pearson_corr(sample_wc_scores, sample_cer_deltas),
            "retok_vs_cer_corr": pearson_corr(sample_retok_scores, sample_cer_deltas),
        }
    else:
        batch_audit = {}

    if cem_overrides and move_scores:
        selected_labels = []
        selected_word = []
        selected_retok = []
        for c, p in cem_overrides.items():
            item = move_scores.get((int(c), int(p)))
            if item is None:
                continue
            selected_labels.append(int(item["label"]))
            selected_word.append(float(item["word_char_delta"]))
            selected_retok.append(float(item["retok_backoff_delta"]))
        selected_audit = {
            "moves_scored": int(len(selected_labels)),
            "good_moves_scored": int(sum(selected_labels)),
            "word_char_auc": auc_from_scores(selected_labels, selected_word),
            "retok_auc": auc_from_scores(selected_labels, selected_retok),
            "mean_word_good": float(np.mean([s for s, y in zip(selected_word, selected_labels) if y])) if any(selected_labels) else 0.0,
            "mean_word_bad": float(np.mean([s for s, y in zip(selected_word, selected_labels) if not y]))
            if len(selected_labels) > sum(selected_labels)
            else 0.0,
        }
    else:
        selected_audit = {}

    print("word_char_objective_rows:", flush=True)
    for row in rows:
        print(
            f"  {row['name']}: cer={row['cer50k']:.6f} retok_nll={row['retok_backoff_nll']:.6f} "
            f"char_nll={row['char_nll']:.6f} word_nll={row['word_nll']:.6f} "
            f"word_char={row['word_char']:.6f} unk={row['unknown_word_rate']:.4f} malformed={row['malformed_rate']:.4f}",
            flush=True,
        )
    print(
        "word_char_objective_summary:",
        json.dumps(
            {
                "move_audit": move_audit,
                "batch_audit": batch_audit,
                "selected_audit": selected_audit,
                "pool_good_rows": pool_good_rows,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return {
        "rows": rows,
        "move_audit": move_audit,
        "batch_audit": batch_audit,
        "selected_audit": selected_audit,
        "pool_good_rows": pool_good_rows,
        "sample_rows_head": sample_rows[:10],
    }


def build_cem_candidate_moves(mapping: np.ndarray) -> list[dict[str, float | int]]:
    focus = [int(c) for c in LAST_C_FOCUS[: min(RETOK_CEM_TYPES, len(LAST_C_FOCUS))]]
    graph_scores = graph_score_pool(LAST_FINAL_EDGES, TORCH_TOPK)
    current_graph_score = {c: {p: score for p, score in candidates} for c, candidates in graph_scores.items()}
    moves: list[dict[str, float | int]] = []
    for c in focus:
        current_p = int(mapping[c])
        for rank, (p, score) in enumerate(graph_scores.get(c, [])[:RETOK_CEM_TOPK], start=1):
            p_int = int(p)
            if p_int == current_p:
                continue
            moves.append(
                {
                    "c": int(c),
                    "p": p_int,
                    "rank": int(rank),
                    "graph_delta": float(score - current_graph_score.get(c, {}).get(current_p, 0.0)),
                    "prior_score": 0.0,
                }
            )
    return moves


def score_cem_move_priors(
    task,
    mapping: np.ndarray,
    emissions: dict[int, tuple[int, ...]],
    moves: list[dict[str, float | int]],
    lm: SparseTokenBigramLM,
) -> None:
    if not moves:
        return
    sample_cipher = task.cipher_ids[:SAMPLE_TOKENS]
    focus = {int(move["c"]) for move in moves}
    positions_by_c: dict[int, list[int]] = {c: [] for c in focus}
    for pos, c in enumerate(sample_cipher):
        c_int = int(c)
        if c_int in positions_by_c:
            positions_by_c[c_int].append(pos)
    per_move_occ = max(1, min(RETOK_CEM_PRIOR_MAX_OCC, RETOK_CEM_PRIOR_MAX_WINDOWS // max(1, len(moves))))
    current_piece_cache: dict[int, str] = {}
    target_piece_cache: dict[int, str] = {}
    for move in moves:
        c = int(move["c"])
        p = int(move["p"])
        positions = positions_by_c.get(c, [])
        if not positions:
            move["prior_score"] = -999.0
            continue
        sampled_positions = choose_occurrence_sample(positions, per_move_occ)
        candidate_piece = target_piece(p, task.target_adapter, target_piece_cache)
        old_texts: list[str] = []
        new_texts: list[str] = []
        for pos in sampled_positions:
            start = max(0, pos - RETOK_WINDOW_RADIUS)
            stop = min(len(sample_cipher), pos + RETOK_WINDOW_RADIUS + 1)
            old_parts: list[str] = []
            new_parts: list[str] = []
            for j in range(start, stop):
                c_j = int(sample_cipher[j])
                old_piece = current_piece_for_cipher(c_j, mapping, emissions, task.target_adapter, current_piece_cache)
                old_parts.append(old_piece)
                new_parts.append(candidate_piece if j == pos else old_piece)
            old_texts.append("".join(old_parts))
            new_texts.append("".join(new_parts))
        old_ids_batch = task.target_adapter.encode_batch(old_texts)
        new_ids_batch = task.target_adapter.encode_batch(new_texts)
        nll_deltas: list[float] = []
        for old_ids, new_ids in zip(old_ids_batch, new_ids_batch):
            old_score = lm.score_ids(old_ids)
            new_score = lm.score_ids(new_ids)
            nll_deltas.append(float(old_score["backoff_nll"] - new_score["backoff_nll"]))
        move["prior_score"] = float(np.median(np.asarray(nll_deltas, dtype=np.float64)))


def sample_weighted_cem_batch(
    pool: list[dict[str, float | int]],
    weights: np.ndarray,
    rng: np.random.Generator,
    batch_moves: int,
) -> list[dict[str, float | int]]:
    if not pool or batch_moves <= 0:
        return []
    probs = weights / max(float(weights.sum()), 1.0e-30)
    order = rng.choice(len(pool), size=len(pool), replace=False, p=probs)
    used_c: set[int] = set()
    used_p: set[int] = set()
    chosen: list[dict[str, float | int]] = []
    for idx in order:
        move = pool[int(idx)]
        c = int(move["c"])
        p = int(move["p"])
        if c in used_c or p in used_p:
            continue
        chosen.append(move)
        used_c.add(c)
        used_p.add(p)
        if len(chosen) >= batch_moves:
            break
    return chosen


def score_cem_overrides(
    task,
    sample_cipher: np.ndarray,
    mapping: np.ndarray,
    emissions: dict[int, tuple[int, ...]],
    lm: SparseTokenBigramLM,
    baseline_chars: int,
    owner_of_target: dict[int, int],
    overrides: dict[int, int],
) -> tuple[float, dict[str, float | int]]:
    text = decode_with_singleton_overrides(sample_cipher, mapping, emissions, overrides, task.target_adapter)
    retok_ids = task.target_adapter.encode(text)
    retok_score = lm.score_ids(retok_ids)
    conflicts = sum(
        1
        for c, p in overrides.items()
        if owner_of_target.get(int(p), int(c)) != int(c)
    )
    length_delta = len(text) - baseline_chars
    avg_rank = 0.0
    if overrides:
        # Rank is filled by the caller through the override move list.
        avg_rank = 0.0
    objective = float(retok_score["backoff_nll"])
    objective += RETOK_CEM_DUP_PENALTY * (conflicts / max(1, len(overrides)))
    objective += RETOK_CEM_LEN_PENALTY * (abs(length_delta) / max(1, baseline_chars))
    details = {
        "objective": objective,
        "retok_backoff_nll": float(retok_score["backoff_nll"]),
        "retok_bigram_nll": float(retok_score["bigram_nll"]),
        "retok_tokens": int(len(retok_ids)),
        "target_conflicts": int(conflicts),
        "length_delta": int(length_delta),
        "moves": int(len(overrides)),
        "avg_rank": float(avg_rank),
    }
    return objective, details


def cem_objective_with_rank_penalty(
    task,
    sample_cipher: np.ndarray,
    mapping: np.ndarray,
    emissions: dict[int, tuple[int, ...]],
    lm: SparseTokenBigramLM,
    baseline_chars: int,
    owner_of_target: dict[int, int],
    overrides: dict[int, int],
    move_by_key: dict[tuple[int, int], dict[str, float | int]],
) -> tuple[float, dict[str, float | int]]:
    objective, details = score_cem_overrides(
        task,
        sample_cipher,
        mapping,
        emissions,
        lm,
        baseline_chars,
        owner_of_target,
        overrides,
    )
    ranks = [float(move_by_key[(int(c), int(p))]["rank"]) for c, p in overrides.items() if (int(c), int(p)) in move_by_key]
    avg_rank = float(np.mean(ranks)) if ranks else 0.0
    objective += RETOK_CEM_GRAPH_RANK_PENALTY * (avg_rank / max(1.0, float(RETOK_CEM_TOPK)))
    details["objective"] = float(objective)
    details["avg_rank"] = avg_rank
    return float(objective), details


def select_elite_samples(
    samples: list[tuple[float, dict[int, int], list[int], dict[str, float | int], int]],
) -> list[tuple[float, dict[int, int], list[int], dict[str, float | int], int]]:
    if not samples:
        return []
    sorted_samples = sorted(samples, key=lambda item: item[0])
    elite_count = max(1, int(np.ceil(RETOK_ELITE_GLOBAL_FRAC * len(sorted_samples))))
    selected: dict[int, tuple[float, dict[int, int], list[int], dict[str, float | int], int]] = {
        id(item): item for item in sorted_samples[:elite_count]
    }
    by_round: dict[int, list[tuple[float, dict[int, int], list[int], dict[str, float | int], int]]] = {}
    for item in sorted_samples:
        by_round.setdefault(int(item[4]), []).append(item)
    for round_samples in by_round.values():
        for item in round_samples[:RETOK_ELITE_TOP_PER_ROUND]:
            selected[id(item)] = item
    return list(selected.values())


def elite_move_statistics(
    pool: list[dict[str, float | int]],
    all_samples: list[tuple[float, dict[int, int], list[int], dict[str, float | int], int]],
    elites: list[tuple[float, dict[int, int], list[int], dict[str, float | int], int]],
) -> list[dict[str, float | int]]:
    n_all = max(1, len(all_samples))
    n_elite = max(1, len(elites))
    elite_ids = {id(item) for item in elites}
    move_stats: list[dict[str, float | int]] = []
    eps = 0.5
    for idx, move in enumerate(pool):
        elite_contains = 0
        nonelite_contains = 0
        elite_objectives: list[float] = []
        for sample in all_samples:
            contains = idx in sample[2]
            if not contains:
                continue
            if id(sample) in elite_ids:
                elite_contains += 1
                elite_objectives.append(float(sample[0]))
            else:
                nonelite_contains += 1
        nonelite_total = max(1, n_all - n_elite)
        enrichment = np.log((elite_contains + eps) / (n_elite - elite_contains + eps)) - np.log(
            (nonelite_contains + eps) / (nonelite_total - nonelite_contains + eps)
        )
        move_stats.append(
            {
                "idx": int(idx),
                "c": int(move["c"]),
                "p": int(move["p"]),
                "rank": int(move["rank"]),
                "prior_score": float(move.get("prior_score", 0.0)),
                "elite_frequency": float(elite_contains / n_elite),
                "nonelite_frequency": float(nonelite_contains / nonelite_total),
                "elite_enrichment": float(enrichment),
                "mean_elite_objective": float(np.mean(elite_objectives)) if elite_objectives else 999.0,
                "ablation_support": 0.0,
            }
        )
    move_stats.sort(
        key=lambda row: (
            float(row["elite_enrichment"]),
            float(row["elite_frequency"]),
            float(row["prior_score"]),
            -float(row["rank"]),
        ),
        reverse=True,
    )
    return move_stats


def add_ablation_support(
    task,
    sample_cipher: np.ndarray,
    mapping: np.ndarray,
    emissions: dict[int, tuple[int, ...]],
    lm: SparseTokenBigramLM,
    baseline_chars: int,
    owner_of_target: dict[int, int],
    move_by_key: dict[tuple[int, int], dict[str, float | int]],
    all_samples: list[tuple[float, dict[int, int], list[int], dict[str, float | int], int]],
    elites: list[tuple[float, dict[int, int], list[int], dict[str, float | int], int]],
    move_stats: list[dict[str, float | int]],
) -> None:
    elite_ids = {id(item) for item in elites}
    for row in move_stats[:RETOK_ELITE_ABLATION_MOVES]:
        idx = int(row["idx"])
        supports: list[float] = []
        containing = [sample for sample in all_samples if id(sample) in elite_ids and idx in sample[2]]
        containing = sorted(containing, key=lambda item: item[0])[:RETOK_ELITE_ABLATION_SAMPLES]
        for objective, overrides, _indices, _details, _round in containing:
            c = int(row["c"])
            if c not in overrides:
                continue
            ablated = dict(overrides)
            ablated.pop(c, None)
            ablated_objective, _ = cem_objective_with_rank_penalty(
                task,
                sample_cipher,
                mapping,
                emissions,
                lm,
                baseline_chars,
                owner_of_target,
                ablated,
                move_by_key,
            )
            supports.append(float(ablated_objective - objective))
        if supports:
            row["ablation_support"] = float(np.median(np.asarray(supports, dtype=np.float64)))


def elite_prefix_finalizer(
    task,
    sample_cipher: np.ndarray,
    mapping: np.ndarray,
    emissions: dict[int, tuple[int, ...]],
    lm: SparseTokenBigramLM,
    baseline_chars: int,
    owner_of_target: dict[int, int],
    pool: list[dict[str, float | int]],
    all_samples: list[tuple[float, dict[int, int], list[int], dict[str, float | int], int]],
    baseline_objective: float,
) -> tuple[dict[int, int], dict[str, object]]:
    elites = select_elite_samples(all_samples)
    if not elites:
        return {}, {}
    move_by_key = {(int(move["c"]), int(move["p"])): move for move in pool}
    move_stats = elite_move_statistics(pool, all_samples, elites)
    add_ablation_support(
        task,
        sample_cipher,
        mapping,
        emissions,
        lm,
        baseline_chars,
        owner_of_target,
        move_by_key,
        all_samples,
        elites,
        move_stats,
    )
    ranked = sorted(
        move_stats,
        key=lambda row: (
            float(row["ablation_support"]),
            float(row["elite_enrichment"]),
            float(row["elite_frequency"]),
            float(row["prior_score"]),
            -float(row["rank"]),
        ),
        reverse=True,
    )

    selected: dict[int, int] = {}
    best_overrides: dict[int, int] = {}
    best_objective = baseline_objective
    best_details: dict[str, float | int] = {}
    prefix_rows: list[dict[str, float | int]] = []
    for row in ranked[:RETOK_ELITE_PREFIX_MOVES]:
        c = int(row["c"])
        p = int(row["p"])
        if c in selected:
            continue
        selected[c] = p
        objective, details = cem_objective_with_rank_penalty(
            task,
            sample_cipher,
            mapping,
            emissions,
            lm,
            baseline_chars,
            owner_of_target,
            selected,
            move_by_key,
        )
        prefix_rows.append(
            {
                "moves": int(len(selected)),
                "objective": float(objective),
                "target_conflicts": int(details["target_conflicts"]),
                "length_delta": int(details["length_delta"]),
            }
        )
        if objective < best_objective:
            best_objective = float(objective)
            best_overrides = dict(selected)
            best_details = dict(details)

    selected = dict(best_overrides)
    prune_removed = 0
    improved = True
    while improved and selected:
        improved = False
        for c in list(selected.keys()):
            trial = dict(selected)
            trial.pop(c, None)
            objective, details = cem_objective_with_rank_penalty(
                task,
                sample_cipher,
                mapping,
                emissions,
                lm,
                baseline_chars,
                owner_of_target,
                trial,
                move_by_key,
            )
            if objective < best_objective:
                selected = trial
                best_objective = float(objective)
                best_details = dict(details)
                prune_removed += 1
                improved = True
                break

    used_p = set(map(int, selected.values()))
    addback_added = 0
    for row in ranked[:RETOK_ELITE_ADDBACK_MOVES]:
        c = int(row["c"])
        p = int(row["p"])
        if c in selected or p in used_p:
            continue
        trial = dict(selected)
        trial[c] = p
        objective, details = cem_objective_with_rank_penalty(
            task,
            sample_cipher,
            mapping,
            emissions,
            lm,
            baseline_chars,
            owner_of_target,
            trial,
            move_by_key,
        )
        if objective < best_objective:
            selected = trial
            used_p.add(p)
            best_objective = float(objective)
            best_details = dict(details)
            addback_added += 1

    report = {
        "elite_samples": int(len(elites)),
        "all_samples": int(len(all_samples)),
        "ranked_moves": int(len(ranked)),
        "selected_moves": int(len(selected)),
        "best_objective": float(best_objective),
        "best_details": best_details,
        "prune_removed": int(prune_removed),
        "addback_added": int(addback_added),
        "top_ranked_moves": [
            {
                "c": int(row["c"]),
                "p": int(row["p"]),
                "rank": int(row["rank"]),
                "elite_frequency": float(row["elite_frequency"]),
                "elite_enrichment": float(row["elite_enrichment"]),
                "ablation_support": float(row["ablation_support"]),
                "prior_score": float(row["prior_score"]),
            }
            for row in ranked[:10]
        ],
        "prefix_rows": prefix_rows[:20],
    }
    print(
        "retok_elite_filter_summary:",
        json.dumps({k: v for k, v in report.items() if k not in {"top_ranked_moves", "prefix_rows"}}, sort_keys=True),
        flush=True,
    )
    return selected, report


def selected_cem_move_oracle_audit(
    task,
    sample_cipher: np.ndarray,
    mapping: np.ndarray,
    emissions: dict[int, tuple[int, ...]],
    selected_overrides: dict[int, int],
    pool: list[dict[str, float | int]],
    baseline_cer: float,
) -> dict[str, object]:
    inv = inverse_permutation(task.perm)
    truth: dict[int, int] = {}
    for c in set(selected_overrides) | {int(move["c"]) for move in pool}:
        true_p, _piece, _cls = singleton_truth_for_cipher(task, int(c), inv)
        if true_p is not None:
            truth[int(c)] = int(true_p)

    good_selected = {c: p for c, p in selected_overrides.items() if truth.get(int(c)) == int(p)}
    bad_selected = {c: p for c, p in selected_overrides.items() if truth.get(int(c)) != int(p)}
    selected_text = decode_with_singleton_overrides(sample_cipher, mapping, emissions, selected_overrides, task.target_adapter)
    selected_metrics = evaluate_recovery(task, selected_text, SAMPLE_TOKENS)
    good_only_text = decode_with_singleton_overrides(sample_cipher, mapping, emissions, good_selected, task.target_adapter)
    good_only_metrics = evaluate_recovery(task, good_only_text, SAMPLE_TOKENS)
    bad_reverted_overrides = dict(good_selected)
    bad_reverted_text = decode_with_singleton_overrides(
        sample_cipher,
        mapping,
        emissions,
        bad_reverted_overrides,
        task.target_adapter,
    )
    bad_reverted_metrics = evaluate_recovery(task, bad_reverted_text, SAMPLE_TOKENS)

    pool_good: list[tuple[float, int, int]] = []
    for move in pool:
        c = int(move["c"])
        p = int(move["p"])
        if truth.get(c) == p:
            pool_good.append((float(move.get("prior_score", 0.0)), c, p))
    pool_good.sort(reverse=True)
    best_n_rows: list[dict[str, float | int]] = []
    for n in (8, 16, 32, 64, 128):
        overrides: dict[int, int] = {}
        used_p: set[int] = set()
        for _prior, c, p in pool_good:
            if c in overrides or p in used_p:
                continue
            overrides[c] = p
            used_p.add(p)
            if len(overrides) >= n:
                break
        if not overrides:
            continue
        text = decode_with_singleton_overrides(sample_cipher, mapping, emissions, overrides, task.target_adapter)
        metrics = evaluate_recovery(task, text, SAMPLE_TOKENS)
        best_n_rows.append(
            {
                "n": int(n),
                "moves": int(len(overrides)),
                "cer50k": float(metrics["cer50k"]),
                "cer_delta": float(baseline_cer - float(metrics["cer50k"])),
            }
        )

    c_counts = counts(task.cipher_ids, int(max(len(mapping), int(task.cipher_ids.max(initial=0)) + 1)))
    good_mass = float(sum(int(c_counts[c]) for c in good_selected)) / max(1.0, float(len(sample_cipher)))
    bad_mass = float(sum(int(c_counts[c]) for c in bad_selected)) / max(1.0, float(len(sample_cipher)))
    audit = {
        "selected_moves": int(len(selected_overrides)),
        "oracle_good_moves": int(len(good_selected)),
        "oracle_bad_moves": int(len(bad_selected)),
        "good_move_precision": float(len(good_selected) / max(1, len(selected_overrides))),
        "good_move_mass": good_mass,
        "bad_move_mass": bad_mass,
        "selected_cer50k": float(selected_metrics["cer50k"]),
        "selected_cer_delta": float(baseline_cer - float(selected_metrics["cer50k"])),
        "good_only_cer50k": float(good_only_metrics["cer50k"]),
        "good_only_cer_delta": float(baseline_cer - float(good_only_metrics["cer50k"])),
        "bad_reverted_cer50k": float(bad_reverted_metrics["cer50k"]),
        "bad_reverted_cer_delta": float(baseline_cer - float(bad_reverted_metrics["cer50k"])),
        "best_n_pool_good": best_n_rows,
    }
    print("retok_cem_selected_oracle_audit:", json.dumps(audit, sort_keys=True), flush=True)
    return audit


def retok_cem_search(
    task,
    mapping: np.ndarray,
    emissions: dict[int, tuple[int, ...]],
) -> tuple[np.ndarray, dict[int, tuple[int, ...]], dict[str, object]]:
    sample_cipher = task.cipher_ids[:SAMPLE_TOKENS]
    baseline_text = decode_with_variable_emissions(sample_cipher, mapping, emissions, task.target_adapter)
    baseline_metrics = evaluate_recovery(task, baseline_text, SAMPLE_TOKENS)
    baseline_chars = len(baseline_text)
    lm = SparseTokenBigramLM(task.ref_ids, task.target_adapter.spec.vocab_size)
    baseline_objective, baseline_details = score_cem_overrides(
        task,
        sample_cipher,
        mapping,
        emissions,
        lm,
        baseline_chars,
        {},
        {},
    )
    moves = build_cem_candidate_moves(mapping)
    score_cem_move_priors(task, mapping, emissions, moves, lm)
    moves.sort(key=lambda move: (float(move["prior_score"]), -int(move["rank"]), float(move["graph_delta"])), reverse=True)
    pool = moves[: min(RETOK_CEM_MOVE_POOL, len(moves))]
    if not pool:
        return mapping, emissions, {}
    prior_scores = np.asarray([float(move["prior_score"]) for move in pool], dtype=np.float64)
    finite = np.isfinite(prior_scores)
    if bool(finite.any()):
        lo = float(np.percentile(prior_scores[finite], 5))
        hi = float(np.percentile(prior_scores[finite], 95))
        denom = max(hi - lo, 1.0e-9)
        norm_prior = np.clip((prior_scores - lo) / denom, 0.0, 1.0)
    else:
        norm_prior = np.zeros(len(pool), dtype=np.float64)
    rank_prior = np.asarray([1.0 / max(1.0, float(move["rank"])) for move in pool], dtype=np.float64)
    weights = 0.75 * norm_prior + 0.25 * rank_prior + RETOK_CEM_ENTROPY_FLOOR
    weights = weights.astype(np.float64)

    owner_of_target: dict[int, int] = {}
    for c in LAST_C_FOCUS[: min(RETOK_CEM_TYPES, len(LAST_C_FOCUS))]:
        c_int = int(c)
        p_int = int(mapping[c_int])
        if p_int not in owner_of_target:
            owner_of_target[p_int] = c_int

    rng = np.random.default_rng(SEED + 4242)
    best_overrides: dict[int, int] = {}
    best_objective = baseline_objective
    best_details = dict(baseline_details)
    round_reports: list[dict[str, float | int]] = []
    elite_n = max(1, int(round(RETOK_CEM_SAMPLES * RETOK_CEM_ELITE_FRAC)))
    all_samples: list[tuple[float, dict[int, int], list[int], dict[str, float | int], int]] = []

    for round_idx in range(RETOK_CEM_ROUNDS):
        samples: list[tuple[float, dict[int, int], list[int], dict[str, float | int], int]] = []
        for _ in range(RETOK_CEM_SAMPLES):
            chosen = sample_weighted_cem_batch(pool, weights, rng, RETOK_CEM_BATCH_MOVES)
            if not chosen:
                continue
            overrides = {int(move["c"]): int(move["p"]) for move in chosen}
            objective, details = score_cem_overrides(
                task,
                sample_cipher,
                mapping,
                emissions,
                lm,
                baseline_chars,
                owner_of_target,
                overrides,
            )
            avg_rank = float(np.mean([float(move["rank"]) for move in chosen]))
            objective += RETOK_CEM_GRAPH_RANK_PENALTY * (avg_rank / max(1.0, float(RETOK_CEM_TOPK)))
            details["objective"] = objective
            details["avg_rank"] = avg_rank
            indices = [pool.index(move) for move in chosen]
            sample_record = (objective, overrides, indices, details, round_idx + 1)
            samples.append(sample_record)
            all_samples.append(sample_record)
        if not samples:
            break
        samples.sort(key=lambda item: item[0])
        elites = samples[:elite_n]
        counts_elite = np.zeros(len(pool), dtype=np.float64)
        for _objective, _overrides, indices, _details, _round in elites:
            counts_elite[indices] += 1.0
        elite_weights = counts_elite + RETOK_CEM_ENTROPY_FLOOR
        weights = (1.0 - RETOK_CEM_UPDATE_RATE) * weights + RETOK_CEM_UPDATE_RATE * elite_weights
        if samples[0][0] < best_objective:
            best_objective = float(samples[0][0])
            best_overrides = dict(samples[0][1])
            best_details = dict(samples[0][3])
        round_reports.append(
            {
                "round": int(round_idx + 1),
                "best_objective": float(samples[0][0]),
                "elite_mean_objective": float(np.mean([item[0] for item in elites])),
                "best_moves": int(samples[0][3]["moves"]),
                "best_conflicts": int(samples[0][3]["target_conflicts"]),
                "best_length_delta": int(samples[0][3]["length_delta"]),
            }
        )
        print(
            f"retok_cem_round={round_idx + 1} best_objective={samples[0][0]:.6f} "
            f"elite_mean={round_reports[-1]['elite_mean_objective']:.6f} "
            f"best_moves={samples[0][3]['moves']} conflicts={samples[0][3]['target_conflicts']} "
            f"length_delta={samples[0][3]['length_delta']}",
            flush=True,
        )

    report: dict[str, object] = {
        "baseline_cer50k": float(baseline_metrics["cer50k"]),
        "baseline_objective": float(baseline_objective),
        "baseline_details": baseline_details,
        "candidate_moves": len(moves),
        "move_pool": len(pool),
        "pool_prior_top": [
            {
                "c": int(move["c"]),
                "p": int(move["p"]),
                "rank": int(move["rank"]),
                "prior_score": float(move["prior_score"]),
            }
            for move in pool[:10]
        ],
        "rounds": round_reports,
        "best_objective": float(best_objective),
        "best_details": best_details,
        "accepted": bool(best_overrides),
    }
    cem_sample_overrides = dict(best_overrides)
    elite_overrides: dict[int, int] = {}
    if RETOK_ELITE_FILTER and all_samples:
        elite_overrides, elite_report = elite_prefix_finalizer(
            task,
            sample_cipher,
            mapping,
            emissions,
            lm,
            baseline_chars,
            owner_of_target,
            pool,
            all_samples,
            baseline_objective,
        )
        report["elite_filter"] = elite_report
        if elite_overrides:
            elite_objective = float(elite_report.get("best_objective", baseline_objective))
            if elite_objective < best_objective:
                best_overrides = dict(elite_overrides)
                best_objective = elite_objective
                best_details = dict(elite_report.get("best_details", {}))
                report["best_source"] = "elite_filter"
            else:
                report["best_source"] = "cem_sample"
        else:
            report["best_source"] = "cem_sample"
    else:
        report["best_source"] = "cem_sample"
    if WORD_CHAR_OBJECTIVE_AUDIT:
        report["word_char_objective_audit"] = word_char_source_audit(
            task,
            sample_cipher,
            mapping,
            emissions,
            lm,
            float(baseline_metrics["cer50k"]),
            cem_sample_overrides if cem_sample_overrides else None,
            elite_overrides if elite_overrides else None,
            pool,
            all_samples,
        )
    if not best_overrides:
        print("retok_cem_no_objective_improvement", flush=True)
        return mapping, emissions, report

    repaired = mapping.copy()
    repaired_emissions = dict(emissions)
    for c, p in best_overrides.items():
        repaired[int(c)] = int(p)
        repaired_emissions.pop(int(c), None)
    repaired_text = decode_with_variable_emissions(sample_cipher, repaired, repaired_emissions, task.target_adapter)
    repaired_metrics = evaluate_recovery(task, repaired_text, SAMPLE_TOKENS)
    selected_audit = selected_cem_move_oracle_audit(
        task,
        sample_cipher,
        mapping,
        emissions,
        best_overrides,
        pool,
        float(baseline_metrics["cer50k"]),
    )
    report["selected_move_oracle_audit"] = selected_audit
    report["repaired_metrics"] = repaired_metrics
    report["cer_delta"] = float(baseline_metrics["cer50k"] - repaired_metrics["cer50k"])
    print(
        "retok_cem_result:",
        json.dumps(
            {
                "baseline_cer50k": float(baseline_metrics["cer50k"]),
                "repaired_cer50k": float(repaired_metrics["cer50k"]),
                "cer_delta": report["cer_delta"],
                "moves": len(best_overrides),
                "best_objective": best_objective,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return repaired, repaired_emissions, report


def oracle_audit(task, mapping: np.ndarray, emissions: dict[int, tuple[int, ...]]) -> dict[str, object]:
    sample_cipher = task.cipher_ids[:SAMPLE_TOKENS]
    focus = LAST_C_FOCUS[: min(ORACLE_AUDIT_TYPES, len(LAST_C_FOCUS))]
    if len(focus) == 0:
        return {}

    inv = inverse_permutation(task.perm)
    c_counts = counts(task.cipher_ids, int(max(len(mapping), int(task.cipher_ids.max(initial=0)) + 1)))
    truth: dict[int, int] = {}
    classes: dict[int, str] = {}
    for c in focus:
        c_int = int(c)
        true_p, _piece, cls = singleton_truth_for_cipher(task, c_int, inv)
        classes[c_int] = cls
        if true_p is not None:
            truth[c_int] = true_p

    baseline_text = decode_with_variable_emissions(sample_cipher, mapping, emissions, task.target_adapter)
    baseline_metrics = evaluate_recovery(task, baseline_text, SAMPLE_TOKENS)
    baseline_cer = float(baseline_metrics["cer50k"])

    graph_pool = graph_candidate_pool(LAST_FINAL_EDGES, TORCH_TOPK)
    bigram_pool = {int(c): set(map(int, values)) for c, values in LAST_BIGRAM_CANDIDATES.items()}
    tail_pool = {int(c): set(map(int, values)) for c, values in LAST_TAIL_CANDIDATES.items()}
    owner_pool = {int(c): set(map(int, values)) for c, values in LAST_OWNER_CANDIDATES.items()}
    union_pool = union_pools(graph_pool, bigram_pool, tail_pool, owner_pool)

    focus_set = set(map(int, focus))
    sample_mass_den = max(1, len(sample_cipher))

    def run_pool(name: str, pool: dict[int, set[int]], allowed: set[int] | None = None) -> dict[str, float | int | str]:
        allowed_set = focus_set if allowed is None else allowed
        overrides = {
            c: true_p
            for c, true_p in truth.items()
            if c in allowed_set and true_p in pool.get(c, set())
        }
        changed = {
            c: true_p
            for c, true_p in overrides.items()
            if int(mapping[c]) != int(true_p) or c in emissions
        }
        text = decode_with_singleton_overrides(sample_cipher, mapping, emissions, overrides, task.target_adapter)
        metrics = evaluate_recovery(task, text, SAMPLE_TOKENS)
        mass = float(sum(int(c_counts[c]) for c in overrides)) / sample_mass_den
        return {
            "name": name,
            "cer50k": float(metrics["cer50k"]),
            "gain": baseline_cer - float(metrics["cer50k"]),
            "eligible_types": len(overrides),
            "changed_types": len(changed),
            "eligible_mass": mass,
        }

    rows: list[dict[str, float | int | str]] = [
        {
            "name": "current_mapping",
            "cer50k": baseline_cer,
            "gain": 0.0,
            "eligible_types": 0,
            "changed_types": 0,
            "eligible_mass": 0.0,
        },
        run_pool("graph_top64_oracle", graph_pool),
        run_pool("bigram_swap_pool_oracle", bigram_pool),
        run_pool("tail_pool_oracle", tail_pool),
        run_pool("owner_pool_oracle", owner_pool),
        run_pool("union_pool_oracle", union_pool),
    ]

    for limit in (128, 512, 2048, 4096, min(8192, len(focus))):
        allowed = set(map(int, focus[:limit]))
        all_truth_pool = {c: {p} for c, p in truth.items()}
        rows.append(run_pool(f"true_singleton_top{limit}", all_truth_pool, allowed))

    if len(focus) > 4096:
        tail_allowed = set(map(int, focus[4096 : min(8192, len(focus))]))
        rows.append(run_pool("union_tail_4096_8192", union_pool, tail_allowed))

    for cls in ("alnum", "punctuation", "whitespace", "digit"):
        allowed = {c for c in focus_set if classes.get(c) == cls}
        rows.append(run_pool(f"union_class_{cls}", union_pool, allowed))

    owner_allowed = set(LAST_OWNER_CANDIDATES.keys()) & focus_set
    rows.append(run_pool("union_owner_conflict_types", union_pool, owner_allowed))

    not_in_p_nodes = {
        c
        for c, true_p in truth.items()
        if true_p in graph_pool.get(c, set()) and true_p not in bigram_pool.get(c, set())
    }
    rows.append(run_pool("union_graph_true_not_in_bigram_pool", union_pool, not_in_p_nodes))

    singleton_types = len(truth)
    current_correct = sum(1 for c, true_p in truth.items() if int(mapping[c]) == int(true_p) and c not in emissions)
    graph_recall = sum(1 for c, true_p in truth.items() if true_p in graph_pool.get(c, set()))
    union_recall = sum(1 for c, true_p in truth.items() if true_p in union_pool.get(c, set()))
    audit = {
        "baseline_cer50k": baseline_cer,
        "focus_types": int(len(focus)),
        "singleton_truth_types": singleton_types,
        "current_top1_type_acc": current_correct / max(1, singleton_types),
        "graph_top64_type_recall": graph_recall / max(1, singleton_types),
        "union_type_recall": union_recall / max(1, singleton_types),
        "rows": rows,
    }
    print("oracle_audit_summary:", json.dumps({k: v for k, v in audit.items() if k != "rows"}, sort_keys=True), flush=True)
    print("oracle_audit_rows:", flush=True)
    for row in rows:
        print(
            f"  {row['name']}: cer={row['cer50k']:.6f} gain={row['gain']:.6f} "
            f"types={row['eligible_types']} changed={row['changed_types']} mass={row['eligible_mass']:.4f}",
            flush=True,
        )
    return audit


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
    print(f"cipher_tokens: {len(task.cipher_ids):,}")
    print(f"reference_tokens: {len(task.ref_ids):,}")

    mapping = align_shuffled(task.cipher_ids, task.ref_ids, task.target_adapter.spec.vocab_size)
    emissions = variable_emission_repair(task.cipher_ids, task.ref_ids, mapping, task.target_adapter.spec.vocab_size)
    cem_report: dict[str, object] = {}
    if RETOK_CEM_SEARCH:
        mapping, emissions, cem_report = retok_cem_search(task, mapping, emissions)
    recovered_sample = decode_with_variable_emissions(
        task.cipher_ids[:SAMPLE_TOKENS],
        mapping,
        emissions,
        task.target_adapter,
    )
    metrics = evaluate_recovery(task, recovered_sample, SAMPLE_TOKENS)
    diagnostics: dict[str, int] = {}
    if ENABLE_DIAGNOSTICS:
        original_sample = task.source_adapter.decode(task.secret_ids[:SAMPLE_TOKENS].astype(int).tolist())
        diagnostics = error_breakdown(original_sample, recovered_sample)
        print(f"error_breakdown: {json.dumps(diagnostics, sort_keys=True)}", flush=True)
    oracle_report: dict[str, object] = {}
    if ORACLE_AUDIT:
        oracle_report = oracle_audit(task, mapping, emissions)
    retok_report: dict[str, object] = {}
    if RETOK_OBJECTIVE_AUDIT:
        retok_report = retokenized_objective_audit(task, mapping, emissions)
    retok_batch_report: dict[str, object] = {}
    if RETOK_BATCH_AUDIT:
        retok_batch_report = retok_batch_signal_audit(task, mapping, emissions)

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
        "retok_cem_search": cem_report,
        "oracle_audit": oracle_report,
        "retokenized_objective_audit": retok_report,
        "retokenized_batch_audit": retok_batch_report,
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
