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
SEGMENTAL_TRANSDUCER_REPAIR = os.environ.get("DETOK_SEGMENTAL_REPAIR", "0") == "1"
SEGMENTAL_MAX_TOKENS = int(os.environ.get("DETOK_SEGMENTAL_MAX_TOKENS", "100000"))
SEGMENTAL_MAX_ISLANDS = int(os.environ.get("DETOK_SEGMENTAL_MAX_ISLANDS", "512"))
SEGMENTAL_ISLAND_WINDOW = int(os.environ.get("DETOK_SEGMENTAL_ISLAND_WINDOW", "8"))
SEGMENTAL_CONTEXT = int(os.environ.get("DETOK_SEGMENTAL_CONTEXT", "8"))
SEGMENTAL_MAX_ISLAND_LEN = int(os.environ.get("DETOK_SEGMENTAL_MAX_ISLAND_LEN", "12"))
SEGMENTAL_CANDIDATES_1 = int(os.environ.get("DETOK_SEGMENTAL_CANDIDATES_1", "12"))
SEGMENTAL_CANDIDATES_2 = int(os.environ.get("DETOK_SEGMENTAL_CANDIDATES_2", "16"))
SEGMENTAL_BEAM = int(os.environ.get("DETOK_SEGMENTAL_BEAM", "128"))
SEGMENTAL_LM_WEIGHT = float(os.environ.get("DETOK_SEGMENTAL_LM_WEIGHT", "0.035"))
SEGMENTAL_ALT_PENALTY = float(os.environ.get("DETOK_SEGMENTAL_ALT_PENALTY", "0.35"))
SEGMENTAL_GRAPH_WEIGHT = float(os.environ.get("DETOK_SEGMENTAL_GRAPH_WEIGHT", "0.25"))
SEGMENTAL_SPAN2_PENALTY = float(os.environ.get("DETOK_SEGMENTAL_SPAN2_PENALTY", "1.75"))
SEGMENTAL_MIN_GAIN = float(os.environ.get("DETOK_SEGMENTAL_MIN_GAIN", "1.5"))
SEGMENTAL_MAX_LENGTH_DELTA = int(os.environ.get("DETOK_SEGMENTAL_MAX_LENGTH_DELTA", "16"))
SEGMENTAL_ORACLE_LATTICE = os.environ.get("DETOK_SEGMENTAL_ORACLE_LATTICE", "0") == "1"

LAST_FINAL_EDGES: list[tuple[float, int, int]] = []
SEGMENTAL_LAST_DIAGNOSTICS: dict[str, float | int] = {}


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
    global LAST_FINAL_EDGES
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


def byte_lm_bits(lm, text: str) -> float:
    data = text.encode("utf-8", errors="replace")
    if not data:
        return 0.0
    return float(lm.bits_per_byte(data) * len(data))


def lm_start_history(lm, prefix: str) -> bytes:
    data = prefix.encode("utf-8", errors="replace")
    padded = bytes([0]) * (lm.order - 1) + data
    return padded[-(lm.order - 1) :] if lm.order > 1 else b""


def lm_extend_bits(lm, history: bytes, data: bytes) -> tuple[float, bytes]:
    hist = bytearray(history)
    bits = 0.0
    for nxt in data:
        max_order = min(lm.order - 1, len(hist))
        for n in range(max_order, -1, -1):
            ctx = tuple(hist[-n:]) if n else ()
            total = lm.context_counts[n].get(ctx, 0)
            if total:
                count = lm.next_counts[n].get(ctx, {}).get(int(nxt), 0)
                bits -= math.log2((count + lm.alpha) / (total + lm.alpha * 256))
                break
        else:
            bits += 8.0
        hist.append(int(nxt))
        if lm.order > 1 and len(hist) > lm.order - 1:
            del hist[: len(hist) - (lm.order - 1)]
    return bits, bytes(hist)


def token_piece(adapter, token_id: int, cache: dict[int, str]) -> str:
    token_id = int(token_id)
    piece = cache.get(token_id)
    if piece is None:
        piece = adapter.decode([token_id])
        cache[token_id] = piece
    return piece


def emission_piece(
    cipher_id: int,
    mapping: np.ndarray,
    emissions: dict[int, tuple[int, ...]],
    target_adapter,
    piece_cache: dict[int, str],
) -> str:
    emission = emissions.get(int(cipher_id))
    if emission is not None:
        return target_adapter.decode([int(p) for p in emission])
    return token_piece(target_adapter, int(mapping[int(cipher_id)]), piece_cache)


def plausible_piece(piece: str) -> bool:
    if not piece or "\ufffd" in piece:
        return False
    return len(piece.encode("utf-8", errors="replace")) <= 48


def build_segmental_edge_index(
    edges: list[tuple[float, int, int]],
    target_adapter,
    max_candidates: int,
) -> dict[int, list[tuple[float, int, str]]]:
    piece_cache: dict[int, str] = {}
    by_c: dict[int, list[tuple[float, int, str]]] = {}
    seen: dict[int, set[str]] = {}
    for score, c, p in edges:
        bucket = by_c.setdefault(int(c), [])
        if len(bucket) >= max_candidates:
            continue
        piece = token_piece(target_adapter, int(p), piece_cache)
        if not plausible_piece(piece):
            continue
        seen_bucket = seen.setdefault(int(c), set())
        if piece in seen_bucket:
            continue
        seen_bucket.add(piece)
        bucket.append((float(score), int(p), piece))
    return by_c


def segmental_singleton_candidates(
    c: int,
    mapping: np.ndarray,
    emissions: dict[int, tuple[int, ...]],
    edge_index: dict[int, list[tuple[float, int, str]]],
    target_adapter,
    piece_cache: dict[int, str],
    oracle_piece: str | None = None,
) -> list[tuple[str, float, str]]:
    current = emission_piece(c, mapping, emissions, target_adapter, piece_cache)
    candidates: list[tuple[str, float, str]] = [(current, 0.0, "current")]
    seen = {current}
    if oracle_piece is not None and oracle_piece not in seen and plausible_piece(oracle_piece):
        candidates.append((oracle_piece, -SEGMENTAL_ALT_PENALTY, "oracle"))
        seen.add(oracle_piece)
    edge_bucket = edge_index.get(int(c), [])
    top_score = edge_bucket[0][0] if edge_bucket else 0.0
    for rank, (score, _p, piece) in enumerate(edge_bucket[:SEGMENTAL_CANDIDATES_1]):
        if piece in seen or not plausible_piece(piece):
            continue
        prior = -SEGMENTAL_ALT_PENALTY * (rank + 1) + SEGMENTAL_GRAPH_WEIGHT * (score - top_score)
        candidates.append((piece, float(prior), "edge"))
        seen.add(piece)
    return candidates[:SEGMENTAL_CANDIDATES_1]


def segmental_pair_candidates(
    left: int,
    right: int,
    left_cands: list[tuple[str, float, str]],
    right_cands: list[tuple[str, float, str]],
    target_adapter,
) -> list[tuple[str, float, str]]:
    candidates: list[tuple[str, float, str]] = []
    seen: set[str] = set()

    def add(piece: str, prior: float, source: str) -> None:
        if piece in seen or not plausible_piece(piece):
            return
        seen.add(piece)
        candidates.append((piece, prior, source))

    add(left_cands[0][0] + right_cands[0][0], -SEGMENTAL_SPAN2_PENALTY, "pair_current")
    for i, (left_piece, left_prior, _) in enumerate(left_cands[:4]):
        for j, (right_piece, right_prior, _) in enumerate(right_cands[:4]):
            add(
                left_piece + right_piece,
                -SEGMENTAL_SPAN2_PENALTY + 0.5 * (left_prior + right_prior) - 0.05 * (i + j),
                "pair_concat",
            )
    base = left_cands[0][0] + right_cands[0][0]
    encoded = target_adapter.encode(base)
    if 0 < len(encoded) <= 3:
        normalized = target_adapter.decode(encoded)
        add(normalized, -SEGMENTAL_SPAN2_PENALTY - 0.15, "pair_retokenized")
    return candidates[:SEGMENTAL_CANDIDATES_2]


def inverse_perm(perm: np.ndarray) -> np.ndarray:
    inv = np.full(int(perm.max(initial=0)) + 1, -1, dtype=np.int64)
    inv[perm.astype(np.int64, copy=False)] = np.arange(len(perm), dtype=np.int64)
    return inv


def oracle_single_piece(task, cipher_id: int, inv: np.ndarray, cache: dict[int, str | None]) -> str | None:
    cipher_id = int(cipher_id)
    if cipher_id in cache:
        return cache[cipher_id]
    if cipher_id >= len(inv) or int(inv[cipher_id]) < 0:
        cache[cipher_id] = None
        return None
    source_id = int(inv[cipher_id])
    source_piece = task.source_adapter.decode([source_id])
    target_ids = task.target_adapter.encode(source_piece)
    if len(target_ids) != 1:
        cache[cipher_id] = None
        return None
    piece = task.target_adapter.decode(target_ids)
    cache[cipher_id] = piece
    return piece


def choose_segmental_islands(
    cipher_ids: np.ndarray,
    pieces: list[str],
    edge_index: dict[int, list[tuple[float, int, str]]],
    lm,
) -> list[tuple[int, int]]:
    n = len(cipher_ids)
    if n == 0:
        return []
    width = min(SEGMENTAL_ISLAND_WINDOW, SEGMENTAL_MAX_ISLAND_LEN, n)
    step = max(1, width // 2)
    scored: list[tuple[float, int, int]] = []
    for start in range(0, max(1, n - width + 1), step):
        end = min(n, start + width)
        uncertainty = 0.0
        for c in cipher_ids[start:end]:
            bucket = edge_index.get(int(c), [])
            if len(bucket) >= 2:
                margin = max(0.0, bucket[0][0] - bucket[1][0])
                uncertainty += 1.0 / (1.0 + 25.0 * margin)
            elif bucket:
                uncertainty += 0.15
        text = "".join(pieces[start:end])
        local_bpb = lm.bits_per_byte(text.encode("utf-8", errors="replace")) if text else 0.0
        score = uncertainty + 0.20 * float(local_bpb)
        scored.append((score, start, end))
    scored.sort(reverse=True)
    islands: list[tuple[int, int]] = []
    occupied = np.zeros(n, dtype=bool)
    for score, start, end in scored:
        if len(islands) >= SEGMENTAL_MAX_ISLANDS:
            break
        if score <= 0.0 or bool(occupied[max(0, start - 1) : min(n, end + 1)].any()):
            continue
        islands.append((start, end))
        occupied[start:end] = True
    islands.sort()
    return islands


def run_segmental_viterbi(
    cipher_slice: np.ndarray,
    prefix: str,
    suffix: str,
    old_mid: str,
    singleton_by_pos: list[list[tuple[str, float, str]]],
    pair_by_pos: dict[int, list[tuple[str, float, str]]],
    lm,
) -> tuple[str, float, dict[str, float | int]]:
    start_hist = lm_start_history(lm, prefix)
    old_mid_bits, old_hist = lm_extend_bits(lm, start_hist, old_mid.encode("utf-8", errors="replace"))
    old_suffix_bits, _ = lm_extend_bits(lm, old_hist, suffix.encode("utf-8", errors="replace"))
    old_lm_bits = old_mid_bits + old_suffix_bits
    old_score = -SEGMENTAL_LM_WEIGHT * old_lm_bits
    beams: dict[int, list[tuple[float, str, int, float, bytes, float]]] = {
        0: [(0.0, "", 0, 0.0, start_hist, 0.0)]
    }
    n = len(cipher_slice)
    for pos in range(n):
        states = beams.get(pos)
        if not states:
            continue
        for _approx_score, text, span2_count, channel_score, hist, lm_bits_so_far in states:
            for piece, prior, _source in singleton_by_pos[pos]:
                piece_bytes = piece.encode("utf-8", errors="replace")
                piece_bits, next_hist = lm_extend_bits(lm, hist, piece_bytes)
                new_text = text + piece
                new_channel = channel_score + prior
                new_lm_bits = lm_bits_so_far + piece_bits
                approx = new_channel - SEGMENTAL_LM_WEIGHT * new_lm_bits
                beams.setdefault(pos + 1, []).append(
                    (approx, new_text, span2_count, new_channel, next_hist, new_lm_bits)
                )
            if pos + 1 < n:
                for piece, prior, _source in pair_by_pos.get(pos, []):
                    piece_bytes = piece.encode("utf-8", errors="replace")
                    piece_bits, next_hist = lm_extend_bits(lm, hist, piece_bytes)
                    new_text = text + piece
                    new_channel = channel_score + prior
                    new_lm_bits = lm_bits_so_far + piece_bits
                    approx = new_channel - SEGMENTAL_LM_WEIGHT * new_lm_bits
                    beams.setdefault(pos + 2, []).append(
                        (approx, new_text, span2_count + 1, new_channel, next_hist, new_lm_bits)
                    )
        for key in (pos + 1, pos + 2):
            if key in beams and len(beams[key]) > SEGMENTAL_BEAM:
                beams[key].sort(key=lambda item: item[0], reverse=True)
                beams[key] = beams[key][:SEGMENTAL_BEAM]
    finals = beams.get(n, [])
    if not finals:
        return old_mid, 0.0, {"span2": 0, "channel_delta": 0.0, "lm_delta": 0.0}
    best_text = old_mid
    best_score = old_score
    best_span2 = 0
    best_channel = 0.0
    best_lm_delta = 0.0
    suffix_bytes = suffix.encode("utf-8", errors="replace")
    for _approx, text, span2_count, channel_score, hist, lm_bits_so_far in finals:
        suffix_bits, _ = lm_extend_bits(lm, hist, suffix_bytes)
        lm_bits = lm_bits_so_far + suffix_bits
        total = channel_score - SEGMENTAL_LM_WEIGHT * lm_bits
        if total > best_score:
            best_score = total
            best_text = text
            best_span2 = span2_count
            best_channel = channel_score
            best_lm_delta = old_lm_bits - lm_bits
    return (
        best_text,
        best_score - old_score,
        {"span2": best_span2, "channel_delta": best_channel, "lm_delta": best_lm_delta},
    )


def segmental_island_repair(
    task,
    cipher_ids: np.ndarray,
    mapping: np.ndarray,
    emissions: dict[int, tuple[int, ...]],
) -> str:
    global SEGMENTAL_LAST_DIAGNOSTICS
    SEGMENTAL_LAST_DIAGNOSTICS = {}
    if (
        not SEGMENTAL_TRANSDUCER_REPAIR
        or len(cipher_ids) > SEGMENTAL_MAX_TOKENS
        or not LAST_FINAL_EDGES
    ):
        return decode_with_variable_emissions(cipher_ids, mapping, emissions, task.target_adapter)

    piece_cache: dict[int, str] = {}
    pieces = [emission_piece(int(c), mapping, emissions, task.target_adapter, piece_cache) for c in cipher_ids]
    edge_index = build_segmental_edge_index(LAST_FINAL_EDGES, task.target_adapter, SEGMENTAL_CANDIDATES_1)
    islands = choose_segmental_islands(cipher_ids, pieces, edge_index, task.byte_lm)
    if not islands:
        return "".join(pieces)

    inv = inverse_perm(task.perm)
    oracle_cache: dict[int, str | None] = {}
    singleton_cache: dict[int, list[tuple[str, float, str]]] = {}

    def singleton(c: int) -> list[tuple[str, float, str]]:
        c = int(c)
        cached = singleton_cache.get(c)
        if cached is not None:
            return cached
        oracle = oracle_single_piece(task, c, inv, oracle_cache) if SEGMENTAL_ORACLE_LATTICE else None
        cached = segmental_singleton_candidates(
            c,
            mapping,
            emissions,
            edge_index,
            task.target_adapter,
            piece_cache,
            oracle_piece=oracle,
        )
        singleton_cache[c] = cached
        return cached

    oracle_total = 0
    top1 = top4 = top12 = 0
    for start, end in islands:
        for c in cipher_ids[start:end]:
            oracle = oracle_single_piece(task, int(c), inv, oracle_cache)
            if oracle is None:
                continue
            oracle_total += 1
            cand = [piece for piece, _prior, _source in singleton(int(c))]
            top1 += int(bool(cand[:1]) and cand[0] == oracle)
            top4 += int(oracle in cand[:4])
            top12 += int(oracle in cand[:12])

    repaired: list[str] = []
    cursor = 0
    accepted = 0
    span2_used = 0
    gains: list[float] = []
    lm_gains: list[float] = []
    channel_deltas: list[float] = []
    for start, end in islands:
        repaired.extend(pieces[cursor:start])
        prefix_start = max(0, start - SEGMENTAL_CONTEXT)
        suffix_end = min(len(cipher_ids), end + SEGMENTAL_CONTEXT)
        prefix = "".join(pieces[prefix_start:start])
        suffix = "".join(pieces[end:suffix_end])
        old_mid = "".join(pieces[start:end])
        singleton_by_pos = [singleton(int(c)) for c in cipher_ids[start:end]]
        pair_by_pos: dict[int, list[tuple[str, float, str]]] = {}
        for rel in range(0, end - start - 1):
            pair_by_pos[rel] = segmental_pair_candidates(
                int(cipher_ids[start + rel]),
                int(cipher_ids[start + rel + 1]),
                singleton_by_pos[rel],
                singleton_by_pos[rel + 1],
                task.target_adapter,
            )
        new_mid, gain, stats = run_segmental_viterbi(
            cipher_ids[start:end],
            prefix,
            suffix,
            old_mid,
            singleton_by_pos,
            pair_by_pos,
            task.byte_lm,
        )
        length_delta = abs(len(new_mid.encode("utf-8", errors="replace")) - len(old_mid.encode("utf-8", errors="replace")))
        if new_mid != old_mid and gain >= SEGMENTAL_MIN_GAIN and length_delta <= SEGMENTAL_MAX_LENGTH_DELTA:
            repaired.append(new_mid)
            accepted += 1
            span2_used += int(stats["span2"])
            gains.append(float(gain))
            lm_gains.append(float(stats["lm_delta"]))
            channel_deltas.append(float(stats["channel_delta"]))
        else:
            repaired.append(old_mid)
        cursor = end
    repaired.extend(pieces[cursor:])

    diagnostics: dict[str, float | int] = {
        "segmental_islands": len(islands),
        "segmental_accepted": accepted,
        "segmental_span2_used": span2_used,
        "segmental_oracle_singletons": oracle_total,
        "segmental_c1_top1_recall": (top1 / oracle_total) if oracle_total else 0.0,
        "segmental_c1_top4_recall": (top4 / oracle_total) if oracle_total else 0.0,
        "segmental_c1_top12_recall": (top12 / oracle_total) if oracle_total else 0.0,
    }
    if gains:
        gains_arr = np.asarray(gains, dtype=np.float32)
        diagnostics["segmental_gain_median"] = float(np.median(gains_arr))
        diagnostics["segmental_gain_p90"] = float(np.percentile(gains_arr, 90))
        diagnostics["segmental_lm_delta_mean"] = float(np.mean(lm_gains))
        diagnostics["segmental_channel_delta_mean"] = float(np.mean(channel_deltas))
    SEGMENTAL_LAST_DIAGNOSTICS = diagnostics
    print(f"segmental_islands={len(islands)} accepted={accepted} span2_used={span2_used}", flush=True)
    print(
        "segmental_c1_recall="
        f"top1:{diagnostics['segmental_c1_top1_recall']:.4f} "
        f"top4:{diagnostics['segmental_c1_top4_recall']:.4f} "
        f"top12:{diagnostics['segmental_c1_top12_recall']:.4f} "
        f"oracle_n:{oracle_total}",
        flush=True,
    )
    if gains:
        print(
            f"segmental_gain_median={diagnostics['segmental_gain_median']:.6f} "
            f"p90={diagnostics['segmental_gain_p90']:.6f}",
            flush=True,
        )
    return "".join(repaired)


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
    sample_cipher = task.cipher_ids[:SAMPLE_TOKENS]
    recovered_sample = segmental_island_repair(
        task,
        sample_cipher,
        mapping,
        emissions,
    )
    if not recovered_sample:
        recovered_sample = decode_with_variable_emissions(
            sample_cipher,
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
        "segmental_transducer_repair": SEGMENTAL_TRANSDUCER_REPAIR,
        "segmental_diagnostics": SEGMENTAL_LAST_DIAGNOSTICS,
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
