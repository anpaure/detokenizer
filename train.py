"""Mutable detokenizer experiment.

This file is the hillclimb target. The baseline implements the current
frequency + bigram-context graph aligner for a shuffled token-ID stream. Agents
should modify this file only, run `uv run train.py`, and keep changes that lower
cer50k.
"""

from __future__ import annotations

import json
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
TAIL_REPAIR_NODES = 1_024
TAIL_REPAIR_CONTEXTS = 8_192
TAIL_REPAIR_CANDIDATES = 8
TAIL_REPAIR_MIN_GAIN_PER_OCC = 0.15
EXTERNAL_OWNER_REPAIR = True
EXTERNAL_OWNER_MAX_TOKENS = 100_000
EXTERNAL_OWNER_NODES = 2_048
EXTERNAL_OWNER_CONTEXTS = 8_192
EXTERNAL_OWNER_CANDIDATES = 5
EXTERNAL_OWNER_MIN_COUNT = 5
EXTERNAL_OWNER_MIN_GAIN_PER_OCC = 0.35
EXTERNAL_OWNER_PROTECT_TOP = 512


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
    protected = set(map(int, c_focus[: min(len(c_focus), EXTERNAL_OWNER_PROTECT_TOP)]))
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
        if c in protected or owner in protected:
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
    mapped_sample = mapping[task.cipher_ids[:SAMPLE_TOKENS]]
    recovered_sample = task.target_adapter.decode(mapped_sample.tolist())
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
