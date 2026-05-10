"""Mutable detokenizer experiment.

This file is the hillclimb target. The baseline implements the current
frequency + bigram-context graph aligner for a shuffled token-ID stream. Agents
should modify this file only, run `uv run train.py`, and keep changes that lower
cer50k.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
import os
import re
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
ALGEBRAIC_PSEUDOWORD = os.environ.get("DETOK_ALGEBRAIC_PSEUDOWORD", "0") == "1"
ALGEBRAIC_START_TOKENS = int(os.environ.get("DETOK_ALGEBRAIC_START_TOKENS", "512"))
ALGEBRAIC_MAX_PSEUDOWORDS = int(os.environ.get("DETOK_ALGEBRAIC_MAX_PSEUDOWORDS", "2000"))
ALGEBRAIC_MAX_WORD_TOKENS = int(os.environ.get("DETOK_ALGEBRAIC_MAX_WORD_TOKENS", "6"))
ALGEBRAIC_MAX_SURFACE_LEN = int(os.environ.get("DETOK_ALGEBRAIC_MAX_SURFACE_LEN", "24"))
ALGEBRAIC_REF_CHARS = int(os.environ.get("DETOK_ALGEBRAIC_REF_CHARS", "5000000"))
ALGEBRAIC_MAX_SOLVED_APPLY = int(os.environ.get("DETOK_ALGEBRAIC_MAX_SOLVED_APPLY", "4096"))
ALGEBRAIC_MIN_GROUP = int(os.environ.get("DETOK_ALGEBRAIC_MIN_GROUP", "2"))
ALGEBRAIC_START_SCORE = os.environ.get("DETOK_ALGEBRAIC_START_SCORE", "right-left")

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


def align_shuffled(cipher_ids: np.ndarray, ref_ids: np.ndarray, target_vocab_size: int) -> np.ndarray:
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


def entropy(bucket: dict[int, int]) -> float:
    total = float(sum(bucket.values()))
    if total <= 0.0:
        return 0.0
    return float(-sum((count / total) * math.log(count / total) for count in bucket.values()))


def inverse_permutation(perm: np.ndarray) -> np.ndarray:
    inv = np.empty(len(perm), dtype=np.int64)
    inv[np.asarray(perm, dtype=np.int64)] = np.arange(len(perm), dtype=np.int64)
    return inv


def token_piece(adapter, token_id: int, cache: dict[int, str]) -> str:
    token_id = int(token_id)
    if token_id not in cache:
        cache[token_id] = adapter.decode([token_id])
    return cache[token_id]


def likely_word_start_tokens(ids: np.ndarray, topn: int) -> tuple[set[int], dict[str, float]]:
    arr = np.asarray(ids, dtype=np.int64)
    token_counts = np.bincount(arr, minlength=int(arr.max(initial=0)) + 1)
    left: dict[int, dict[int, int]] = defaultdict(dict)
    right: dict[int, dict[int, int]] = defaultdict(dict)
    for pos, token_raw in enumerate(arr):
        token = int(token_raw)
        if pos > 0:
            prev = int(arr[pos - 1])
            left[token][prev] = left[token].get(prev, 0) + 1
        if pos + 1 < len(arr):
            nxt = int(arr[pos + 1])
            right[token][nxt] = right[token].get(nxt, 0) + 1
    rows: list[tuple[float, int, float, float, int]] = []
    for token in np.flatnonzero(token_counts >= 4):
        l_ent = entropy(left.get(int(token), {}))
        r_ent = entropy(right.get(int(token), {}))
        freq_bonus = 0.05 * math.log1p(int(token_counts[token]))
        if ALGEBRAIC_START_SCORE == "left-right":
            score = l_ent - r_ent + freq_bonus
        elif ALGEBRAIC_START_SCORE == "left":
            score = l_ent + freq_bonus
        elif ALGEBRAIC_START_SCORE == "right":
            score = r_ent + freq_bonus
        elif ALGEBRAIC_START_SCORE == "freq":
            score = math.log1p(int(token_counts[token]))
        else:
            score = r_ent - l_ent + freq_bonus
        rows.append((score, int(token), l_ent, r_ent, int(token_counts[token])))
    rows.sort(reverse=True)
    starts = {token for _, token, _, _, _ in rows[:topn]}
    if rows:
        best_score, best_token, best_left, best_right, best_count = rows[0]
        report = {
            "best_score": best_score,
            "best_token": float(best_token),
            "best_left_entropy": best_left,
            "best_right_entropy": best_right,
            "best_count": float(best_count),
        }
    else:
        report = {"best_score": 0.0, "best_token": -1.0, "best_left_entropy": 0.0, "best_right_entropy": 0.0, "best_count": 0.0}
    return starts, report


def pseudo_word_counts(ids: np.ndarray, starts: set[int]) -> Counter[tuple[int, ...]]:
    counts_by_word: Counter[tuple[int, ...]] = Counter()
    current: list[int] = []
    for raw in ids:
        token = int(raw)
        if current and token in starts:
            if len(current) <= ALGEBRAIC_MAX_WORD_TOKENS:
                counts_by_word[tuple(current)] += 1
            current = [token]
        else:
            current.append(token)
            if len(current) > ALGEBRAIC_MAX_WORD_TOKENS:
                current = [token] if token in starts else []
    if current and len(current) <= ALGEBRAIC_MAX_WORD_TOKENS:
        counts_by_word[tuple(current)] += 1
    return counts_by_word


def reference_surfaces(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="ignore")[:ALGEBRAIC_REF_CHARS]
    # Keep a leading blank on word-like surfaces so equations can solve
    # space-attached subwords without a separate word-boundary variable.
    matches = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\w\s]", text)
    counts_by_surface: Counter[str] = Counter()
    for match in matches:
        if len(match) > ALGEBRAIC_MAX_SURFACE_LEN:
            continue
        if re.match(r"[A-Za-z0-9]", match):
            surface = " " + match.lower()
        else:
            surface = match
        counts_by_surface[surface] += 1
    return [surface for surface, _ in counts_by_surface.most_common(ALGEBRAIC_MAX_PSEUDOWORDS)]


def longest_common_prefix(strings: list[str]) -> str:
    if not strings:
        return ""
    prefix = strings[0]
    for value in strings[1:]:
        limit = min(len(prefix), len(value))
        idx = 0
        while idx < limit and prefix[idx] == value[idx]:
            idx += 1
        prefix = prefix[:idx]
        if not prefix:
            break
    return prefix


def longest_common_suffix(strings: list[str]) -> str:
    reversed_suffix = longest_common_prefix([value[::-1] for value in strings])
    return reversed_suffix[::-1]


def solve_word_equations(equations: list[tuple[tuple[int, ...], str]]) -> dict[int, str]:
    solved: dict[int, str] = {}
    by_first: dict[int, list[str]] = defaultdict(list)
    by_last: dict[int, list[str]] = defaultdict(list)
    for tokens, surface in equations:
        if not tokens:
            continue
        by_first[int(tokens[0])].append(surface)
        by_last[int(tokens[-1])].append(surface)
        if len(tokens) == 1 and int(tokens[0]) not in solved:
            solved[int(tokens[0])] = surface

    for token, surfaces in by_first.items():
        if token in solved or len(surfaces) < ALGEBRAIC_MIN_GROUP:
            continue
        prefix = longest_common_prefix(surfaces)
        if prefix and len(prefix) <= 12:
            solved[token] = prefix
    for token, surfaces in by_last.items():
        if token in solved or len(surfaces) < ALGEBRAIC_MIN_GROUP:
            continue
        suffix = longest_common_suffix(surfaces)
        if suffix and len(suffix) <= 12:
            solved[token] = suffix

    for _ in range(12):
        changed = 0
        for tokens, surface in equations:
            remaining = surface
            open_tokens = list(map(int, tokens))
            while open_tokens and open_tokens[0] in solved and remaining.startswith(solved[open_tokens[0]]):
                remaining = remaining[len(solved[open_tokens[0]]) :]
                open_tokens.pop(0)
            while open_tokens and open_tokens[-1] in solved and remaining.endswith(solved[open_tokens[-1]]):
                remaining = remaining[: -len(solved[open_tokens[-1]])] if solved[open_tokens[-1]] else remaining
                open_tokens.pop()
            if len(open_tokens) == 1 and remaining and len(remaining) <= 16:
                token = open_tokens[0]
                old = solved.get(token)
                if old is None:
                    solved[token] = remaining
                    changed += 1
                elif old != remaining:
                    # Conflicting rank cribs are common; drop the token instead
                    # of letting one bad equation poison the decode.
                    solved.pop(token, None)
        if changed == 0:
            break
    return solved


def algebraic_pseudoword_decode(task, mapping: np.ndarray, baseline_text: str, baseline_metrics: dict[str, float]) -> dict[str, object]:
    print("algebraic_pseudoword=1", flush=True)
    sample = np.asarray(task.cipher_ids[:SAMPLE_TOKENS], dtype=np.int64)
    starts, entropy_report = likely_word_start_tokens(sample, ALGEBRAIC_START_TOKENS)
    pseudo_counts = pseudo_word_counts(sample, starts)
    pseudo_words = [word for word, _ in pseudo_counts.most_common(ALGEBRAIC_MAX_PSEUDOWORDS)]
    surfaces = reference_surfaces(task.reference_text)
    equations: list[tuple[tuple[int, ...], str]] = []
    for pseudo, surface in zip(pseudo_words, surfaces):
        if len(pseudo) <= ALGEBRAIC_MAX_WORD_TOKENS and len(surface) <= ALGEBRAIC_MAX_SURFACE_LEN:
            equations.append((pseudo, surface))

    solved = solve_word_equations(equations)
    inv_perm = inverse_permutation(task.perm)
    exact = 0
    token_mass = 0
    source_piece_cache: dict[int, str] = {}
    for token, piece in solved.items():
        count = int(np.count_nonzero(sample == token))
        token_mass += count
        if 0 <= token < len(inv_perm):
            true_piece = token_piece(task.source_adapter, int(inv_perm[token]), source_piece_cache)
            if piece == true_piece:
                exact += 1
    exact_rate = exact / max(1, len(solved))
    mass_rate = token_mass / max(1, len(sample))

    target_piece_cache: dict[int, str] = {}
    apply_items = sorted(solved.items(), key=lambda item: -int(np.count_nonzero(sample == item[0])))[:ALGEBRAIC_MAX_SOLVED_APPLY]
    applied = dict(apply_items)
    pieces: list[str] = []
    for raw in sample:
        token = int(raw)
        if token in applied:
            pieces.append(applied[token])
        else:
            pieces.append(token_piece(task.target_adapter, int(mapping[token]), target_piece_cache))
    recovered = "".join(pieces)
    metrics = evaluate_recovery(task, recovered, SAMPLE_TOKENS)

    # Diagnostic upper bound for this equation solver: apply only exact solved
    # pieces. This uses oracle labels only to judge whether the equations found
    # enough useful material to justify a safer selector later.
    oracle_good = {}
    for token, piece in applied.items():
        if 0 <= token < len(inv_perm):
            true_piece = token_piece(task.source_adapter, int(inv_perm[token]), source_piece_cache)
            if piece == true_piece:
                oracle_good[token] = piece
    oracle_pieces = []
    for raw in sample:
        token = int(raw)
        if token in oracle_good:
            oracle_pieces.append(oracle_good[token])
        else:
            oracle_pieces.append(token_piece(task.target_adapter, int(mapping[token]), target_piece_cache))
    oracle_metrics = evaluate_recovery(task, "".join(oracle_pieces), SAMPLE_TOKENS)

    report = {
        "start_tokens": len(starts),
        "pseudo_word_types": len(pseudo_counts),
        "equations": len(equations),
        "solved_tokens": len(solved),
        "applied_tokens": len(applied),
        "exact_solved_tokens": exact,
        "exact_solved_rate": exact_rate,
        "solved_token_mass": mass_rate,
        "baseline_cer50k": float(baseline_metrics["cer50k"]),
        "algebraic_cer50k": float(metrics["cer50k"]),
        "algebraic_bpb": float(metrics["byte_lm_bpb"]),
        "oracle_good_cer50k": float(oracle_metrics["cer50k"]),
        "oracle_good_bpb": float(oracle_metrics["byte_lm_bpb"]),
        "start_score": ALGEBRAIC_START_SCORE,
        **entropy_report,
    }
    print(
        "algebraic_pseudoword_report "
        + " ".join(f"{key}={value:.6f}" if isinstance(value, float) else f"{key}={value}" for key, value in report.items()),
        flush=True,
    )
    return {"report": report, "preview": recovered[:1000]}


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
    algebraic_report = (
        algebraic_pseudoword_decode(task, mapping, recovered_sample, metrics) if ALGEBRAIC_PSEUDOWORD else None
    )

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
        "elapsed_seconds": time.time() - t0,
        "metrics": metrics,
        "preview": recovered_sample[:1000],
    }
    if algebraic_report is not None:
        report["algebraic_pseudoword"] = algebraic_report
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
