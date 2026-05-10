"""Mutable detokenizer experiment.

This file is the hillclimb target. The baseline implements the current
frequency + bigram-context graph aligner for a shuffled token-ID stream. Agents
should modify this file only, run `uv run train.py`, and keep changes that lower
cer50k.
"""

from __future__ import annotations

from collections import defaultdict
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
TOKENIZER_PRIOR_AUDIT = os.environ.get("DETOK_TOKENIZER_PRIOR_AUDIT", "0") == "1"
TOKENIZER_PRIOR_TYPES = int(os.environ.get("DETOK_TOKENIZER_PRIOR_TYPES", "8192"))
TOKENIZER_PRIOR_CHARS = int(os.environ.get("DETOK_TOKENIZER_PRIOR_CHARS", "50000"))
TOKENIZER_PRIOR_REF_CHARS = int(os.environ.get("DETOK_TOKENIZER_PRIOR_REF_CHARS", "1000000"))
TOKENIZER_PRIOR_BPE_VOCAB = int(os.environ.get("DETOK_TOKENIZER_PRIOR_BPE_VOCAB", "16000"))
TOKENIZER_PRIOR_MOVE_TYPES = int(os.environ.get("DETOK_TOKENIZER_PRIOR_MOVE_TYPES", "512"))
TOKENIZER_PRIOR_MOVE_TOPK = int(os.environ.get("DETOK_TOKENIZER_PRIOR_MOVE_TOPK", "16"))
TOKENIZER_PRIOR_MOVE_WINDOWS = int(os.environ.get("DETOK_TOKENIZER_PRIOR_MOVE_WINDOWS", "16"))
TOKENIZER_PRIOR_WINDOW_RADIUS = int(os.environ.get("DETOK_TOKENIZER_PRIOR_WINDOW_RADIUS", "12"))
RECURRENCE_PROPOSAL_AUDIT = os.environ.get("DETOK_RECURRENCE_PROPOSAL_AUDIT", "0") == "1"
RECURRENCE_TYPES = int(os.environ.get("DETOK_RECURRENCE_TYPES", "512"))
RECURRENCE_TOPK = int(os.environ.get("DETOK_RECURRENCE_TOPK", "16"))
RECURRENCE_STAT_TOKENS = int(os.environ.get("DETOK_RECURRENCE_STAT_TOKENS", "500000"))
RECURRENCE_SYNTH_CHARS = int(os.environ.get("DETOK_RECURRENCE_SYNTH_CHARS", "500000"))
RECURRENCE_EPOCHS = int(os.environ.get("DETOK_RECURRENCE_EPOCHS", "80"))
RECURRENCE_LR = float(os.environ.get("DETOK_RECURRENCE_LR", "0.05"))

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

LAST_FINAL_EDGES: list[tuple[float, int, int]] = []
LAST_C_FOCUS: np.ndarray = np.empty(0, dtype=np.int64)


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
    LAST_FINAL_EDGES = []

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


def inverse_permutation(perm: np.ndarray) -> np.ndarray:
    inv = np.empty(len(perm), dtype=np.int64)
    inv[np.asarray(perm, dtype=np.int64)] = np.arange(len(perm), dtype=np.int64)
    return inv


def graph_candidate_pool(edges: list[tuple[float, int, int]], topk: int = 64) -> dict[int, list[int]]:
    pool: dict[int, list[int]] = defaultdict(list)
    seen: dict[int, set[int]] = defaultdict(set)
    for _, c, p in edges:
        if len(pool[c]) >= topk or p in seen[c]:
            continue
        pool[c].append(p)
        seen[c].add(p)
    return pool


def target_piece(target_adapter, token_id: int, cache: dict[int, str]) -> str:
    token_id = int(token_id)
    if token_id not in cache:
        cache[token_id] = target_adapter.decode([token_id])
    return cache[token_id]


def singleton_truth_for_cipher(task, cipher_id: int, inv_perm: np.ndarray, cache: dict[int, tuple[int | None, str]]):
    cipher_id = int(cipher_id)
    if cipher_id in cache:
        return cache[cipher_id]
    if cipher_id < 0 or cipher_id >= len(inv_perm):
        cache[cipher_id] = (None, "")
        return cache[cipher_id]
    source_id = int(inv_perm[cipher_id])
    piece = task.source_adapter.decode([source_id])
    encoded = task.target_adapter.encode(piece)
    truth = int(encoded[0]) if len(encoded) == 1 else None
    cache[cipher_id] = (truth, piece)
    return cache[cipher_id]


def decode_pieces_with_overrides(
    cipher_ids: np.ndarray,
    mapping: np.ndarray,
    target_adapter,
    overrides: dict[int, int] | None = None,
    max_chars: int | None = None,
) -> tuple[list[str], str]:
    overrides = overrides or {}
    cache: dict[int, str] = {}
    pieces: list[str] = []
    total_chars = 0
    for c_raw in cipher_ids:
        c = int(c_raw)
        p = int(overrides.get(c, int(mapping[c])))
        piece = target_piece(target_adapter, p, cache)
        pieces.append(piece)
        total_chars += len(piece)
        if max_chars is not None and total_chars >= max_chars:
            break
    return pieces, "".join(pieces)


def build_unigram_piece_model(pieces: list[str], extra_pieces: list[str] | None = None) -> dict[str, object]:
    counts_by_piece: dict[str, int] = {}
    for piece in pieces:
        if piece:
            counts_by_piece[piece] = counts_by_piece.get(piece, 0) + 1
    for piece in extra_pieces or []:
        if piece:
            counts_by_piece.setdefault(piece, 0)
    alpha = 0.1
    vocab = {piece: count for piece, count in counts_by_piece.items() if len(piece) <= 48}
    total = float(sum(vocab.values()))
    denom = total + alpha * (len(vocab) + 512)
    logp_by_piece = {piece: math.log((count + alpha) / denom) for piece, count in vocab.items()}
    starts: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for piece, logp in logp_by_piece.items():
        starts[piece[0]].append((piece, logp))
    chars = set("".join(pieces))
    for ch in chars:
        starts[ch].append((ch, math.log(alpha / denom)))
    for key, values in list(starts.items()):
        best: dict[str, float] = {}
        for piece, logp in values:
            old = best.get(piece)
            if old is None or logp > old:
                best[piece] = logp
        starts[key] = sorted(best.items(), key=lambda item: item[1], reverse=True)[:256]
    return {
        "starts": starts,
        "empty_logp": math.log(alpha / denom),
        "piece_count": len(pieces),
        "vocab_size": len(vocab),
    }


def score_unigram_pieces(pieces: list[str], model: dict[str, object]) -> dict[str, float]:
    text = "".join(pieces)
    chars = len(text)
    if not text:
        return {
            "chars": 0.0,
            "piece_count": float(len(pieces)),
            "vocab_size": float(model["vocab_size"]),
            "observed_nll": 99.0,
            "partition_nll": 99.0,
            "boundary_regret": 99.0,
        }
    starts = model["starts"]  # type: ignore[assignment]
    empty_logp = float(model["empty_logp"])
    observed_logp = 0.0
    for piece in pieces:
        if not piece:
            observed_logp += empty_logp
            continue
        candidates = dict(starts.get(piece[0], []))  # type: ignore[union-attr]
        observed_logp += float(candidates.get(piece, empty_logp * max(1, len(piece))))

    dp = np.full(chars + 1, -np.inf, dtype=np.float64)
    dp[0] = 0.0
    for pos in range(chars):
        base = float(dp[pos])
        if not np.isfinite(base):
            continue
        for piece, logp in starts.get(text[pos], []):  # type: ignore[union-attr]
            end = pos + len(piece)
            if end <= chars and text.startswith(piece, pos):
                dp[end] = np.logaddexp(dp[end], base + float(logp))
    log_z = float(dp[chars])
    return {
        "chars": float(chars),
        "piece_count": float(len(pieces)),
        "vocab_size": float(model["vocab_size"]),
        "observed_nll": -observed_logp / chars,
        "partition_nll": -log_z / chars,
        "boundary_regret": (log_z - observed_logp) / chars,
    }


def unigram_segmentation_prior(pieces: list[str]) -> dict[str, float]:
    return score_unigram_pieces(pieces, build_unigram_piece_model(pieces))


def train_proxy_bpe(reference_text: Path):
    try:
        from tokenizers import Tokenizer
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import ByteLevel
        from tokenizers.trainers import BpeTrainer
    except Exception as exc:  # pragma: no cover - optional dependency on some hosts
        print(f"proxy_bpe_unavailable={type(exc).__name__}: {exc}", flush=True)
        return None

    raw = reference_text.read_text(encoding="utf-8", errors="ignore")[:TOKENIZER_PRIOR_REF_CHARS]
    if not raw:
        return None
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(vocab_size=TOKENIZER_PRIOR_BPE_VOCAB, special_tokens=["[UNK]"])
    chunks = [raw[i : i + 20_000] for i in range(0, len(raw), 20_000)]
    tokenizer.train_from_iterator(chunks, trainer=trainer)
    return tokenizer


def proxy_bpe_boundary_score(text: str, pieces: list[str], tokenizer) -> dict[str, float]:
    if tokenizer is None or not text:
        return {"bpe_boundary_f1": 0.0, "bpe_boundary_mismatch": 1.0, "bpe_token_ratio": 0.0}
    observed: set[int] = set()
    pos = 0
    for piece in pieces:
        pos += len(piece)
        if 0 < pos < len(text):
            observed.add(pos)
    encoded = tokenizer.encode(text)
    proxy = {int(end) for _, end in encoded.offsets if 0 < int(end) < len(text)}
    denom = len(observed) + len(proxy)
    if denom == 0:
        f1 = 1.0
    else:
        f1 = 2.0 * len(observed & proxy) / denom
    return {
        "bpe_boundary_f1": f1,
        "bpe_boundary_mismatch": 1.0 - f1,
        "bpe_token_ratio": len(proxy) / max(1, len(observed)),
    }


def auc_from_scores(labels: list[int], scores: list[float]) -> float:
    if not labels or len(set(labels)) < 2:
        return 0.5
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    order = np.argsort(s)
    ranks = np.empty(len(s), dtype=np.float64)
    i = 0
    while i < len(s):
        j = i + 1
        while j < len(s) and s[order[j]] == s[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0 + 1.0
        i = j
    pos = y == 1
    n_pos = int(pos.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def top_precision(labels: list[int], scores: list[float], k: int) -> float:
    if not labels:
        return 0.0
    order = np.argsort(-np.asarray(scores, dtype=np.float64))[: min(k, len(labels))]
    return float(np.asarray(labels, dtype=np.float64)[order].mean()) if len(order) else 0.0


def tokenizer_prior_move_audit(
    task,
    mapping: np.ndarray,
    graph_pool: dict[int, list[int]],
    inv_perm: np.ndarray,
    proxy_bpe,
) -> dict[str, float]:
    sample = np.asarray(task.cipher_ids[:SAMPLE_TOKENS], dtype=np.int64)
    focus = [int(c) for c in LAST_C_FOCUS[:TOKENIZER_PRIOR_MOVE_TYPES]]
    positions_by_c: dict[int, list[int]] = defaultdict(list)
    focus_set = set(focus)
    for pos, c_raw in enumerate(sample):
        c = int(c_raw)
        if c in focus_set and len(positions_by_c[c]) < TOKENIZER_PRIOR_MOVE_WINDOWS:
            positions_by_c[c].append(pos)

    piece_cache: dict[int, str] = {}
    extra_pieces: list[str] = []
    for c in focus:
        for p in graph_pool.get(c, [])[:TOKENIZER_PRIOR_MOVE_TOPK]:
            extra_pieces.append(target_piece(task.target_adapter, p, piece_cache))
    baseline_pieces, _ = decode_pieces_with_overrides(
        sample,
        mapping,
        task.target_adapter,
        max_chars=TOKENIZER_PRIOR_CHARS,
    )
    unigram_model = build_unigram_piece_model(baseline_pieces, extra_pieces)

    truth_cache: dict[int, tuple[int | None, str]] = {}
    labels: list[int] = []
    seg_scores: list[float] = []
    bpe_scores: list[float] = []
    old_cache: dict[tuple[int, int], tuple[float, float]] = {}

    def window_pieces(center: int, overrides: dict[int, int] | None = None) -> tuple[list[str], str]:
        start = max(0, center - TOKENIZER_PRIOR_WINDOW_RADIUS)
        stop = min(len(sample), center + TOKENIZER_PRIOR_WINDOW_RADIUS + 1)
        return decode_pieces_with_overrides(sample[start:stop], mapping, task.target_adapter, overrides)

    for c in focus:
        positions = positions_by_c.get(c)
        if not positions:
            continue
        true_p, _ = singleton_truth_for_cipher(task, c, inv_perm, truth_cache)
        for p in graph_pool.get(c, [])[:TOKENIZER_PRIOR_MOVE_TOPK]:
            if p == int(mapping[c]):
                continue
            seg_delta_sum = 0.0
            bpe_delta_sum = 0.0
            used = 0
            for center in positions:
                start = max(0, center - TOKENIZER_PRIOR_WINDOW_RADIUS)
                stop = min(len(sample), center + TOKENIZER_PRIOR_WINDOW_RADIUS + 1)
                old_key = (start, stop)
                if old_key not in old_cache:
                    old_pieces, old_text = window_pieces(center)
                    old_seg = score_unigram_pieces(old_pieces, unigram_model)["boundary_regret"]
                    old_bpe = proxy_bpe_boundary_score(old_text, old_pieces, proxy_bpe)["bpe_boundary_mismatch"]
                    old_cache[old_key] = (old_seg, old_bpe)
                old_seg, old_bpe = old_cache[old_key]
                new_pieces, new_text = window_pieces(center, {c: p})
                new_seg = score_unigram_pieces(new_pieces, unigram_model)["boundary_regret"]
                new_bpe = proxy_bpe_boundary_score(new_text, new_pieces, proxy_bpe)["bpe_boundary_mismatch"]
                seg_delta_sum += old_seg - new_seg
                bpe_delta_sum += old_bpe - new_bpe
                used += 1
            if used:
                labels.append(1 if true_p is not None and int(p) == int(true_p) else 0)
                seg_scores.append(seg_delta_sum / used)
                bpe_scores.append(bpe_delta_sum / used)

    report = {
        "moves": float(len(labels)),
        "positive_rate": float(np.mean(labels)) if labels else 0.0,
        "seg_auc": auc_from_scores(labels, seg_scores),
        "seg_top100_precision": top_precision(labels, seg_scores, 100),
        "seg_top500_precision": top_precision(labels, seg_scores, 500),
        "bpe_auc": auc_from_scores(labels, bpe_scores),
        "bpe_top100_precision": top_precision(labels, bpe_scores, 100),
        "bpe_top500_precision": top_precision(labels, bpe_scores, 500),
    }
    print(
        "tokenizer_prior_move_audit "
        + " ".join(f"{key}={value:.6f}" for key, value in report.items()),
        flush=True,
    )
    return report


def tokenizer_prior_audit(task, mapping: np.ndarray, baseline_metrics: dict[str, float]) -> dict[str, object]:
    print("tokenizer_prior_audit=1", flush=True)
    inv_perm = inverse_permutation(task.perm)
    graph_pool = graph_candidate_pool(LAST_FINAL_EDGES, topk=64)
    focus = [int(c) for c in LAST_C_FOCUS[:TOKENIZER_PRIOR_TYPES]]
    truth_cache: dict[int, tuple[int | None, str]] = {}
    graph_oracle: dict[int, int] = {}
    true_singleton: dict[int, int] = {}
    for c in focus:
        true_p, _ = singleton_truth_for_cipher(task, c, inv_perm, truth_cache)
        if true_p is None:
            continue
        true_singleton[c] = true_p
        if true_p in graph_pool.get(c, []):
            graph_oracle[c] = true_p

    proxy_bpe = train_proxy_bpe(task.reference_text)
    rows: dict[str, dict[str, float]] = {}
    variants = [
        ("baseline", {}),
        ("graph_top64_oracle", graph_oracle),
        ("true_singleton_top_focus", true_singleton),
    ]
    for name, overrides in variants:
        mapped = np.asarray(
            [int(overrides.get(int(c), int(mapping[int(c)]))) for c in task.cipher_ids[:SAMPLE_TOKENS]],
            dtype=np.int64,
        )
        text = task.target_adapter.decode(mapped.tolist())
        metrics = baseline_metrics if name == "baseline" else evaluate_recovery(task, text, SAMPLE_TOKENS)
        pieces, prior_text = decode_pieces_with_overrides(
            task.cipher_ids[:SAMPLE_TOKENS],
            mapping,
            task.target_adapter,
            overrides,
            max_chars=TOKENIZER_PRIOR_CHARS,
        )
        seg = unigram_segmentation_prior(pieces)
        bpe = proxy_bpe_boundary_score(prior_text, pieces, proxy_bpe)
        row = {
            "cer50k": float(metrics["cer50k"]),
            "byte_lm_bpb": float(metrics["byte_lm_bpb"]),
            "overrides": float(len(overrides)),
            **seg,
            **bpe,
        }
        rows[name] = row
        print(
            f"tokenizer_prior_row={name} "
            f"cer={row['cer50k']:.6f} bpb={row['byte_lm_bpb']:.6f} "
            f"overrides={int(row['overrides'])} "
            f"boundary_regret={row['boundary_regret']:.6f} "
            f"bpe_mismatch={row['bpe_boundary_mismatch']:.6f}",
            flush=True,
        )

    move_report = tokenizer_prior_move_audit(task, mapping, graph_pool, inv_perm, proxy_bpe)
    return {
        "focus_types": len(focus),
        "graph_pool_types": len(graph_pool),
        "graph_oracle_overrides": len(graph_oracle),
        "true_singleton_overrides": len(true_singleton),
        "rows": rows,
        "move_audit": move_report,
    }


def entropy_from_counts(bucket: dict[int, int]) -> float:
    total = float(sum(bucket.values()))
    if total <= 0.0:
        return 0.0
    return float(-sum((count / total) * math.log(count / total) for count in bucket.values()))


def recurrence_stats(ids: np.ndarray, tokens: list[int], vocab_size: int) -> dict[int, np.ndarray]:
    arr = np.asarray(ids[:RECURRENCE_STAT_TOKENS], dtype=np.int64)
    if len(arr) == 0:
        return {int(t): np.zeros(7, dtype=np.float32) for t in tokens}
    counts_arr = np.bincount(arr, minlength=max(vocab_size, int(arr.max(initial=0)) + 1))
    order = np.argsort(-counts_arr)
    ranks = np.empty(len(order), dtype=np.int64)
    ranks[order] = np.arange(len(order), dtype=np.int64)
    wanted = set(map(int, tokens))
    last_seen: dict[int, int] = {}
    gap_sum: dict[int, float] = defaultdict(float)
    gap_count: dict[int, int] = defaultdict(int)
    left_counts: dict[int, dict[int, int]] = defaultdict(dict)
    right_counts: dict[int, dict[int, int]] = defaultdict(dict)
    for pos, token_raw in enumerate(arr):
        token = int(token_raw)
        if token not in wanted:
            continue
        prev = last_seen.get(token)
        if prev is not None:
            gap_sum[token] += math.log1p(pos - prev)
            gap_count[token] += 1
        last_seen[token] = pos
        if pos > 0:
            left = int(arr[pos - 1])
            bucket = left_counts[token]
            bucket[left] = bucket.get(left, 0) + 1
        if pos + 1 < len(arr):
            right = int(arr[pos + 1])
            bucket = right_counts[token]
            bucket[right] = bucket.get(right, 0) + 1

    out: dict[int, np.ndarray] = {}
    default_gap = math.log1p(len(arr))
    for token in wanted:
        count = int(counts_arr[token]) if 0 <= token < len(counts_arr) else 0
        rank = int(ranks[token]) if 0 <= token < len(ranks) else len(ranks)
        gaps = gap_count.get(token, 0)
        mean_gap = gap_sum[token] / gaps if gaps else default_gap
        left_bucket = left_counts.get(token, {})
        right_bucket = right_counts.get(token, {})
        out[token] = np.asarray(
            [
                math.log1p(count),
                math.log1p(rank),
                mean_gap,
                entropy_from_counts(left_bucket),
                entropy_from_counts(right_bucket),
                math.log1p(len(left_bucket)),
                math.log1p(len(right_bucket)),
            ],
            dtype=np.float32,
        )
    return out


def piece_class_features(piece: str) -> np.ndarray:
    stripped = piece.strip()
    alpha = any(ch.isalpha() for ch in piece)
    digit = any(ch.isdigit() for ch in piece)
    punct = bool(piece) and all((not ch.isalnum()) and (not ch.isspace()) for ch in piece)
    return np.asarray(
        [
            math.log1p(len(piece)),
            math.log1p(len(piece.encode("utf-8", errors="replace"))),
            1.0 if piece.startswith(" ") else 0.0,
            1.0 if "\n" in piece else 0.0,
            1.0 if punct else 0.0,
            1.0 if digit else 0.0,
            1.0 if alpha else 0.0,
            1.0 if alpha and digit else 0.0,
            1.0 if stripped and stripped[0].isupper() else 0.0,
        ],
        dtype=np.float32,
    )


def recurrence_pair_features(src: np.ndarray, tgt: np.ndarray, piece: str) -> np.ndarray:
    diff = np.abs(src - tgt)
    return np.concatenate([src, tgt, -diff, piece_class_features(piece)]).astype(np.float32)


def standardize_groups(groups: list[tuple[np.ndarray, int, float]]):
    if not groups:
        return groups, np.zeros(1, dtype=np.float32), np.ones(1, dtype=np.float32)
    all_x = np.vstack([x for x, _, _ in groups])
    mean = all_x.mean(axis=0).astype(np.float32)
    std = all_x.std(axis=0).astype(np.float32)
    std[std < 1e-4] = 1.0
    return [((x - mean) / std, y, w) for x, y, w in groups], mean, std


def train_recurrence_linear(groups: list[tuple[np.ndarray, int, float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    groups, mean, std = standardize_groups(groups)
    if not groups:
        return np.zeros(1, dtype=np.float32), mean, std
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dim = groups[0][0].shape[1]
    weight = torch.zeros(dim, dtype=torch.float32, device=device, requires_grad=True)
    opt = torch.optim.Adam([weight], lr=RECURRENCE_LR)
    torch_groups = [
        (
            torch.as_tensor(x, dtype=torch.float32, device=device),
            int(y),
            float(w),
        )
        for x, y, w in groups
    ]
    for _ in range(RECURRENCE_EPOCHS):
        opt.zero_grad(set_to_none=True)
        loss = torch.zeros((), dtype=torch.float32, device=device)
        total_w = 0.0
        for x, y, group_w in torch_groups:
            logits = x @ weight
            loss = loss - torch.log_softmax(logits, dim=0)[y] * group_w
            total_w += group_w
        loss = loss / max(1.0, total_w) + 0.005 * torch.sum(weight * weight)
        loss.backward()
        opt.step()
    return weight.detach().cpu().numpy().astype(np.float32), mean, std


def build_synthetic_recurrence_groups(task, proxy_bpe, piece_cache: dict[int, str]) -> tuple[list[tuple[np.ndarray, int, float]], int]:
    raw = task.reference_text.read_text(encoding="utf-8", errors="ignore")[:RECURRENCE_SYNTH_CHARS]
    if proxy_bpe is None or not raw:
        return [], 0
    proxy_ids = np.asarray(proxy_bpe.encode(raw).ids, dtype=np.int64)
    target_ids = np.asarray(task.target_adapter.encode(raw), dtype=np.int64)
    if len(proxy_ids) == 0 or len(target_ids) == 0:
        return [], 0
    proxy_vocab = max(proxy_bpe.get_vocab_size(), int(proxy_ids.max(initial=0)) + 1)
    target_vocab = task.target_adapter.spec.vocab_size
    proxy_counts = np.bincount(proxy_ids, minlength=proxy_vocab)
    target_counts = np.bincount(target_ids, minlength=target_vocab)
    proxy_order = np.argsort(-proxy_counts)
    target_order = np.argsort(-target_counts)
    target_rank = np.empty(target_vocab, dtype=np.int64)
    target_rank[target_order] = np.arange(target_vocab)
    proxy_focus = [int(t) for t in proxy_order[:RECURRENCE_TYPES] if proxy_counts[int(t)] > 0]
    singleton_truth: dict[int, int] = {}
    target_candidates: set[int] = set()
    for u in proxy_focus:
        piece = proxy_bpe.decode([u])
        encoded = task.target_adapter.encode(piece)
        if len(encoded) != 1:
            continue
        true_p = int(encoded[0])
        singleton_truth[u] = true_p
        center = int(min(max(0, target_rank[true_p]), max(0, len(target_order) - RECURRENCE_TOPK)))
        for p in target_order[center : center + RECURRENCE_TOPK]:
            target_candidates.add(int(p))
        target_candidates.add(true_p)
    proxy_stats = recurrence_stats(proxy_ids, proxy_focus, proxy_vocab)
    target_stats = recurrence_stats(target_ids, list(target_candidates), target_vocab)
    groups: list[tuple[np.ndarray, int, float]] = []
    for u in proxy_focus:
        true_p = singleton_truth.get(u)
        if true_p is None:
            continue
        center = int(min(max(0, target_rank[true_p]), max(0, len(target_order) - RECURRENCE_TOPK)))
        candidates = [int(p) for p in target_order[center : center + RECURRENCE_TOPK]]
        if true_p not in candidates:
            candidates[-1] = true_p
        feats = []
        label_idx = -1
        for idx, p in enumerate(candidates):
            if p == true_p:
                label_idx = idx
            feats.append(
                recurrence_pair_features(
                    proxy_stats.get(u, np.zeros(7, dtype=np.float32)),
                    target_stats.get(p, np.zeros(7, dtype=np.float32)),
                    target_piece(task.target_adapter, p, piece_cache),
                )
            )
        if label_idx >= 0:
            groups.append((np.vstack(feats), label_idx, min(math.sqrt(float(proxy_counts[u])), 32.0)))
    return groups, len(proxy_focus)


def recurrence_proposal_audit(task, mapping: np.ndarray, baseline_metrics: dict[str, float]) -> dict[str, object]:
    print("recurrence_proposal_audit=1", flush=True)
    graph_pool = graph_candidate_pool(LAST_FINAL_EDGES, topk=RECURRENCE_TOPK)
    if not graph_pool:
        return {"error": "missing_graph_pool"}
    proxy_bpe = train_proxy_bpe(task.reference_text)
    piece_cache: dict[int, str] = {}
    train_groups, synthetic_focus = build_synthetic_recurrence_groups(task, proxy_bpe, piece_cache)
    weight, mean, std = train_recurrence_linear(train_groups)

    focus = [int(c) for c in LAST_C_FOCUS[:RECURRENCE_TYPES]]
    sample = np.asarray(task.cipher_ids[:SAMPLE_TOKENS], dtype=np.int64)
    candidate_targets = sorted({int(p) for c in focus for p in graph_pool.get(c, [])[:RECURRENCE_TOPK]})
    source_stats = recurrence_stats(sample, focus, max(int(sample.max(initial=0)) + 1, len(mapping)))
    target_stats = recurrence_stats(
        np.asarray(task.ref_ids[:RECURRENCE_STAT_TOKENS], dtype=np.int64),
        candidate_targets,
        task.target_adapter.spec.vocab_size,
    )
    inv_perm = inverse_permutation(task.perm)
    truth_cache: dict[int, tuple[int | None, str]] = {}
    labels: list[int] = []
    neural_scores: list[float] = []
    rank_scores: list[float] = []
    neural_overrides: dict[int, int] = {}
    graph_top1_good = 0
    neural_top1_good = 0
    labeled_groups = 0
    for c in focus:
        candidates = graph_pool.get(c, [])[:RECURRENCE_TOPK]
        if not candidates:
            continue
        true_p, _ = singleton_truth_for_cipher(task, c, inv_perm, truth_cache)
        feats = np.vstack(
            [
                recurrence_pair_features(
                    source_stats.get(c, np.zeros(7, dtype=np.float32)),
                    target_stats.get(int(p), np.zeros(7, dtype=np.float32)),
                    target_piece(task.target_adapter, int(p), piece_cache),
                )
                for p in candidates
            ]
        )
        scores = ((feats - mean) / std) @ weight
        best_p = int(candidates[int(np.argmax(scores))])
        if best_p != int(mapping[c]):
            neural_overrides[c] = best_p
        if true_p is not None and true_p in candidates:
            labeled_groups += 1
            if int(candidates[0]) == int(true_p):
                graph_top1_good += 1
            if best_p == int(true_p):
                neural_top1_good += 1
        for rank, p in enumerate(candidates):
            labels.append(1 if true_p is not None and int(p) == int(true_p) else 0)
            neural_scores.append(float(scores[rank]))
            rank_scores.append(float(-rank))

    mapped = np.asarray(
        [int(neural_overrides.get(int(c), int(mapping[int(c)]))) for c in task.cipher_ids[:SAMPLE_TOKENS]],
        dtype=np.int64,
    )
    neural_text = task.target_adapter.decode(mapped.tolist())
    neural_metrics = evaluate_recovery(task, neural_text, SAMPLE_TOKENS)
    report = {
        "synthetic_focus": synthetic_focus,
        "synthetic_groups": len(train_groups),
        "real_groups": labeled_groups,
        "moves": len(labels),
        "positive_rate": float(np.mean(labels)) if labels else 0.0,
        "graph_auc": auc_from_scores(labels, rank_scores),
        "graph_top100_precision": top_precision(labels, rank_scores, 100),
        "graph_top500_precision": top_precision(labels, rank_scores, 500),
        "neural_auc": auc_from_scores(labels, neural_scores),
        "neural_top100_precision": top_precision(labels, neural_scores, 100),
        "neural_top500_precision": top_precision(labels, neural_scores, 500),
        "graph_group_top1": graph_top1_good / max(1, labeled_groups),
        "neural_group_top1": neural_top1_good / max(1, labeled_groups),
        "neural_overrides": len(neural_overrides),
        "baseline_cer50k": float(baseline_metrics["cer50k"]),
        "neural_cer50k": float(neural_metrics["cer50k"]),
        "neural_bpb": float(neural_metrics["byte_lm_bpb"]),
    }
    print(
        "recurrence_proposal_report "
        + " ".join(
            f"{key}={value:.6f}" if isinstance(value, float) else f"{key}={value}"
            for key, value in report.items()
        ),
        flush=True,
    )
    return report


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
    prior_report = tokenizer_prior_audit(task, mapping, metrics) if TOKENIZER_PRIOR_AUDIT else None
    recurrence_report = recurrence_proposal_audit(task, mapping, metrics) if RECURRENCE_PROPOSAL_AUDIT else None

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
    if prior_report is not None:
        report["tokenizer_prior_audit"] = prior_report
    if recurrence_report is not None:
        report["recurrence_proposal_audit"] = recurrence_report
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
