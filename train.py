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

TOP_TOKENS = 50_000
ANCHORS = 8_192
CANDIDATE_WINDOW = 10_000
ROUNDS = 6
FREQ_WEIGHT = 0.12
TORCH_TOPK = 64
TORCH_BATCH_SIZE = 256
TORCH_CONTEXT_CHUNK = 5_000_000


def effective_candidate_window(num_cipher_tokens: int) -> int:
    return 25_000 if num_cipher_tokens >= 1_000_000 else CANDIDATE_WINDOW


def candidate_window_for_round(num_cipher_tokens: int, round_idx: int, rounds: int) -> int:
    if num_cipher_tokens < 1_000_000:
        return CANDIDATE_WINDOW
    return 10_000 if round_idx >= rounds // 2 else 25_000


def effective_rounds(num_cipher_tokens: int) -> int:
    return 8 if num_cipher_tokens >= 1_000_000 else ROUNDS


def counts(ids: np.ndarray, size: int) -> np.ndarray:
    return np.bincount(ids.astype(np.int64, copy=False), minlength=size)


def torch_context_maps(ids: np.ndarray, focus: np.ndarray, anchors: np.ndarray, device: str):
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
    for start in range(0, max(0, len(ids) - 1), TORCH_CONTEXT_CHUNK):
        stop = min(len(ids) - 1, start + TORCH_CONTEXT_CHUNK)
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

    return left_flat.view(len(focus), len(anchors)), right_flat.view(len(focus), len(anchors))


def normalize_features(x, token_counts: np.ndarray, device: str):
    counts_t = torch.as_tensor(np.maximum(1.0, token_counts.astype(np.float32)), dtype=torch.float32, device=device)
    y = x / torch.sqrt(counts_t)[:, None]
    norm = torch.linalg.vector_norm(y, dim=1).clamp_min(1e-12)
    return y / norm[:, None]


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


def align_shuffled(cipher_ids: np.ndarray, ref_ids: np.ndarray, target_vocab_size: int) -> np.ndarray:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)
    rounds = effective_rounds(len(cipher_ids))
    candidate_windows = [candidate_window_for_round(len(cipher_ids), i, rounds) for i in range(rounds)]
    print(f"candidate_window: {candidate_windows[0]}->{candidate_windows[-1]}", flush=True)
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

    for round_idx in range(rounds):
        candidate_window = candidate_windows[round_idx]
        c_anchors = c_focus[: min(ANCHORS, len(c_focus))]
        p_anchors = mapping[c_anchors]
        print(
            f"round {round_idx + 1}/{rounds}: focus={len(c_focus)} anchors={len(c_anchors)} "
            f"window={candidate_window}",
            flush=True,
        )
        with torch.no_grad():
            c_left, c_right = torch_context_maps(cipher_ids, c_focus, c_anchors, device)
            p_left, p_right = torch_context_maps(ref_ids, p_focus, p_anchors, device)
            c_vec = normalize_features(torch.cat([c_left, c_right], dim=1), c_counts[c_focus], device)
            p_vec = normalize_features(torch.cat([p_left, p_right], dim=1), p_counts[p_focus], device)
            del c_left, c_right, p_left, p_right
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
        next_mapping = mapping.copy()
        for _, c, p in edges:
            if c in used_c or p in used_p:
                continue
            next_mapping[c] = p
            used_c.add(c)
            used_p.add(p)
        mapping = next_mapping
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
    print(f"cipher_tokens: {len(task.cipher_ids):,}")
    print(f"reference_tokens: {len(task.ref_ids):,}")

    mapping = align_shuffled(task.cipher_ids, task.ref_ids, task.target_adapter.spec.vocab_size)
    mapped_sample = mapping[task.cipher_ids[:SAMPLE_TOKENS]]
    recovered_sample = task.target_adapter.decode(mapped_sample.tolist())
    metrics = evaluate_recovery(task, recovered_sample, SAMPLE_TOKENS)

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
        "candidate_window_schedule": [
            candidate_window_for_round(len(task.cipher_ids), i, effective_rounds(len(task.cipher_ids)))
            for i in range(effective_rounds(len(task.cipher_ids)))
        ],
        "rounds": effective_rounds(len(task.cipher_ids)),
        "freq_weight": FREQ_WEIGHT,
        "torch_topk": TORCH_TOPK,
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
