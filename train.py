"""Graphless segmental transducer experiment.

This branch intentionally removes the graph aligner. The decoder starts from a
frequency-rank lexicon, then lets a byte-level segmental transducer rewrite
short high-loss islands with span-1/span-2 byte-string emissions.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from prepare import (
    CACHE_DIR,
    DEFAULT_REFERENCE_TOKENS,
    DEFAULT_SAMPLE_TOKENS,
    DEFAULT_SEED,
    DEFAULT_TARGET_TOKENS,
    evaluate_recovery,
    load_task,
)


SOURCE_TOKENIZER = os.environ.get("DETOK_SOURCE", "kimi_k2")
TARGET_TOKENIZER = os.environ.get("DETOK_TARGET", "openai_o200k")
TARGET_TOKENS = int(os.environ.get("DETOK_TARGET_TOKENS", DEFAULT_TARGET_TOKENS))
REFERENCE_TOKENS = int(os.environ.get("DETOK_REFERENCE_TOKENS", DEFAULT_REFERENCE_TOKENS))
SAMPLE_TOKENS = int(os.environ.get("DETOK_SAMPLE_TOKENS", DEFAULT_SAMPLE_TOKENS))
SEED = int(os.environ.get("DETOK_SEED", DEFAULT_SEED))

FOCUS_TYPES = 50_000
RANK_WINDOW = 4_096
SINGLE_CANDIDATES = 12
PAIR_CANDIDATES = 16
REFERENCE_SPAN_BYTES = 2_000_000
REFERENCE_SPANS_PER_BUCKET = 64

ISLAND_WINDOW = 8
ISLAND_CONTEXT = 8
MAX_ISLANDS = 256
MAX_ISLAND_LEN = 12
MAX_SPAN = 2
BEAM_SIZE = 64
LM_WEIGHT = 0.08
ALT_PENALTY = 0.45
RANK_PENALTY = 0.08
SPAN2_PENALTY = 1.8
REFERENCE_SPAN_PENALTY = 2.6
MIN_SCORE_GAIN = 2.0
MAX_EMIT_BYTES = 40
MAX_LENGTH_DELTA = 20


def counts(ids: np.ndarray, size: int) -> np.ndarray:
    return np.bincount(ids.astype(np.int64, copy=False), minlength=size)


def lm_total_bits(byte_lm, text: str) -> float:
    data = text.encode("utf-8", errors="replace")
    if not data:
        return 0.0
    return float(byte_lm.bits_per_byte(data) * len(data))


def target_piece(adapter, token_id: int, cache: dict[int, str]) -> str:
    token_id = int(token_id)
    piece = cache.get(token_id)
    if piece is None:
        piece = adapter.decode([token_id])
        cache[token_id] = piece
    return piece


def valid_piece(piece: str) -> bool:
    if not piece:
        return False
    if "\ufffd" in piece:
        return False
    return len(piece.encode("utf-8", errors="replace")) <= MAX_EMIT_BYTES


def piece_class(piece: str) -> str:
    stripped = piece.strip()
    if "\n" in piece:
        return "newline"
    if piece.isspace():
        return "space"
    if stripped and stripped.isdigit():
        return "digit"
    if stripped and stripped.isalpha():
        return "word"
    if stripped and stripped.isalnum():
        return "alnum"
    if stripped and all(not ch.isalnum() and not ch.isspace() for ch in stripped):
        return "punct"
    return "mixed"


def class_compatible(base: str, candidate: str) -> bool:
    base_class = piece_class(base)
    cand_class = piece_class(candidate)
    if base_class in {"newline", "space"}:
        return cand_class == base_class
    if base_class == "punct":
        return cand_class in {"punct", "mixed"}
    if base_class in {"word", "alnum", "digit"}:
        return cand_class in {"word", "alnum", "digit", "mixed"}
    return cand_class not in {"newline", "space"}


def length_bucket(piece: str) -> int:
    return min(32, len(piece.encode("utf-8", errors="replace")))


def build_initial_rank_mapping(
    cipher_ids: np.ndarray,
    ref_ids: np.ndarray,
    target_vocab_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    c_size = int(max(target_vocab_size, int(cipher_ids.max(initial=0)) + 1))
    c_counts = counts(cipher_ids, c_size)
    p_counts = counts(ref_ids, target_vocab_size)
    c_order = np.argsort(-c_counts)
    p_order = np.argsort(-p_counts)
    observed = c_order[c_counts[c_order] > 0]

    mapping = np.zeros(c_size, dtype=np.int64)
    if len(p_order):
        repeated = np.resize(p_order, len(observed))
        mapping[observed] = repeated

    c_rank = np.full(c_size, len(observed) + 1, dtype=np.int64)
    c_rank[observed] = np.arange(len(observed), dtype=np.int64)
    p_rank = np.empty(target_vocab_size, dtype=np.int64)
    p_rank[p_order] = np.arange(target_vocab_size, dtype=np.int64)
    return mapping, c_rank, p_rank, c_counts, p_counts


def add_candidate(out: list[tuple[str, float]], seen: set[str], piece: str, score: float) -> None:
    if piece in seen or not valid_piece(piece):
        return
    seen.add(piece)
    out.append((piece, score))


@dataclass
class CandidateState:
    mapping: np.ndarray
    c_rank: np.ndarray
    p_rank: np.ndarray
    p_order: np.ndarray
    target_adapter: object
    piece_cache: dict[int, str]
    single_cache: dict[int, list[tuple[str, float]]]
    pair_cache: dict[tuple[int, int], list[tuple[str, float]]]
    reference_spans: dict[tuple[str, int], list[str]]


def singleton_candidates(state: CandidateState, c: int) -> list[tuple[str, float]]:
    c = int(c)
    cached = state.single_cache.get(c)
    if cached is not None:
        return cached

    current_p = int(state.mapping[c])
    current_piece = target_piece(state.target_adapter, current_p, state.piece_cache)
    out: list[tuple[str, float]] = []
    seen: set[str] = set()
    add_candidate(out, seen, current_piece, 0.0)

    center = int(min(max(state.c_rank[c], 0), len(state.p_order) - 1))
    lo = max(0, center - RANK_WINDOW)
    hi = min(len(state.p_order), center + RANK_WINDOW + 1)
    scored: list[tuple[float, int, str]] = []
    for p in state.p_order[lo:hi]:
        p_int = int(p)
        piece = target_piece(state.target_adapter, p_int, state.piece_cache)
        if not valid_piece(piece) or not class_compatible(current_piece, piece):
            continue
        rank_delta = abs(int(state.p_rank[p_int]) - center)
        prior = -ALT_PENALTY - RANK_PENALTY * math.log1p(rank_delta)
        scored.append((prior, p_int, piece))
    scored.sort(reverse=True)
    for rank, (prior, _, piece) in enumerate(scored[: SINGLE_CANDIDATES - 1]):
        add_candidate(out, seen, piece, prior - 0.08 * rank)

    state.single_cache[c] = out[:SINGLE_CANDIDATES]
    return state.single_cache[c]


def pair_candidates(state: CandidateState, left: int, right: int) -> list[tuple[str, float]]:
    key = (int(left), int(right))
    cached = state.pair_cache.get(key)
    if cached is not None:
        return cached

    left_options = singleton_candidates(state, key[0])[:4]
    right_options = singleton_candidates(state, key[1])[:4]
    base = left_options[0][0] + right_options[0][0]
    out: list[tuple[str, float]] = []
    seen: set[str] = set()
    add_candidate(out, seen, base, -SPAN2_PENALTY)

    for li, (lp, lscore) in enumerate(left_options):
        for ri, (rp, rscore) in enumerate(right_options):
            piece = lp + rp
            if abs(len(piece) - len(base)) > MAX_LENGTH_DELTA:
                continue
            score = -SPAN2_PENALTY + 0.5 * (lscore + rscore) - 0.12 * (li + ri)
            add_candidate(out, seen, piece, score)

    if len(base.encode("utf-8", errors="replace")) <= MAX_EMIT_BYTES:
        encoded = state.target_adapter.encode(base)
        if 1 <= len(encoded) <= 2:
            normalized = state.target_adapter.decode(encoded)
            add_candidate(out, seen, normalized, -SPAN2_PENALTY - 0.2)

    bucket = (piece_class(base), length_bucket(base))
    for span in state.reference_spans.get(bucket, [])[:8]:
        if abs(len(span) - len(base)) > MAX_LENGTH_DELTA:
            continue
        add_candidate(out, seen, span, -REFERENCE_SPAN_PENALTY)

    state.pair_cache[key] = out[:PAIR_CANDIDATES]
    return state.pair_cache[key]


def build_reference_span_inventory(reference_text: Path) -> dict[tuple[str, int], list[str]]:
    raw = reference_text.read_bytes()[:REFERENCE_SPAN_BYTES].decode("utf-8", errors="ignore")
    pattern = re.compile(
        r"\n+|\s+[A-Za-z]{1,24}|[A-Za-z]{2,24}|\s+\d{1,10}|\d{1,10}|"
        r"\s*[.,;:!?()\[\]{}\"'`\\/-]+"
    )
    counts_by_bucket: dict[tuple[str, int], Counter[str]] = {}
    for match in pattern.finditer(raw):
        piece = match.group(0)
        if not valid_piece(piece):
            continue
        bucket = (piece_class(piece), length_bucket(piece))
        counts_by_bucket.setdefault(bucket, Counter())[piece] += 1

    inventory: dict[tuple[str, int], list[str]] = {}
    for bucket, counter in counts_by_bucket.items():
        inventory[bucket] = [piece for piece, _ in counter.most_common(REFERENCE_SPANS_PER_BUCKET)]
    return inventory


def baseline_piece_sequence(ids: np.ndarray, state: CandidateState) -> list[str]:
    return [target_piece(state.target_adapter, int(state.mapping[int(c)]), state.piece_cache) for c in ids]


def pick_high_loss_islands(pieces: list[str], byte_lm) -> list[tuple[int, int]]:
    scored: list[tuple[float, int, int]] = []
    n = len(pieces)
    for start in range(0, max(0, n - ISLAND_WINDOW + 1), max(1, ISLAND_WINDOW // 2)):
        end = min(n, start + ISLAND_WINDOW)
        text = "".join(pieces[start:end])
        data_len = max(1, len(text.encode("utf-8", errors="replace")))
        score = lm_total_bits(byte_lm, text) / data_len
        scored.append((score, start, end))
    scored.sort(reverse=True)

    selected: list[tuple[int, int]] = []
    occupied = np.zeros(n, dtype=bool)
    for _, start, end in scored:
        start = max(0, start)
        end = min(n, start + min(MAX_ISLAND_LEN, end - start))
        if end <= start or occupied[max(0, start - ISLAND_CONTEXT) : min(n, end + ISLAND_CONTEXT)].any():
            continue
        selected.append((start, end))
        occupied[start:end] = True
        if len(selected) >= MAX_ISLANDS:
            break
    selected.sort()
    return selected


def segmental_decode_island(
    island_ids: np.ndarray,
    old_pieces: list[str],
    prefix: str,
    suffix: str,
    state: CandidateState,
    byte_lm,
) -> tuple[str, float, int]:
    n = len(island_ids)
    old_mid = "".join(old_pieces)
    prefix_bits = lm_total_bits(byte_lm, prefix)
    old_score = -LM_WEIGHT * (lm_total_bits(byte_lm, prefix + old_mid + suffix) - prefix_bits)

    beams: list[list[tuple[float, str, int]]] = [[] for _ in range(n + 1)]
    beams[0] = [(0.0, "", 0)]
    bits_cache: dict[str, float] = {prefix: prefix_bits}

    def bits(text: str) -> float:
        value = bits_cache.get(text)
        if value is None:
            value = lm_total_bits(byte_lm, text)
            bits_cache[text] = value
        return value

    for i in range(n):
        if not beams[i]:
            continue
        for score, emitted, span2_count in beams[i]:
            before = prefix + emitted
            before_bits = bits(before)
            c = int(island_ids[i])
            for piece, prior in singleton_candidates(state, c):
                after = emitted + piece
                delta_bits = bits(prefix + after) - before_bits
                beams[i + 1].append((score - LM_WEIGHT * delta_bits + prior, after, span2_count))
            if MAX_SPAN >= 2 and i + 1 < n:
                right = int(island_ids[i + 1])
                for piece, prior in pair_candidates(state, c, right):
                    after = emitted + piece
                    delta_bits = bits(prefix + after) - before_bits
                    beams[i + 2].append((score - LM_WEIGHT * delta_bits + prior, after, span2_count + 1))
        for j in (i + 1, i + 2):
            if j <= n and len(beams[j]) > BEAM_SIZE:
                beams[j].sort(key=lambda item: item[0], reverse=True)
                beams[j] = beams[j][:BEAM_SIZE]

    best_text = old_mid
    best_score = -math.inf
    best_span2 = 0
    for score, emitted, span2_count in beams[n]:
        if abs(len(emitted) - len(old_mid)) > MAX_LENGTH_DELTA:
            continue
        suffix_bits = bits(prefix + emitted + suffix) - bits(prefix + emitted)
        total = score - LM_WEIGHT * suffix_bits
        if total > best_score:
            best_score = total
            best_text = emitted
            best_span2 = span2_count
    gain = best_score - old_score
    return best_text, gain, best_span2


def graphless_segmental_decode(task) -> str:
    sample_ids = task.cipher_ids[: min(len(task.cipher_ids), SAMPLE_TOKENS)]
    mapping, c_rank, p_rank, _, p_counts = build_initial_rank_mapping(
        task.cipher_ids,
        task.ref_ids,
        task.target_adapter.spec.vocab_size,
    )
    p_order = np.argsort(-p_counts)
    state = CandidateState(
        mapping=mapping,
        c_rank=c_rank,
        p_rank=p_rank,
        p_order=p_order,
        target_adapter=task.target_adapter,
        piece_cache={},
        single_cache={},
        pair_cache={},
        reference_spans=build_reference_span_inventory(task.reference_text),
    )
    pieces = baseline_piece_sequence(sample_ids, state)
    islands = pick_high_loss_islands(pieces, task.byte_lm)

    accepted = 0
    span2_used = 0
    gains: list[float] = []
    for start, end in islands:
        prefix = "".join(pieces[max(0, start - ISLAND_CONTEXT) : start])
        suffix = "".join(pieces[end : min(len(pieces), end + ISLAND_CONTEXT)])
        new_text, gain, span2 = segmental_decode_island(
            sample_ids[start:end],
            pieces[start:end],
            prefix,
            suffix,
            state,
            task.byte_lm,
        )
        if new_text == "".join(pieces[start:end]) or gain < MIN_SCORE_GAIN:
            continue
        pieces[start:end] = [new_text] + [""] * (end - start - 1)
        accepted += 1
        span2_used += span2
        gains.append(gain)

    print(f"rank_init_focus={min(FOCUS_TYPES, len(np.unique(task.cipher_ids)))}", flush=True)
    print(f"segmental_islands={len(islands)} accepted={accepted} span2_used={span2_used}", flush=True)
    if gains:
        arr = np.asarray(gains, dtype=np.float32)
        print(f"segmental_gain_median={float(np.median(arr)):.6f}", flush=True)
        print(f"segmental_gain_p90={float(np.percentile(arr, 90)):.6f}", flush=True)
    return "".join(pieces)


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

    recovered_sample = graphless_segmental_decode(task)
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
        "decoder": "graphless_segmental_transducer",
        "rank_window": RANK_WINDOW,
        "single_candidates": SINGLE_CANDIDATES,
        "pair_candidates": PAIR_CANDIDATES,
        "beam_size": BEAM_SIZE,
        "lm_weight": LM_WEIGHT,
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


if __name__ == "__main__":
    main()
