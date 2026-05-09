from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from recover_text import (
    counts,
    encode_reference,
    torch_context_maps,
    torch_normalize_features,
    torch_topk_edges,
)
from tokenizer_registry import build_tokenizer


def raw_token_bytes(adapter, token_id: int) -> bytes:
    if hasattr(adapter, "enc"):
        return adapter.token_repr(token_id)
    tok = getattr(adapter, "tok", None)
    model = getattr(tok, "model", None)
    if hasattr(model, "decode_single_token_bytes"):
        try:
            return model.decode_single_token_bytes(int(token_id))
        except KeyError:
            return f"<INVALID:{token_id}>".encode()
    return adapter.token_repr(token_id)


def build_token_bytes(adapter, token_ids: np.ndarray) -> dict[int, bytes]:
    out: dict[int, bytes] = {}
    for token_id in token_ids:
        tid = int(token_id)
        if tid not in out:
            out[tid] = raw_token_bytes(adapter, tid)
    return out


def build_vocab_byte_set(adapter, vocab_size: int) -> set[bytes]:
    out: set[bytes] = set()
    for token_id in range(vocab_size):
        out.add(raw_token_bytes(adapter, token_id))
    return out


def evaluate_topk(
    *,
    cipher_ids: np.ndarray,
    true_id_by_cipher: np.ndarray,
    source_adapter,
    target_adapter,
    target_vocab_size: int,
    c_focus: np.ndarray,
    p_focus: np.ndarray,
    mapping: np.ndarray,
    final_edges: list[tuple[float, int, int]],
    topk_values: tuple[int, ...],
) -> dict[str, object]:
    max_k = max(topk_values)
    c_counts = counts(cipher_ids, max(int(cipher_ids.max()) + 1, len(mapping), len(true_id_by_cipher)))
    total_occ = int(c_counts.sum())
    observed_types = int(np.count_nonzero(c_counts))
    focus_counts = c_counts[c_focus]
    focus_occ = int(focus_counts.sum())

    candidates: dict[int, list[int]] = {}
    for _, c, p in final_edges:
        current = candidates.setdefault(int(c), [])
        if len(current) < max_k:
            current.append(int(p))

    needed_source_ids: list[int] = []
    needed_target_ids: set[int] = set(int(p) for p in p_focus)
    for c in c_focus:
        true_id = int(true_id_by_cipher[int(c)]) if int(c) < len(true_id_by_cipher) else -1
        if true_id >= 0:
            needed_source_ids.append(true_id)
        if int(c) < len(mapping):
            needed_target_ids.add(int(mapping[int(c)]))
        needed_target_ids.update(candidates.get(int(c), []))

    source_bytes = build_token_bytes(source_adapter, np.asarray(needed_source_ids, dtype=np.int64))
    target_bytes = build_token_bytes(target_adapter, np.asarray(sorted(needed_target_ids), dtype=np.int64))
    target_vocab_bytes = build_vocab_byte_set(target_adapter, target_vocab_size)
    p_focus_bytes = {target_bytes[int(p)] for p in p_focus if int(p) in target_bytes}

    occ_hits = {k: 0 for k in topk_values}
    type_hits = {k: 0 for k in topk_values}
    occ_hits_all = {k: 0 for k in topk_values}
    type_hits_all = {k: 0 for k in topk_values}
    mapped_occ_hits = 0
    mapped_type_hits = 0
    focus_vocab_cover_occ = 0
    focus_vocab_cover_types = 0
    focus_pfocus_cover_occ = 0
    focus_pfocus_cover_types = 0
    valid_focus_types = 0

    for c in c_focus:
        ci = int(c)
        n = int(c_counts[ci])
        if n <= 0 or ci >= len(true_id_by_cipher):
            continue
        true_id = int(true_id_by_cipher[ci])
        if true_id < 0:
            continue
        valid_focus_types += 1
        src = source_bytes[true_id]
        if src in target_vocab_bytes:
            focus_vocab_cover_occ += n
            focus_vocab_cover_types += 1
        if src in p_focus_bytes:
            focus_pfocus_cover_occ += n
            focus_pfocus_cover_types += 1
        mapped_id = int(mapping[ci]) if ci < len(mapping) else -1
        if mapped_id in target_bytes and target_bytes[mapped_id] == src:
            mapped_occ_hits += n
            mapped_type_hits += 1
        cand = candidates.get(ci, [])
        cand_bytes = [target_bytes[p] for p in cand if p in target_bytes]
        for k in topk_values:
            if src in cand_bytes[:k]:
                occ_hits[k] += n
                type_hits[k] += 1
                occ_hits_all[k] += n
                type_hits_all[k] += 1

    return {
        "target_total_occurrences": total_occ,
        "target_observed_types": observed_types,
        "focus_types": int(len(c_focus)),
        "focus_valid_oracle_types": int(valid_focus_types),
        "focus_occurrences": focus_occ,
        "focus_occurrence_coverage": focus_occ / max(1, total_occ),
        "focus_type_coverage": len(c_focus) / max(1, observed_types),
        "oracle_target_vocab_coverage_occ_focus": focus_vocab_cover_occ / max(1, focus_occ),
        "oracle_target_vocab_coverage_type_focus": focus_vocab_cover_types / max(1, valid_focus_types),
        "oracle_p_focus_coverage_occ_focus": focus_pfocus_cover_occ / max(1, focus_occ),
        "oracle_p_focus_coverage_type_focus": focus_pfocus_cover_types / max(1, valid_focus_types),
        "greedy_mapping_exact_occ_focus": mapped_occ_hits / max(1, focus_occ),
        "greedy_mapping_exact_type_focus": mapped_type_hits / max(1, valid_focus_types),
        "topk_exact_occ_focus": {str(k): occ_hits[k] / max(1, focus_occ) for k in topk_values},
        "topk_exact_type_focus": {str(k): type_hits[k] / max(1, valid_focus_types) for k in topk_values},
        "topk_exact_occ_all_count_nonfocus_miss": {str(k): occ_hits_all[k] / max(1, total_occ) for k in topk_values},
        "topk_exact_type_all_count_nonfocus_miss": {str(k): type_hits_all[k] / max(1, observed_types) for k in topk_values},
    }


def run_alignment(args) -> dict[str, object]:
    import torch

    cipher_ids = np.load(args.ids, mmap_mode="r")
    perm = np.load(args.perm, mmap_mode="r")
    source_adapter = build_tokenizer(args.source_tokenizer)
    target_adapter = build_tokenizer(args.target_tokenizer)
    ref_ids = encode_reference(target_adapter, Path(args.reference_text), args.reference_tokens)

    true_id_by_cipher = np.full(max(int(cipher_ids.max()) + 1, len(perm)), -1, dtype=np.int64)
    true_id_by_cipher[np.asarray(perm, dtype=np.int64)] = np.arange(len(perm), dtype=np.int64)

    device = "cuda" if args.torch_device == "auto" and torch.cuda.is_available() else args.torch_device
    if device == "auto":
        device = "cpu"
    print(f"topk_eval device={device}", flush=True)

    vocab_size = target_adapter.spec.vocab_size
    c_counts = counts(cipher_ids, int(max(vocab_size, int(cipher_ids.max()) + 1)))
    p_counts = counts(ref_ids, vocab_size)
    c_order_all = np.argsort(-c_counts)
    p_order_all = np.argsort(-p_counts)
    c_focus = c_order_all[: min(args.top_tokens, np.count_nonzero(c_counts))].astype(np.int64)
    p_focus = p_order_all[: min(args.top_tokens, np.count_nonzero(p_counts))].astype(np.int64)

    mapping = np.zeros(max(len(c_counts), vocab_size), dtype=np.int64)
    initial = c_order_all[: len(p_order_all)]
    mapping[initial] = p_order_all[: len(initial)]

    c_log = np.log(np.maximum(c_counts, 1) / max(1, int(c_counts.sum())))
    p_log = np.log(np.maximum(p_counts, 1) / max(1, int(p_counts.sum())))
    p_rank = np.empty(vocab_size, dtype=np.int64)
    p_rank[p_order_all] = np.arange(vocab_size)
    final_edges: list[tuple[float, int, int]] = []

    for round_idx in range(args.rounds):
        c_anchors = c_focus[: min(args.anchors, len(c_focus))]
        p_anchors = mapping[c_anchors]
        print(
            f"topk_eval round={round_idx + 1}/{args.rounds} "
            f"focus={len(c_focus)} anchors={len(c_anchors)}",
            flush=True,
        )
        with torch.no_grad():
            c_left, c_right = torch_context_maps(cipher_ids, c_focus, c_anchors, device, args.torch_context_chunk)
            p_left, p_right = torch_context_maps(ref_ids, p_focus, p_anchors, device, args.torch_context_chunk)
            c_vec = torch_normalize_features(torch.cat([c_left, c_right], dim=1), c_counts[c_focus], device)
            p_vec = torch_normalize_features(torch.cat([p_left, p_right], dim=1), p_counts[p_focus], device)
            del c_left, c_right, p_left, p_right
            edges = torch_topk_edges(
                c_vec,
                p_vec,
                c_focus,
                p_focus,
                c_log,
                p_log,
                mapping,
                p_rank,
                args.candidate_window,
                args.freq_weight,
                args.torch_topk,
                args.torch_batch_size,
                device,
            )
            del c_vec, p_vec
            if device == "cuda":
                torch.cuda.empty_cache()

        final_edges = edges
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
        print(f"topk_eval assigned={len(used_c)}", flush=True)

    metrics = evaluate_topk(
        cipher_ids=cipher_ids,
        true_id_by_cipher=true_id_by_cipher,
        source_adapter=source_adapter,
        target_adapter=target_adapter,
        target_vocab_size=vocab_size,
        c_focus=c_focus,
        p_focus=p_focus,
        mapping=mapping,
        final_edges=final_edges,
        topk_values=tuple(args.topk_values),
    )
    metrics["source_tokenizer"] = source_adapter.spec.name
    metrics["target_tokenizer"] = target_adapter.spec.name
    metrics["target_vocab_size"] = int(vocab_size)
    metrics["top_tokens"] = int(args.top_tokens)
    metrics["anchors"] = int(args.anchors)
    metrics["candidate_window"] = int(args.candidate_window)
    metrics["rounds"] = int(args.rounds)
    metrics["torch_topk"] = int(args.torch_topk)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", required=True)
    parser.add_argument("--perm", required=True)
    parser.add_argument("--source-tokenizer", required=True)
    parser.add_argument("--target-tokenizer", required=True)
    parser.add_argument("--reference-text", required=True)
    parser.add_argument("--reference-tokens", type=int, default=100_000_000)
    parser.add_argument("--out", required=True)
    parser.add_argument("--top-tokens", type=int, default=50_000)
    parser.add_argument("--anchors", type=int, default=8192)
    parser.add_argument("--candidate-window", type=int, default=10_000)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--freq-weight", type=float, default=0.12)
    parser.add_argument("--torch-topk", type=int, default=64)
    parser.add_argument("--torch-batch-size", type=int, default=256)
    parser.add_argument("--torch-context-chunk", type=int, default=5_000_000)
    parser.add_argument("--torch-device", default="auto")
    parser.add_argument("--topk-values", type=int, nargs="+", default=[1, 2, 5])
    args = parser.parse_args()

    metrics = run_alignment(args)
    Path(args.out).write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
