#!/usr/bin/env python3
"""Generate and recover shuffled-ID fixtures for every tokenizer adapter.

This is the experiment harness. `recover_text.py` is the final plug-and-play
tool for unknown ID streams; this file creates controlled test cases so we can
measure how well the recovery pipeline handles OpenAI, Qwen, Kimi, etc.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from rapidfuzz.distance import Levenshtein

from tokenizer_registry import DEFAULT_TOKENIZERS, build_tokenizer
from tokenizer_types.base import read_text_batches


def char_error_rate(a: str, b: str, max_chars: int = 20_000) -> float:
    a = a[:max_chars]
    b = b[:max_chars]
    if not a:
        return 1.0 if b else 0.0
    return Levenshtein.distance(a, b) / len(a)


def encode_text_file(adapter, text_file: Path, token_limit: int) -> np.ndarray:
    chunks: list[np.ndarray] = []
    total = 0
    for batch in read_text_batches(text_file):
        for ids in adapter.encode_batch(batch):
            need = token_limit - total
            if need <= 0:
                break
            if ids:
                arr = np.asarray(ids[:need], dtype=np.int64)
                chunks.append(arr)
                total += len(arr)
        if total >= token_limit:
            break
    if not chunks:
        return np.asarray([], dtype=np.int64)
    return np.concatenate(chunks)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-file", default=".cache/fineweb_100m.txt")
    parser.add_argument("--out-dir", default="zoo_eval")
    parser.add_argument("--source-tokenizers", default=",".join(DEFAULT_TOKENIZERS))
    parser.add_argument("--candidate-tokenizers", default=",".join(DEFAULT_TOKENIZERS + ["trained_bpe"]))
    parser.add_argument("--target-tokens", type=int, default=1_000_000)
    parser.add_argument("--reference-tokens", type=int, default=2_000_000)
    parser.add_argument("--sample-tokens", type=int, default=200_000)
    parser.add_argument("--top-tokens", type=int, default=1200)
    parser.add_argument("--anchors", type=int, default=256)
    parser.add_argument("--candidate-window", type=int, default=160)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--native", action="store_true", help="Do not shuffle IDs.")
    args = parser.parse_args()

    text_file = Path(args.text_file)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = [x.strip() for x in args.source_tokenizers.split(",") if x.strip()]
    summary: list[dict[str, object]] = []

    for source_name in sources:
        print(f"source={source_name}", flush=True)
        adapter = build_tokenizer(source_name)
        ids = encode_text_file(adapter, text_file, args.target_tokens)
        if ids.size == 0:
            raise RuntimeError(f"no ids encoded for {source_name}")
        rng = np.random.default_rng(args.seed)
        if args.native:
            observed = ids
            mode_label = "native"
        else:
            perm = rng.permutation(adapter.spec.vocab_size)
            observed = perm[ids]
            mode_label = "shuffled"
            np.save(out_dir / f"{source_name}.perm.npy", perm)
        ids_path = out_dir / f"{source_name}.{mode_label}.ids.npy"
        np.save(ids_path, observed)

        recovered = out_dir / f"{source_name}.{mode_label}.recovered.txt"
        report = out_dir / f"{source_name}.{mode_label}.report.json"
        mapping_path = out_dir / f"{source_name}.{mode_label}.mapping.npy"
        cmd = [
            sys.executable,
            "recover_text.py",
            "--ids",
            str(ids_path),
            "--out",
            str(recovered),
            "--report",
            str(report),
            "--save-mapping",
            str(mapping_path),
            "--reference-text",
            str(text_file),
            "--tokenizers",
            args.candidate_tokenizers,
            "--reference-tokens",
            str(args.reference_tokens),
            "--sample-tokens",
            str(args.sample_tokens),
            "--top-tokens",
            str(args.top_tokens),
            "--anchors",
            str(args.anchors),
            "--candidate-window",
            str(args.candidate_window),
            "--rounds",
            str(args.rounds),
        ]
        subprocess.run(cmd, check=True)
        data = json.loads(report.read_text(encoding="utf-8"))
        recovered_text = recovered.read_text(encoding="utf-8", errors="ignore")
        true_text = adapter.decode(ids[: args.sample_tokens].tolist())
        eval_record = {
            "prefix_char_error_rate": char_error_rate(true_text, recovered_text),
        }
        if data["chosen"]["mode"] == "shuffled" and data["chosen"]["tokenizer"] == adapter.spec.name and mapping_path.exists() and not args.native:
            mapping = np.load(mapping_path)
            inv_perm = np.empty(adapter.spec.vocab_size, dtype=np.int64)
            inv_perm[perm] = np.arange(adapter.spec.vocab_size)
            observed_counts = np.bincount(observed.astype(np.int64), minlength=max(int(observed.max()) + 1, adapter.spec.vocab_size))
            ok = 0
            total = int(observed_counts.sum())
            for c, n in enumerate(observed_counts):
                if not n or c >= len(mapping):
                    continue
                try:
                    if adapter.token_repr(int(inv_perm[c])) == adapter.token_repr(int(mapping[c])):
                        ok += int(n)
                except Exception:
                    pass
            eval_record["weighted_exact_token_string_accuracy"] = ok / max(1, total)
        summary.append(
            {
                "source": source_name,
                "mode": mode_label,
                "source_vocab": adapter.spec.vocab_size,
                "num_ids": int(len(ids)),
                "chosen_mode": data["chosen"]["mode"],
                "chosen_tokenizer": data["chosen"]["tokenizer"],
                "score": data["chosen"]["metrics"]["score"],
                **eval_record,
                "preview": data["chosen"]["preview"][:240],
            }
        )
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
