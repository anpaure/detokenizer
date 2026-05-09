#!/usr/bin/env python3
"""Download and materialize a 100M-token FineWeb text slice.

This script intentionally downloads a real Hugging Face parquet shard rather
than relying on tiny streaming samples. It uses the dataset's `token_count`
column to stop after at least the requested token budget.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="HuggingFaceFW/fineweb")
    parser.add_argument("--repo-type", default="dataset")
    parser.add_argument("--filename", default="sample/10BT/014_00000.parquet")
    parser.add_argument("--cache-dir", default=".cache/hf")
    parser.add_argument("--out", default=".cache/fineweb_100m.txt")
    parser.add_argument("--meta-out", default=".cache/fineweb_100m_meta.json")
    parser.add_argument("--target-tokens", type=int, default=100_000_000)
    parser.add_argument("--batch-size", type=int, default=2048)
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    parquet_path = hf_hub_download(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        filename=args.filename,
        cache_dir=str(cache_dir),
    )

    rows = 0
    tokens = 0
    bytes_written = 0
    with out_path.open("w", encoding="utf-8", errors="ignore") as out:
        parquet = pq.ParquetFile(parquet_path)
        for batch in parquet.iter_batches(
            batch_size=args.batch_size,
            columns=["text", "token_count", "language"],
        ):
            texts = batch.column("text").to_pylist()
            token_counts = batch.column("token_count").to_pylist()
            languages = batch.column("language").to_pylist()
            for text, token_count, language in zip(texts, token_counts, languages):
                if language and language != "en":
                    continue
                if not text:
                    continue
                out.write(text)
                out.write("\n\n")
                rows += 1
                tokens += int(token_count or 0)
                bytes_written += len(text.encode("utf-8", errors="ignore")) + 2
                if tokens >= args.target_tokens:
                    break
            if tokens >= args.target_tokens:
                break

    meta = {
        "repo_id": args.repo_id,
        "filename": args.filename,
        "parquet_path": parquet_path,
        "out": str(out_path),
        "rows": rows,
        "dataset_token_count_sum": tokens,
        "bytes_written": bytes_written,
        "target_tokens": args.target_tokens,
    }
    Path(args.meta_out).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
