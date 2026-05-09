# detokenizer

Research code for recovering text from raw language-model token ID streams.

The main target is the hard case: the input is a long sequence of integer token
IDs, the ID-to-token mapping may be shuffled, and the exact source tokenizer may
be unknown. The decoder aligns the observed token graph to one or more candidate
tokenizer codebooks using frequency and left/right context statistics, then
decodes and ranks candidates with a byte-level language-model score.

## Install

```bash
uv sync
```

For large runs, use a machine with CUDA and a working PyTorch install. The CPU
path exists, but the full 50k-token graph settings are intended for GPU.

```bash
uv sync --extra torch
```

## Tokenizers

Registered tokenizer names:

- `openai_o200k`: OpenAI `o200k_base`
- `openai_cl100k`: OpenAI `cl100k_base`
- `qwen3`: `Qwen/Qwen3-0.6B`
- `kimi_k2`: `moonshotai/Kimi-K2-Instruct`
- `gemma4_31b`: `google/gemma-4-31B`
- `deepseek_v4_pro`: `deepseek-ai/DeepSeek-V4-Pro`
- `surrogate_bpe`: trained local ByteLevel-BPE surrogate codebooks
- `trained_bpe`: one trained local ByteLevel-BPE surrogate

## Decode An ID File

Input formats:

- `.npy` integer array
- `.json` array
- text/CSV/TSV containing integer IDs

Full-scale shuffled-ID command, matching the 100M-token experiments:

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 uv run python recover_text.py \
  --ids /path/to/token_ids.npy \
  --out recovered.txt \
  --report recovered.report.json \
  --reference-text .cache/fineweb_ref.txt \
  --reference-tokens 100000000 \
  --tokenizers openai_o200k,qwen3,kimi_k2,deepseek_v4_pro,gemma4_31b \
  --force-mode shuffled \
  --aligner torch \
  --sample-tokens 500000 \
  --top-tokens 50000 \
  --anchors 8192 \
  --candidate-window 10000 \
  --rounds 6 \
  --freq-weight 0.12 \
  --torch-topk 64 \
  --torch-batch-size 256 \
  --torch-context-chunk 5000000 \
  --save-mapping recovered.mapping.npy
```

If `--reference-text` does not exist, `recover_text.py` materializes a FineWeb
slice automatically. To create a specific reference shard yourself:

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 uv run python download_fineweb_100m.py \
  --filename sample/10BT/013_00000.parquet \
  --out .cache/fineweb_ref.txt \
  --meta-out .cache/fineweb_ref.meta.json \
  --target-tokens 100000000
```

Small smoke-test command:

```bash
uv run python recover_text.py \
  --ids /path/to/token_ids.npy \
  --out recovered.txt \
  --reference-tokens 2000000 \
  --sample-tokens 100000 \
  --top-tokens 5000 \
  --anchors 1024 \
  --candidate-window 2000 \
  --rounds 3 \
  --tokenizers openai_o200k,qwen3,kimi_k2 \
  --force-mode shuffled \
  --aligner torch
```

For native, unshuffled public-tokenizer IDs, use:

```bash
uv run python recover_text.py \
  --ids /path/to/token_ids.npy \
  --out recovered.txt \
  --tokenizers openai_o200k,qwen3,kimi_k2,deepseek_v4_pro,gemma4_31b \
  --force-mode auto
```

For strict unknown-vocabulary experiments, include `surrogate_bpe`:

```bash
uv run python recover_text.py \
  --ids /path/to/token_ids.npy \
  --out recovered.txt \
  --tokenizers surrogate_bpe \
  --surrogate-vocab-scales 0.5,0.75,1.0,1.25,1.5 \
  --surrogate-train-bytes 128000000 \
  --reference-tokens 10000000
```

## Evaluate Controlled Fixtures

Create shuffled fixtures from known tokenizers, run recovery, and report prefix
character error rate:

```bash
uv run python evaluate_tokenizer_zoo.py \
  --text-file .cache/fineweb_ref.txt \
  --out-dir zoo_eval \
  --source-tokenizers openai_o200k,qwen3,kimi_k2,deepseek_v4_pro,gemma4_31b \
  --candidate-tokenizers openai_o200k,qwen3,kimi_k2,deepseek_v4_pro,gemma4_31b \
  --target-tokens 1000000 \
  --reference-tokens 2000000
```

Evaluate oracle top-k candidate accuracy for a shuffled fixture:

```bash
uv run python evaluate_candidate_topk.py \
  --ids fixture.cipher_ids.npy \
  --perm fixture.perm.npy \
  --source-tokenizer kimi_k2 \
  --target-tokenizer openai_o200k \
  --reference-text .cache/fineweb_ref.txt \
  --reference-tokens 100000000 \
  --out candidate_topk.json \
  --top-tokens 50000 \
  --anchors 8192 \
  --candidate-window 10000 \
  --rounds 6 \
  --topk-values 1 2 5
```

## Files

- `recover_text.py`: main decoder
- `tokenizer_registry.py`, `tokenizer_types/`: tokenizer adapters
- `download_fineweb_100m.py`: materialize a public reference text slice
- `evaluate_tokenizer_zoo.py`: controlled tokenizer-zoo experiments
- `evaluate_candidate_topk.py`: oracle top-k candidate evaluation
- `bpe_100m_attack.py`: synthetic strict-unknown ByteLevel-BPE experiment

## Notes

Token IDs are not anonymization. If the tokenizer is public and IDs are native,
decoding is exact. If IDs are shuffled by a fixed permutation, large corpora
still leak frequency and context structure. Exact recovery is not identifiable
for arbitrary private tokenizers without a prior, oracle, or surrogate codebook.
