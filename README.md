# autoresearch-detokenizer

Autonomous hillclimbing harness for the detokenization problem.

This is adapted from Karpathy's [`autoresearch`](https://github.com/karpathy/autoresearch)
shape: a fixed prep/evaluation file, one mutable experiment file, and a
Markdown program for the agent. Instead of optimizing an LLM training script, the
agent optimizes recovery of shuffled token IDs.

## Files

- `prepare.py` - fixed data prep, tokenizer adapters, fixture creation, and CER
  evaluation. Do not edit during a run.
- `train.py` - mutable recovery algorithm. Agents hillclimb this file.
- `program.md` - operating instructions for the autonomous loop.
- `results.tsv` - untracked experiment log created when a run starts.

## Install

```bash
uv sync
```

The default algorithm uses PyTorch for dense GPU similarity scoring. CUDA is
recommended for the full settings.

## Remote GPU

Run full `prepare.py` and `train.py` jobs on the user-provided remote H100/H200
host. The local Mac is for editing, syntax checks, and small inspections only.
Before a hillclimb run, sync this repo to the remote workspace, run `uv sync`
there, and execute experiments remotely.

## Prepare

Default setup creates a controlled shuffled-ID fixture:

- target text: FineWeb `sample/10BT/014_00000.parquet`
- reference text: FineWeb `sample/10BT/013_00000.parquet`
- source/cipher tokenizer: `kimi_k2`
- target/reference tokenizer: `openai_o200k`
- target length: `1,000,000` source tokens
- reference length: `100,000,000` target tokens

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 uv run prepare.py
```

The cache lives at:

```text
~/.cache/detokenizer-autoresearch
```

You can override the task:

```bash
DETOK_SOURCE=deepseek_v4_pro \
DETOK_TARGET=kimi_k2 \
DETOK_TARGET_TOKENS=100000000 \
DETOK_REFERENCE_TOKENS=100000000 \
uv run prepare.py
```

Registered tokenizers:

- `openai_o200k`
- `openai_cl100k`
- `qwen3`
- `qwen3_6_27b`
- `kimi_k2`
- `gemma4_31b`
- `deepseek_v4_pro`
- `llama3_1_8b`
- `mistral_medium_3_5`
- `mimo_v2_5_pro`

## Run One Experiment

```bash
uv run train.py > run.log 2>&1
grep "^cer50k:\|^byte_lm_bpb:\|^elapsed_seconds:" run.log
```

The summary looks like:

```text
---
cer50k:           0.298200
byte_lm_bpb:      2.702572
replacement_rate: 0.00004535
elapsed_seconds:  74.2
target_tokens_M:  1.000
reference_tokens_M: 95.030
```

Lower `cer50k` is the objective. `byte_lm_bpb` is secondary; it can improve when
the decoded text becomes more fluent but less correct.

## Autonomous Mode

Read `program.md`, create a branch, initialize `results.tsv`, run the baseline,
then iteratively modify only `train.py`.

The default loop:

```bash
uv run train.py > run.log 2>&1
grep "^cer50k:\|^byte_lm_bpb:\|^elapsed_seconds:" run.log
```

Keep commits that lower `cer50k`; discard regressions.
