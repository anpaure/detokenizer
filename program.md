# detokenizer autoresearch

This repo is an autonomous hillclimbing harness for recovering text from
shuffled language-model token IDs.

The setup mirrors Karpathy's `autoresearch`, but the research target is not LLM
training. The agent mutates the detokenizer algorithm in `train.py`, runs a
fixed controlled recovery suite, and keeps changes that lower CER across
tokenizers and target-token budgets.

## Setup

To set up a new run:

1. Agree on a run tag, e.g. `may10-kimi-o200k-1m`.
2. Create a fresh branch from the current branch:

   ```bash
   git checkout -b autoresearch/<tag>
   ```

3. Read the in-scope files:

   - `README.md`
   - `prepare.py` - fixed task/data/evaluation. Do not modify.
   - `train.py` - mutable recovery algorithm. This is what you edit.
   - `program.md`

4. Verify data and fixture prep:

   ```bash
   uv run prepare.py
   ```

5. Initialize `results.tsv` with this header:

   ```text
   commit	suite	mean_cer50k	max_cer50k	mean_bpb	memory_gb	status	description
   ```

6. Wait for the user to say start.

## Remote Compute Policy

Use the user-provided remote H100/H200 host for full data prep and experiment
runs. Do not run 100M-token `prepare.py` or `train.py` jobs on the local Mac
unless the user explicitly asks for a local run.

Local work is limited to editing, git operations, syntax checks, and tiny smoke
tests. Before starting a real hillclimb, sync the repo to a remote workspace
such as `/workspace/autoresearch-detokenizer`, then run:

```bash
uv sync
HF_HUB_ENABLE_HF_TRANSFER=1 uv run prepare.py
uv run train.py > run.log 2>&1
```

All timing, memory, crash logs, and `results.tsv` rows should come from the
remote GPU environment.

## Objective

Optimize for data efficiency and cross-tokenizer robustness. A good change
should recover lower-error text with fewer observed cipher tokens and should not
only work for one tokenizer pair.

Primary metric:

- mean `cer50k` across the active evaluation suite

Secondary metrics:

- max `cer50k` across the suite, to avoid hiding a tokenizer-specific failure
- `cer50k` at `100,000` target tokens, because this measures data efficiency
- `byte_lm_bpb`, only as a tie-breaker when CER is effectively equal
- elapsed time and memory, because slow methods are less useful for hillclimb

Do not optimize directly for `byte_lm_bpb`; fluent wrong text is still wrong.

## Default Single Task

The single-task default remains useful for quick debugging:

- source/cipher tokenizer: `kimi_k2`
- surrogate/reference tokenizer: `openai_o200k`
- shuffled target length: `1,000,000` source tokens
- reference length: `100,000,000` surrogate tokens
- metric: first-50k-character CER against the true source decode

Lower `cer50k` is better. This is not the final acceptance criterion unless the
user asks for a single-pair run.

## Evaluation Suite

Evaluate serious changes on multiple tokenizer pairs and target scales:

- target-token scales: `100,000`, `1,000,000`, `10,000,000`
- reference-token budget: default `100,000,000`, unless the user asks otherwise
- core tokenizer pairs:
  - `kimi_k2 -> openai_o200k`
  - `deepseek_v4_pro -> kimi_k2`
  - `qwen3_6_27b -> openai_o200k`
  - `mistral_medium_3_5 -> openai_o200k`
  - `gemma4_31b -> openai_o200k`
- optional hard/generalization pairs when time allows:
  - `mimo_v2_5_pro -> openai_o200k`
  - `openai_o200k -> kimi_k2`
  - `kimi_k2 -> qwen3_6_27b`

Use staged evaluation to keep iteration speed sane:

1. Quick screen: all core pairs at `100,000` target tokens.
2. Main decision: all core pairs at `1,000,000` target tokens.
3. Keep validation: the best candidate at `10,000,000` target tokens on at
   least `kimi_k2 -> openai_o200k`, `deepseek_v4_pro -> kimi_k2`, and
   `qwen3_6_27b -> openai_o200k`.

Prefer changes that improve the geometric mean of CER ratios versus the current
baseline:

```text
score = mean(log((new_cer50k + 1e-4) / (baseline_cer50k + 1e-4)))
```

Lower score is better. Reject changes that improve the mean only by causing a
large regression on one tokenizer pair or on the `100,000`-token setting.

The task can be overridden by environment variables:

```bash
DETOK_SOURCE=deepseek_v4_pro DETOK_TARGET=kimi_k2 DETOK_TARGET_TOKENS=100000000 uv run train.py
```

Do not change the suite definition during a run unless the user explicitly
asks. Within the suite, each row is a separate controlled task selected with
environment variables.

## What You Can Edit

Edit `train.py` only.

Good places to experiment:

- context feature construction
- anchor selection
- candidate ranking
- one-to-one assignment
- confidence-gated/self-training loops
- top-k candidate reranking
- local search/annealing if it improves true CER
- efficient use of GPU memory/time

## What You Cannot Edit

- Do not modify `prepare.py`; it defines the fixed task and metric.
- Do not modify the fixtures or oracle labels.
- Do not change dependencies unless the user explicitly approves.
- Do not hard-code answers, token IDs, document text, or anything derived from
  the oracle beyond using the printed metrics to choose keep/discard.

## Experiment Loop

The first run is always the baseline. For a single task:

```bash
uv run train.py > run.log 2>&1
grep "^cer50k:\|^byte_lm_bpb:\|^elapsed_seconds:" run.log
```

For the suite, run each task with explicit environment variables. Example:

```bash
DETOK_SOURCE=kimi_k2 DETOK_TARGET=openai_o200k DETOK_TARGET_TOKENS=100000 uv run prepare.py
DETOK_SOURCE=kimi_k2 DETOK_TARGET=openai_o200k DETOK_TARGET_TOKENS=100000 uv run train.py > run.kimi_o200k.100k.log 2>&1

DETOK_SOURCE=kimi_k2 DETOK_TARGET=openai_o200k DETOK_TARGET_TOKENS=1000000 uv run prepare.py
DETOK_SOURCE=kimi_k2 DETOK_TARGET=openai_o200k DETOK_TARGET_TOKENS=1000000 uv run train.py > run.kimi_o200k.1m.log 2>&1
```

Then loop:

1. Inspect current git state.
2. Modify `train.py` with one concrete idea.
3. Commit the change.
4. Run:

   ```bash
   uv run train.py > run.log 2>&1
   ```

5. Extract:

   ```bash
   grep "^cer50k:\|^byte_lm_bpb:\|^elapsed_seconds:" run.log
   ```

6. If the run crashed, inspect:

   ```bash
   tail -n 80 run.log
   ```

7. Append one row per suite tier to `results.tsv`.
8. If mean suite `cer50k` improved without a large per-task regression, keep
   the commit.
9. If the suite got worse or tied with extra complexity, reset to the previous
   good commit.

Use `byte_lm_bpb` only as a tie-breaker when CER is effectively equal.

## Results TSV

Rows are tab-separated. Use one row per suite tier, not one row per individual
task:

```text
commit	suite	mean_cer50k	max_cer50k	mean_bpb	memory_gb	status	description
a1b2c3d	core-100k	0.410000	0.620000	3.100000	8.4	keep	baseline
b2c3d4e	core-100k	0.380000	0.590000	3.050000	8.6	keep	confidence gated anchors
c3d4e5f	core-1m	0.210000	0.350000	2.710000	8.6	discard	mean improved but qwen regressed badly
d4e5f6g	core-10m	0.000000	0.000000	0.000000	0.0	crash	trigram features OOM
```

For memory on CUDA:

```bash
grep "torch.OutOfMemoryError\\|CUDA out of memory" run.log || true
```

`train.py` does not currently print peak VRAM; add it to `train.py` only if the
user asks or if memory becomes a recurring bottleneck.

## Keep/Discard Discipline

Prefer small, interpretable changes. Do not keep a complicated change for a
tiny bpb gain if CER does not improve. The output may become more fluent while
less correct; CER is the ground truth for controlled hillclimbing.

Once the user says start, keep iterating until interrupted.
