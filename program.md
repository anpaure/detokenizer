# detokenizer autoresearch

This repo is an autonomous hillclimbing harness for recovering text from
shuffled language-model token IDs.

The setup mirrors Karpathy's `autoresearch`, but the research target is not LLM
training. The agent mutates the detokenizer algorithm in `train.py`, runs a
fixed controlled recovery task, and keeps changes that lower `cer50k`.

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
   commit	cer50k	byte_lm_bpb	memory_gb	status	description
   ```

6. Wait for the user to say start.

## Task

Default task:

- source/cipher tokenizer: `kimi_k2`
- surrogate/reference tokenizer: `openai_o200k`
- shuffled target length: `1,000,000` source tokens
- reference length: `100,000,000` surrogate tokens
- metric: first-50k-character CER against the true source decode

Lower `cer50k` is better. `byte_lm_bpb` is secondary and can be misleading: it
measures language-likeness, not exact recovery.

The task can be overridden by environment variables:

```bash
DETOK_SOURCE=deepseek_v4_pro DETOK_TARGET=kimi_k2 DETOK_TARGET_TOKENS=100000000 uv run train.py
```

Do not change the source/target tokenizer or token budgets during a run unless
the user explicitly asks.

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

The first run is always the baseline:

```bash
uv run train.py > run.log 2>&1
grep "^cer50k:\|^byte_lm_bpb:\|^elapsed_seconds:" run.log
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

7. Append one row to `results.tsv`.
8. If `cer50k` improved, keep the commit.
9. If `cer50k` got worse or tied with extra complexity, reset to the previous
   good commit.

Use `byte_lm_bpb` only as a tie-breaker when CER is effectively equal.

## Results TSV

Rows are tab-separated:

```text
commit	cer50k	byte_lm_bpb	memory_gb	status	description
a1b2c3d	0.298200	2.702572	8.4	keep	baseline
b2c3d4e	0.284000	2.710000	8.6	keep	confidence gated anchors
c3d4e5f	0.310000	2.650000	8.6	discard	annealing optimized bpb but hurt CER
d4e5f6g	0.000000	0.000000	0.0	crash	trigram features OOM
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
