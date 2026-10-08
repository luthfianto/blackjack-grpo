# Blackjack GRPO on a laptop

Local post-training of `Qwen/Qwen3-0.6B` with LoRA and TRL's GRPOTrainer.
The model returns a short explanation and `HIT` or `STAND` as JSON.
The default setup targets one CUDA GPU with 12 GB VRAM. No account, paid API,
reward model, quantization, or experiment-tracking service is required.

The first laptop experiment is complete: 150 updates in 45 minutes. See
[RESULTS.md](RESULTS.md) for measured improvements, saved artifacts, and limits.
The canonical trained policy matched stand-at-17; explanations remain unreliable.

## Setup and checks

Run these commands from this directory. `uv sync --locked` installs the exact
dependency versions in `uv.lock` into an isolated `.venv`.

```bash
uv sync --locked
uv run --locked blackjack-grpo check
uv run --locked python -m unittest discover -s tests -v
uv run --locked blackjack-grpo download
```

The verified starting runtime is Python 3.12, PyTorch 2.14.0 (CUDA 13.0),
Transformers 5.17.0 and TRL 1.13.0. The check command actually runs BF16
forward/backward on the GPU; `nvidia-smi` alone does not establish compatibility.
Model downloads use the usual local Hugging Face cache. Subsequent runs reuse it.

## First experiment

```bash
# Original model: both prompt wordings, 280 states each, 100,000 hands per policy.
uv run --locked blackjack-grpo evaluate --output runs/before

# Five optimizer updates, with an automatic fresh-process OOM retry if needed.
uv run --locked blackjack-grpo smoke --output runs/smoke

# Resume one more update to verify optimizer/RNG/checkpoint loading.
uv run --locked blackjack-grpo train --output runs/smoke --resume auto --steps 6

# A fresh adapter for the actual experiment: 45 minutes or 300 updates.
uv run --locked blackjack-grpo train --output runs/main --minutes 45 --steps 300

uv run --locked blackjack-grpo evaluate --adapter runs/main/final --output runs/after
uv run --locked blackjack-grpo compare runs/before/evaluation.json runs/after/evaluation.json
```

Setup/download time is separate from the roughly one-hour compute budget.
Training checks its time limit after each completed optimizer update, so it can
overrun by one update and checkpoint-writing time. Evaluation time depends on
measured generation speed. If the smoke test used microbatch 1, also pass
`--microbatch 1` for the main run. This keeps 16 generated completions per
update: microbatch 2 × accumulation 8, or microbatch 1 × accumulation 16,
with four completions for each of four observations.

To continue a run, increase its **total** target step count and resume:

```bash
uv run --locked blackjack-grpo train --output runs/main --resume auto --steps 600 --minutes 45
```

`--minutes` is a fresh budget for each invocation. Checkpoints every 50 updates
and on normal shutdown contain optimizer, scheduler, trainer and RNG state;
`final/` is a portable inference adapter, not a resumable training checkpoint.
Ctrl+C requests a clean stop at the next completed update and saves a checkpoint.
Use another output directory for a fresh run. No artifacts are uploaded.

## Try the model

```bash
uv run --locked blackjack-grpo play --adapter runs/main/final
uv run --locked blackjack-grpo play --adapter runs/main/final --total 16 --dealer 10
uv run --locked blackjack-grpo play --adapter runs/main/final --total 18 --dealer 9 --usable-ace
```

The interactive command accepts `player_total dealer_upcard usable_ace`, for
example `18 9 1`. Dealer upcard 1 means ace. It uses the same greedy decoding
and 128-token budget as evaluation.

## Reward and evaluation meaning

Rules match Gymnasium `Blackjack-v1` with `natural=False, sab=False`: infinite
deck, dealer stands on soft 17, win/draw/loss returns +1/0/−1. There is no dealer
peek conditioning, natural bonus, betting, doubling or splitting.

The reward is the exact expected return of the selected action with **optimal
later play**, calculated by dynamic programming. Invalid JSON or an empty
explanation gets −2. The prompt contains no oracle values or optimal-action
labels. The explanation itself is not scored for factual correctness.

Training covers all 280 reachable observation tuples equally, repeating them
32 times. Alternative wording is withheld; it tests robustness of phrasing,
not generalization to unseen game states. The small state space permits
memorization, so this experiment does not establish general reasoning ability.

Evaluation generates one greedy response per state and wording and caches a
deterministic policy. It simulates that cached policy in independent Gymnasium
hands, rather than invoking the language model again each turn. Malformed
responses explicitly fall back to STAND for simulation; invalid-output rate,
valid-only regret and fallback-policy regret are reported separately.
Each simulated policy gets 100,000 hands and a 95% normal confidence interval.
Always-stand, stand-at-17 and the exact optimal policy use the same game rules.

`runs/*/evaluation.json` contains aggregate metrics; `*-completions.json`
contains every explanation and action for manual inspection. Training saves
session configuration, package versions, the dependency lock, JSONL metrics,
sample explanations, peak allocated GPU memory and adapter-change magnitude.
Qwen's long thinking mode is disabled; visible reasoning is explicitly requested.

The default optimizer is AdamW at a constant learning rate of 2e-5 (so extending
a completed run does not resume at zero learning rate), BF16 base weights, rank-16 LoRA
(alpha 32, zero dropout, all linear layers), gradient checkpointing, SDPA,
KL coefficient 0.02 and DAPO loss normalization inside GRPOTrainer.
This is direct GRPO without a supervised warm-up. A short run can be
inconclusive; the comparison command only marks improvement if action agreement
or fallback-policy regret improves without reducing output validity.
