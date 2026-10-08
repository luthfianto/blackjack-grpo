# Blackjack GRPO laptop results

Recorded October 9, 2026 (Asia/Jakarta).

The local training pipeline completed successfully. Action selection improved over the original model, but the canonical greedy policy learned the stand-at-17 heuristic rather than the optimal dealer-aware strategy. Explanations remain factually unreliable.

## Run and verification

- Model: `Qwen/Qwen3-0.6B`, revision `c1899de289a04d12100db370d81485cdf75e47ca`; seed 42.
- RTX 5070 Ti Laptop GPU, 12 GB VRAM, WSL2; BF16 forward/backward verified in the isolated environment.
- Main run: 150 optimizer updates, 45.12 minutes, stopped by the 45-minute time limit.
- LoRA: 10,092,544 trainable parameters; rank 16, alpha 32, zero dropout, all linear layers.
- GRPOTrainer with DAPO normalization, KL coefficient 0.02, four completions per prompt, 16 completions per generation batch, 128-token output limit, constant learning rate 2e-5.
- Peak allocated GPU memory: 1.87 GiB. Adapter change L2 norm: 0.8105.
- All six unit tests passed, including Gymnasium parity and seeded Monte Carlo action-value checks. All 150 main-run losses and gradient norms were finite.
- Five-update smoke test and checkpoint resume passed; resume reached step 6 with a further adapter change of 0.0278.
- Checkpoints 50, 100 and 150 contain adapter, optimizer, scheduler and RNG state. The final adapter was successfully reloaded for evaluation.

## Decisions across all 280 observations

Each wording is evaluated once per reachable state with greedy decoding and the same 128-token budget. Training covers this state space; withheld wording tests phrasing robustness, not unseen-state generalization. Regret is averaged uniformly over observations; invalid responses explicitly use STAND for the fallback metric.

| Wording | Optimal action before → after | Valid JSON before → after | Mean fallback regret before → after |
| --- | --- | --- | --- |
| train | 77.9% → 87.1% | 98.6% → 100.0% | 0.08454 → 0.01810 |
| heldout | 47.9% → 71.4% | 98.2% → 100.0% | 0.25375 → 0.10844 |

The comparison meets the predefined improvement criterion for both wordings: agreement or regret improved without reducing output validity.

## Independent game simulations

Each policy played 100,000 Gymnasium hands with seed 42, `natural=False, sab=False`. Returns are unit win/draw/loss rewards (+1/0/−1); intervals are 95% normal confidence intervals. The model policies were cached from their greedy state responses; the model was not called again on every turn.

| Policy | Mean return per hand | 95% interval |
| --- | ---: | --- |
| Original, train wording | -0.23201 | [-0.23780, -0.22622] |
| Original, heldout wording | -0.62360 | [-0.62838, -0.61882] |
| Trained, train wording | -0.08377 | [-0.08961, -0.07793] |
| Trained, heldout wording | -0.29495 | [-0.30060, -0.28930] |
| always stand | -0.18646 | [-0.19239, -0.18053] |
| stand at 17 | -0.08377 | [-0.08961, -0.07793] |
| optimal | -0.04725 | [-0.05314, -0.04136] |

Direct inspection of the canonical policy table found zero differences from stand-at-17 across all 280 states. Its identical simulation result is therefore expected. The withheld-wording policy remains worse than the simple baselines despite improving over its own original-model result.

## Explanation review

Correct actions do not imply correct explanations. For a hard 16 against dealer 10, the trained canonical model correctly chooses HIT, but calls the hand soft, incorrectly identifies the dealer upcard as 17, and suggests the hit guarantees a win. For soft 18 against 9, it incorrectly describes the hand as hard and chooses STAND instead of the optimal HIT.

The reward evaluates action values and output format only. It does not establish factual reasoning quality. No reasoning-accuracy percentage is claimed.

## Use and artifacts

Run from this directory:

```bash
uv run --locked blackjack-grpo play --revision c1899de289a04d12100db370d81485cdf75e47ca --adapter runs/main/final
```

- `runs/main/final/`: inference adapter and tokenizer.
- `runs/main/checkpoint-150/`: resumable training checkpoint.
- `runs/main/summary.json`, session configuration, JSONL metrics and samples: measured training behavior.
- `runs/before/evaluation.json` and `runs/after/evaluation.json`: full aggregate metrics.
- `runs/after/*-completions.json`: all generated explanations; `*-policy.json`: cached actions.
- `runs/comparison.json`: controlled before/after comparison.
- `uv.lock` and `runs/main/environment.json`: exact dependencies and runtime versions.

See README.md for setup, smoke tests, timed training, resume and evaluation commands.
