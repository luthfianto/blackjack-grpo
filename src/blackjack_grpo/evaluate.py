import json
from pathlib import Path
import time

from .game import State, action_values, optimal_action, simulate, states
from .prompts import parse_completion, prompt
from .runtime import copy_lock, generate, load_model, write_json


def evaluate(args):
    import torch
    from transformers import set_seed

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "evaluation.json").exists():
        raise ValueError("Evaluation already exists; choose another --output")
    set_seed(args.seed)
    model, tokenizer = load_model(args.model, args.adapter, args.revision)
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    report = dict(model=args.model, adapter=str(args.adapter) if args.adapter else None,
                  revision=getattr(model.config, "_commit_hash", args.revision), seed=args.seed,
                  decoding=dict(do_sample=False, max_new_tokens=128, enable_thinking=False),
                  invalid_gameplay_fallback="STAND", wordings={})
    observations = states()
    for wording in ("train", "heldout"):
        rows, cache = [], {}
        for offset in range(0, len(observations), args.batch_size):
            chunk = observations[offset:offset + args.batch_size]
            completions = generate(model, tokenizer, [prompt(s, wording) for s in chunk])
            for state, (text, length) in zip(chunk, completions, strict=True):
                parsed = parse_completion(text)
                values = action_values(state)
                action = parsed["action"] if parsed else None
                best = max(values.values())
                rows.append(dict(state=state.key, completion=text, action=action, valid=parsed is not None,
                                 tokens=length, at_token_limit=length == 128,
                                 optimal=action is not None and best - values[action] <= 1e-9,
                                 regret=best - values[action] if action else None))
                cache[state.key] = action or "STAND"
            print(f"{wording}: {min(offset + len(chunk), len(observations))}/{len(observations)} states", flush=True)
        valid = [row for row in rows if row["valid"]]
        report["wordings"][wording] = dict(
            states=len(rows), valid_output_rate=len(valid) / len(rows),
            optimal_action_agreement=sum(row["optimal"] for row in rows) / len(rows),
            mean_regret_valid=sum(row["regret"] for row in valid) / len(valid) if valid else None,
            # Invalid responses use the explicitly disclosed stand fallback for whole-policy metrics.
            mean_regret_with_stand_fallback=sum(max(action_values(s).values()) - action_values(s)[cache[s.key]]
                                              for s in observations) / len(observations),
            mean_completion_tokens=sum(row["tokens"] for row in rows) / len(rows),
            token_limit_rate=sum(row["at_token_limit"] for row in rows) / len(rows),
            simulation=simulate(lambda s: cache[s.key], args.episodes, args.seed),
        )
        write_json(output / f"{wording}-completions.json", rows)
        write_json(output / f"{wording}-policy.json", cache)
    report["baselines"] = {
        "always_stand": simulate(lambda s: "STAND", args.episodes, args.seed),
        "stand_at_17": simulate(lambda s: "STAND" if s.total >= 17 else "HIT", args.episodes, args.seed),
        "optimal": simulate(optimal_action, args.episodes, args.seed),
    }
    report.update(elapsed_seconds=time.monotonic() - start, peak_vram_bytes=torch.cuda.max_memory_allocated())
    copy_lock(output)
    write_json(output / "evaluation.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def compare(before_path, after_path, output):
    before, after = [json.loads(Path(path).read_text()) for path in (before_path, after_path)]
    for field in ("model", "revision", "seed", "decoding"):
        if before[field] != after[field]:
            raise ValueError(f"Evaluations differ in {field}; comparison would not be controlled")
    result = {}
    for wording in ("train", "heldout"):
        a, b = before["wordings"][wording], after["wordings"][wording]
        agreement_delta = b["optimal_action_agreement"] - a["optimal_action_agreement"]
        regret_delta = b["mean_regret_with_stand_fallback"] - a["mean_regret_with_stand_fallback"]
        result[wording] = dict(validity_delta=b["valid_output_rate"] - a["valid_output_rate"],
                               agreement_delta=agreement_delta, regret_delta=regret_delta,
                               improvement=(agreement_delta > 1e-9 or regret_delta < -1e-9)
                               and b["valid_output_rate"] >= a["valid_output_rate"])
    write_json(output, result)
    print(json.dumps(result, indent=2))
