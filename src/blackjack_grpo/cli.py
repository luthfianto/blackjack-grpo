import argparse
import json
from pathlib import Path
import subprocess
import sys

from .runtime import DEFAULT_MODEL


def main():
    parser = argparse.ArgumentParser(description="Local Blackjack LoRA + GRPO experiment")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="Verify packages, CUDA and BF16 forward/backward")
    download = commands.add_parser("download", help="Download model weights to the local HF cache")
    for command in ("train", "smoke", "evaluate", "play"):
        item = commands.add_parser(command)
        item.add_argument("--model", default=DEFAULT_MODEL)
        item.add_argument("--revision", default=None)
        item.add_argument("--seed", type=int, default=42)
        if command in ("train", "smoke"):
            item.add_argument("--output", default=f"runs/{command}")
            item.add_argument("--steps", type=int, default=5 if command == "smoke" else 300)
            item.add_argument("--minutes", type=float, default=45)
            item.add_argument("--microbatch", type=int, choices=(1, 2), default=2)
            item.add_argument("--resume", default=None, help="Checkpoint directory or auto")
        else:
            item.add_argument("--adapter", type=Path, default=None)
        if command == "evaluate":
            item.add_argument("--output", default="runs/evaluation")
            item.add_argument("--batch-size", type=int, default=8)
            item.add_argument("--episodes", type=int, default=100_000)
        if command == "play":
            item.add_argument("--total", type=int)
            item.add_argument("--dealer", type=int)
            item.add_argument("--usable-ace", action="store_true")
    download.add_argument("--model", default=DEFAULT_MODEL)
    download.add_argument("--revision", default=None)
    comparison = commands.add_parser("compare")
    comparison.add_argument("before", type=Path)
    comparison.add_argument("after", type=Path)
    comparison.add_argument("--output", default="runs/comparison.json")
    args = parser.parse_args()
    if args.command in ("train", "smoke") and (args.steps <= 0 or args.minutes <= 0):
        parser.error("steps and minutes must be positive")
    if args.command == "evaluate" and (args.batch_size <= 0 or args.episodes < 2):
        parser.error("batch-size must be positive and episodes must be at least 2")
    if args.command == "check":
        from .runtime import environment_check
        print(json.dumps(environment_check(), indent=2))
    elif args.command == "download":
        from huggingface_hub import snapshot_download
        path = snapshot_download(args.model, revision=args.revision,
                                 allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja"])
        print(path)
    elif args.command == "smoke":
        # A fresh process on OOM ensures CUDA allocations and trainer state are released.
        base = [sys.executable, "-m", "blackjack_grpo", "train", "--model", args.model,
                "--output", args.output, "--steps", str(args.steps), "--minutes", str(args.minutes),
                "--seed", str(args.seed)]
        if args.revision:
            base += ["--revision", args.revision]
        if args.resume:
            base += ["--resume", args.resume]
        status = subprocess.run(base + ["--microbatch", str(args.microbatch)]).returncode
        if status and args.microbatch == 2 and (Path(args.output) / "oom.json").exists() and not args.resume:
            print("OOM: retrying with microbatch 1 and accumulation 16", flush=True)
            (Path(args.output) / "oom.json").unlink()
            status = subprocess.run(base + ["--microbatch", "1"]).returncode
        raise SystemExit(status)
    elif args.command == "train":
        import torch
        from .train import train
        try:
            train(args)
        except torch.cuda.OutOfMemoryError:
            # Cover model loading and trainer initialization as well as the
            # training loop, so the smoke subprocess can retry any CUDA OOM.
            output = Path(args.output)
            output.mkdir(parents=True, exist_ok=True)
            (output / "oom.json").write_text(json.dumps({"microbatch": args.microbatch}) + "\n")
            raise
    elif args.command == "evaluate":
        from .evaluate import evaluate
        evaluate(args)
    elif args.command == "compare":
        from .evaluate import compare
        compare(args.before, args.after, args.output)
    elif args.command == "play":
        from .game import State
        from .prompts import prompt
        from .runtime import generate, load_model
        if (args.total is None) != (args.dealer is None):
            parser.error("provide both --total and --dealer")
        model, tokenizer = load_model(args.model, args.adapter, args.revision)
        model.eval()
        if args.total is not None:
            print(generate(model, tokenizer, [prompt(State(args.total, args.dealer, args.usable_ace))])[0][0])
        else:
            print("Enter: player_total dealer_upcard usable_ace(0/1), or quit", flush=True)
            while True:
                try:
                    line = input("> ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if line.lower() in ("quit", "exit", "q"):
                    break
                try:
                    total, dealer, soft = map(int, line.split())
                    if soft not in (0, 1):
                        raise ValueError("usable_ace must be 0 or 1")
                    print(generate(model, tokenizer, [prompt(State(total, dealer, bool(soft)))])[0][0])
                except ValueError as error:
                    print(error)
