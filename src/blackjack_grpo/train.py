import json
import math
from pathlib import Path
import signal
import time

from .game import states
from .prompts import blackjack_reward, parse_completion, prompt
from .runtime import copy_lock, environment_check, load_model, write_json


def train(args):
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import TrainerCallback, set_seed
    from transformers.trainer_utils import get_last_checkpoint
    from trl import GRPOConfig, GRPOTrainer

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    resume = args.resume
    if resume == "auto":
        resume = get_last_checkpoint(str(output))
        if resume is None:
            raise ValueError(f"No resumable checkpoint in {output}")
    if resume:
        saved_step = json.loads((Path(resume) / "trainer_state.json").read_text())["global_step"]
        if args.steps <= saved_step:
            raise ValueError(f"--steps must exceed the checkpoint's total of {saved_step}")
    if not resume and ((output / "final").exists() or get_last_checkpoint(str(output))):
        raise ValueError("Output already contains a run; use --resume auto or choose another --output")
    environment = environment_check()
    write_json(output / "environment.json", environment)
    copy_lock(output)
    set_seed(args.seed)
    model, tokenizer = load_model(args.model, revision=args.revision)
    rows = [dict(prompt=prompt(s), total=s.total, dealer=s.dealer, usable_ace=s.usable_ace)
            for _ in range(32) for s in states()]
    for row in rows[:280]:
        token_ids = tokenizer.apply_chat_template(row["prompt"], enable_thinking=False, add_generation_prompt=True)
        if len(token_ids) > 256:
            raise ValueError("Prompt exceeds the 256-token input budget")
    dataset = Dataset.from_list(rows)
    run_name = f"session-{time.time_ns()}"

    class Monitor(TrainerCallback):
        def __init__(self):
            self.start = None
            self.interrupted = False
            self.stop_reason = "max_steps"

        def on_train_begin(self, args, state, control, **kwargs):
            self.start = time.monotonic()
            # on_train_begin runs after checkpoint loading, so a resume measures
            # changes during this invocation rather than from initial LoRA zeros.
            self.before = {name: parameter.detach().float().cpu().clone()
                           for name, parameter in kwargs["model"].named_parameters() if parameter.requires_grad}
            return control

        def on_step_end(self, args, state, control, **kwargs):
            elapsed = time.monotonic() - self.start
            if self.interrupted or elapsed >= minutes * 60 or state.global_step >= args.max_steps:
                self.stop_reason = "interrupt" if self.interrupted else ("time_limit" if elapsed >= minutes * 60 else "max_steps")
                control.should_training_stop = True
                control.should_save = True
            return control

        def on_log(self, args, state, control, logs=None, **kwargs):
            data = {"step": state.global_step, "elapsed_seconds": time.monotonic() - self.start, **(logs or {})}
            for key in ("loss", "grad_norm"):
                if key in data and not math.isfinite(float(data[key])):
                    raise RuntimeError(f"Non-finite {key}: {data[key]}")
            # TRL can log undefined descriptive statistics (for example, mean
            # length of EOS-terminated responses when all samples were cut off).
            # Preserve those as null while rejecting non-finite loss/gradients.
            data = {key: None if isinstance(value, float) and not math.isfinite(value) else value
                    for key, value in data.items()}
            with (output / f"{run_name}-metrics.jsonl").open("a") as file:
                file.write(json.dumps(data, allow_nan=False) + "\n")
            return control

    minutes = args.minutes
    monitor = Monitor()

    def recorded_reward(completions, total, dealer, usable_ace, trainer_state=None, **kwargs):
        values = blackjack_reward(completions, total, dealer, usable_ace, **kwargs)
        step = trainer_state.global_step if trainer_state else 0
        if step % 10 == 0 or args.steps <= 5:
            with (output / f"{run_name}-samples.jsonl").open("a") as file:
                for i, score in enumerate(values[:8]):
                    text = completions[i][0]["content"] if isinstance(completions[i], list) else completions[i]
                    file.write(json.dumps(dict(step=step, total=total[i], dealer=dealer[i],
                                               usable_ace=usable_ace[i], completion=text,
                                               valid=parse_completion(text) is not None, reward=score)) + "\n")
        return values

    config = GRPOConfig(
        output_dir=str(output), max_steps=args.steps, per_device_train_batch_size=args.microbatch,
        gradient_accumulation_steps=16 // args.microbatch, num_generations=4,
        max_completion_length=128, learning_rate=2e-5, lr_scheduler_type="constant", beta=0.02, loss_type="dapo",
        temperature=0.8, top_p=1.0, top_k=0, bf16=True, fp16=False,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        chat_template_kwargs={"enable_thinking": False}, use_vllm=False,
        logging_steps=1, logging_first_step=True, logging_nan_inf_filter=False,
        save_steps=50, save_total_limit=3, save_only_model=False,
        report_to="none", push_to_hub=False, seed=args.seed, data_seed=args.seed,
        disable_tqdm=True, dataloader_num_workers=0, optim="adamw_torch",
    )
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules="all-linear", task_type="CAUSAL_LM")
    trainer = GRPOTrainer(model=model, processing_class=tokenizer, reward_funcs=recorded_reward,
                          args=config, train_dataset=dataset, peft_config=lora, callbacks=[monitor])
    write_json(output / f"{run_name}-config.json", dict(
        model=args.model, revision=getattr(model.config, "_commit_hash", args.revision),
        minutes=minutes, resumed_from=str(resume) if resume else None,
        grpo=config.to_dict(), lora=lora.to_dict(), reachable_states=len(states()),
        training_reward="Q*(observation, action); malformed=-2",
    ))
    torch.cuda.reset_peak_memory_stats()
    prior_signal = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, lambda *_: setattr(monitor, "interrupted", True))
    try:
        result = trainer.train(resume_from_checkpoint=resume)
    finally:
        signal.signal(signal.SIGINT, prior_signal)
    trainer.save_model(str(output / "final"))
    tokenizer.save_pretrained(str(output / "final"))
    trainer.state.save_to_json(str(output / "final" / "trainer_state.json"))
    before = monitor.before
    delta_squared = sum(float((parameter.detach().float().cpu() - before[name]).square().sum())
                        for name, parameter in trainer.model.named_parameters() if name in before)
    summary = dict(global_step=trainer.state.global_step, elapsed_seconds=time.monotonic() - monitor.start,
                   stop_reason=monitor.stop_reason, peak_vram_bytes=torch.cuda.max_memory_allocated(),
                   trainable_parameters=sum(value.numel() for value in before.values()),
                   adapter_delta_l2=math.sqrt(delta_squared), metrics=result.metrics,
                   final_adapter=str(output / "final"), checkpoint=get_last_checkpoint(str(output)))
    write_json(output / f"{run_name}-summary.json", summary)
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    if summary["adapter_delta_l2"] == 0:
        raise RuntimeError("No adapter change was observed; inspect reward variance and completions")
    return summary
