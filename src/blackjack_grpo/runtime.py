"""Model loading and artifact helpers, imported only by GPU commands."""

import importlib.metadata
import json
from enum import Enum
from pathlib import Path
import shutil

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def encode(value):
        if isinstance(value, (set, frozenset)):
            return sorted(value)
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"Cannot serialize {type(value).__name__}")
    path.write_text(json.dumps(data, indent=2, allow_nan=False, default=encode) + "\n")


def environment_check():
    import torch

    result = {name: importlib.metadata.version(name) for name in
              ("torch", "transformers", "trl", "accelerate", "datasets", "peft", "gymnasium")}
    result.update(cuda_runtime=torch.version.cuda, cuda_available=torch.cuda.is_available())
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this laptop experiment")
    result.update(gpu=torch.cuda.get_device_name(0), architectures=torch.cuda.get_arch_list(),
                  bf16=torch.cuda.is_bf16_supported(), vram_bytes=torch.cuda.get_device_properties(0).total_memory)
    x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    (x @ x).float().mean().backward()
    torch.cuda.synchronize()
    result["bf16_forward_backward"] = "passed"
    if not result["bf16"]:
        raise RuntimeError("This configuration requires BF16 support")
    return result


def load_model(model_id=DEFAULT_MODEL, adapter=None, revision=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run blackjack-grpo check")
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to("cuda")
    if adapter:
        model = PeftModel.from_pretrained(model, str(adapter))
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def copy_lock(output):
    root = Path(__file__).resolve().parents[2]
    if (root / "uv.lock").exists():
        shutil.copy2(root / "uv.lock", Path(output) / "uv.lock")


def generate(model, tokenizer, messages, max_tokens=128):
    import torch

    rendered = [tokenizer.apply_chat_template(
        item, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    ) for item in messages]
    inputs = tokenizer(rendered, padding=True, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        sequences = model.generate(
            **inputs, max_new_tokens=max_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id, use_cache=True,
            temperature=None, top_p=None, top_k=None,
        )
    completions = sequences[:, inputs["input_ids"].shape[1]:]
    texts = tokenizer.batch_decode(completions, skip_special_tokens=True)
    lengths = []
    for ids in completions.tolist():
        # Include the EOS token, exclude padding beyond the first EOS.
        lengths.append(ids.index(tokenizer.eos_token_id) + 1 if tokenizer.eos_token_id in ids else len(ids))
    return list(zip(texts, lengths, strict=True))
