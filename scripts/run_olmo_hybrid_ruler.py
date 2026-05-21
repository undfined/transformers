#!/usr/bin/env python
# Copyright 2026 the HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Run RULER/NIAH-style prompts through OLMo Hybrid."""

from __future__ import annotations

import argparse
import importlib
import json
import re
from pathlib import Path

import torch

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig


def _import_olmo_mod():
    for mod_path in (
        "transformers.models.olmo_hybrid.modeling_olmo_hybrid",
        "transformers.models.olmo3_5_hybrid.modeling_olmo3_5_hybrid",
    ):
        try:
            return importlib.import_module(mod_path)
        except ImportError:
            continue
    raise ImportError("Cannot find olmo_hybrid or olmo3_5_hybrid module in this transformers install")


_FILLER = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
_HEADER = (
    "A special magic number is hidden within the following text. Make sure to memorize it. "
    "I will quiz you about the number afterwards."
)
DEFAULT_EXAMPLES_JSON = Path(__file__).with_name("olmo_hybrid_ruler_examples.json")


def build_ruler_prompt(key: str, value: str, before_needle: int, after_needle: int) -> str:
    needle = f"One of the special magic numbers for {key} is: {value}."
    question = f"What is the special magic number for {key} mentioned in the provided text?"
    completion_prefix = f"The special magic number for {key} mentioned in the provided text is"
    return "\n".join([_HEADER] + [_FILLER] * before_needle + [needle] + [_FILLER] * after_needle + [question, completion_prefix])


MAX_NEW_TOKENS = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model name or local path.")
    parser.add_argument("--examples-json", type=Path, default=DEFAULT_EXAMPLES_JSON)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, mps, etc.")
    parser.add_argument("--revision", help="Branch/tag/commit for --model.")
    parser.add_argument("--token", help="HF token, if needed. Usually HF_TOKEN env var is simpler.")
    parser.add_argument(
        "--fallback",
        choices=["force", "check", "auto"],
        default="force",
        help="'force' replaces FLA GDN callables with torch fallback; 'check' errors unless fallback is already active.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--no-cache", action="store_true", help="Run generate(use_cache=False).")
    parser.add_argument("--print-json", action="store_true")
    l2norm = parser.add_mutually_exclusive_group()
    l2norm.add_argument("--l2norm", dest="l2norm", action="store_true", default=None, help="Enable linear_use_qk_l2norm.")
    l2norm.add_argument("--no-l2norm", dest="l2norm", action="store_false", help="Disable linear_use_qk_l2norm (default).")
    return parser.parse_args()


def load_examples(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_examples = payload["examples"] if isinstance(payload, dict) else payload
    examples = []
    for raw_example in raw_examples:
        example = dict(raw_example)
        if "prompt" not in example:
            example["prompt"] = build_ruler_prompt(
                key=example["key"],
                value=example["expected"],
                before_needle=example["before_needle"],
                after_needle=example["after_needle"],
            )
        if "<<truncated" in example["prompt"]:
            raise ValueError(f"{example.get('name', example.get('key', '<unknown>'))} contains a truncation marker")
        examples.append(example)
    return examples


def pick_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_dtype(dtype: str) -> torch.dtype | str:
    if dtype == "auto":
        return "auto"
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]


def _normalize_rope_parameters(config) -> bool:
    """Ensure config.rope_parameters has a usable rope_theta so the model can construct.

    Returns True if a placeholder rope_theta was injected (no value was present in
    rope_parameters or as a legacy top-level attribute) — in that case the caller
    should disable rope after load via _disable_rope().
    """
    rope_params = getattr(config, "rope_parameters", None)
    if rope_params is None:
        rope_params = {}
        config.rope_parameters = rope_params
    rope_params.setdefault("rope_type", "default")
    if rope_params.get("rope_theta") is not None:
        return False
    top_level = getattr(config, "rope_theta", None)
    if top_level is not None:
        rope_params["rope_theta"] = top_level
        return False
    rope_params["rope_theta"] = 10000.0
    return True


def _disable_rope(model) -> int:
    """Zero out inv_freq on every rotary embedding so RoPE applies no rotation."""
    disabled = 0
    for module in model.modules():
        inv_freq = getattr(module, "inv_freq", None)
        if isinstance(inv_freq, torch.Tensor):
            inv_freq.zero_()
            disabled += 1
        original = getattr(module, "original_inv_freq", None)
        if isinstance(original, torch.Tensor):
            original.zero_()
    return disabled


def iter_linear_attn_modules(model):
    for name, module in model.named_modules():
        if hasattr(module, "chunk_gated_delta_rule") and hasattr(module, "recurrent_gated_delta_rule"):
            yield name, module


def callable_name(fn) -> str:
    module = getattr(fn, "__module__", type(fn).__module__)
    name = getattr(fn, "__qualname__", getattr(fn, "__name__", type(fn).__name__))
    return f"{module}.{name}"


def configure_fallback(model, mode: str) -> list[dict[str, str]]:
    olmo_mod = _import_olmo_mod()
    records = []
    for name, module in iter_linear_attn_modules(model):
        if mode == "force":
            module.chunk_gated_delta_rule = olmo_mod.torch_chunk_gated_delta_rule
            module.recurrent_gated_delta_rule = olmo_mod.torch_recurrent_gated_delta_rule

        chunk_name = callable_name(module.chunk_gated_delta_rule)
        recurrent_name = callable_name(module.recurrent_gated_delta_rule)
        records.append({"layer": name, "chunk": chunk_name, "recurrent": recurrent_name})

        if mode == "check":
            if module.chunk_gated_delta_rule is not olmo_mod.torch_chunk_gated_delta_rule:
                raise RuntimeError(f"{name} chunk GDN is not torch fallback: {chunk_name}")
            if module.recurrent_gated_delta_rule is not olmo_mod.torch_recurrent_gated_delta_rule:
                raise RuntimeError(f"{name} recurrent GDN is not torch fallback: {recurrent_name}")

    if not records:
        raise RuntimeError("No OLMo Hybrid linear-attention modules with GDN callables were found.")

    return records


def install_call_counters(model) -> dict[str, int]:
    calls = {"chunk": 0, "recurrent": 0}

    def wrap_chunk(fn):
        def counted(*args, **kwargs):
            calls["chunk"] += 1
            return fn(*args, **kwargs)
        return counted

    def wrap_recurrent(fn):
        def counted(*args, **kwargs):
            calls["recurrent"] += 1
            return fn(*args, **kwargs)
        return counted

    for _, module in iter_linear_attn_modules(model):
        module.chunk_gated_delta_rule = wrap_chunk(module.chunk_gated_delta_rule)
        module.recurrent_gated_delta_rule = wrap_recurrent(module.recurrent_gated_delta_rule)

    return calls


def run_one_example(args, model, tokenizer, device: torch.device, example: dict, gdn_calls: dict[str, int]) -> dict:
    calls_before = dict(gdn_calls)
    inputs = tokenizer([example["prompt"]], return_tensors="pt", return_token_type_ids=False).to(device)
    input_len = inputs["input_ids"].shape[-1]

    gen_config = GenerationConfig(
        do_sample=False,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=1,
        use_cache=not args.no_cache,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
    )
    with torch.no_grad():
        generated = model.generate(**inputs, generation_config=gen_config)

    new_tokens = generated[:, input_len:]
    continuation = tokenizer.decode(new_tokens[0], skip_special_tokens=True)
    full_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    expected = example["expected"]

    return {
        "name": example["name"],
        "key": example["key"],
        "expected": expected,
        "input_tokens": input_len,
        "new_tokens": new_tokens.shape[-1],
        "gdn_calls": {key: gdn_calls[key] - calls_before[key] for key in gdn_calls},
        "continuation": continuation,
        "full_text_suffix": full_text[-1000:],
        "exact_match": expected in continuation,
        "digit_match": re.sub(r"\D", "", expected) in re.sub(r"\D", "", continuation),
    }


def run_one(args, model_path: str, revision: str | None) -> dict:
    device = pick_device(args.device)
    dtype = pick_dtype(args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        revision=revision,
        token=args.token,
        trust_remote_code=args.trust_remote_code,
    )
    config = AutoConfig.from_pretrained(
        model_path,
        revision=revision,
        token=args.token,
        trust_remote_code=args.trust_remote_code,
    )
    rope_disabled = _normalize_rope_parameters(config)
    if args.l2norm is not None:
        config.linear_use_qk_l2norm = args.l2norm
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        revision=revision,
        token=args.token,
        trust_remote_code=args.trust_remote_code,
    )
    if rope_disabled:
        _disable_rope(model)
    if dtype != "auto":
        model = model.to(dtype=dtype)
    model = model.to(device)
    model.eval()

    fallback_records = configure_fallback(model, args.fallback)
    gdn_calls = install_call_counters(model)
    examples = [run_one_example(args, model, tokenizer, device, example, gdn_calls) for example in load_examples(args.examples_json)]

    return {
        "model": model_path,
        "revision": revision,
        "device": str(device),
        "dtype": str(dtype),
        "l2norm": getattr(model.config, "linear_use_qk_l2norm", None),
        "max_new_tokens": args.max_new_tokens,
        "total_gdn_calls": gdn_calls,
        "fallback_layers": fallback_records,
        "num_examples": len(examples),
        "exact_matches": sum(example["exact_match"] for example in examples),
        "digit_matches": sum(example["digit_match"] for example in examples),
        "examples": examples,
    }


def print_result(result: dict) -> None:
    label = result["model"]
    if result["revision"]:
        label += f"@{result['revision']}"
    print(f"model:        {label}")
    print(f"device:       {result['device']}")
    print(f"dtype:        {result['dtype']}")
    print(f"l2norm:       {result['l2norm']}")
    print(f"examples:     {result['num_examples']}")
    print(f"exact match:  {result['exact_matches']}/{result['num_examples']}")
    print(f"digit match:  {result['digit_matches']}/{result['num_examples']}")
    print(f"GDN calls:    {result['total_gdn_calls']}")
    for example in result["examples"]:
        print(f"\n--- {example['name']} ({example['key']}) ---")
        print(f"input tokens: {example['input_tokens']}")
        print(f"new tokens:   {example['new_tokens']}")
        print(f"GDN calls:    {example['gdn_calls']}")
        print(f"expected:     {example['expected']}")
        print(f"exact match:  {example['exact_match']}")
        print(f"digit match:  {example['digit_match']}")
        print("continuation:")
        print(example["continuation"])


def main() -> None:
    args = parse_args()
    result = run_one(args, args.model, args.revision)

    if args.print_json:
        print(json.dumps(result, indent=2))
        return

    print_result(result)


if __name__ == "__main__":
    main()
