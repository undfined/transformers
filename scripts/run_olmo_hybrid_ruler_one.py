#!/usr/bin/env python
# Copyright 2026 the HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Run one RULER/NIAH-style prompt through OLMo Hybrid with Transformers."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.olmo_hybrid import modeling_olmo_hybrid as olmo_hybrid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model name or local path.")
    prompt = parser.add_mutually_exclusive_group(required=True)
    prompt.add_argument(
        "--example",
        choices=["madly-packet-5449368"],
        help="Built-in full RULER/NIAH-style repro prompt.",
    )
    prompt.add_argument("--prompt", help="Prompt text.")
    prompt.add_argument("--prompt-file", type=Path, help="Path to a text prompt file.")
    prompt.add_argument("--json-file", type=Path, help="Path to a JSON eval record.")
    prompt.add_argument("--jsonl-file", type=Path, help="Path to a RULER/eval JSONL file.")
    parser.add_argument("--jsonl-index", type=int, default=0, help="Zero-based record index for --jsonl-file.")
    parser.add_argument("--prompt-key", help="Dotted prompt key. Defaults to request.context, then input.")
    parser.add_argument("--expected-key", help="Dotted expected key. Defaults to label, then outputs.")
    parser.add_argument("--expected", help="Expected needle/answer text, e.g. 5449368.")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, mps, etc.")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--revision")
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
    return parser.parse_args()


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
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype]


def get_nested(record, dotted_key: str):
    value = record
    for part in dotted_key.split("."):
        if isinstance(value, list):
            value = value[int(part)]
        else:
            value = value[part]
    return value


def first_present(record, keys: list[str]):
    for key in keys:
        try:
            return get_nested(record, key)
        except (KeyError, IndexError, TypeError, ValueError):
            pass
    raise KeyError(f"None of these keys were present: {keys}")


def normalize_expected(raw_expected) -> str:
    if isinstance(raw_expected, list):
        return str(raw_expected[0])
    return str(raw_expected)


def record_to_prompt_expected(record: dict, args: argparse.Namespace) -> tuple[str, str | None, int | None]:
    prompt_keys = [args.prompt_key] if args.prompt_key else ["request.context", "input"]
    expected_keys = [args.expected_key] if args.expected_key else ["label", "outputs"]

    prompt = str(first_present(record, prompt_keys))
    expected = args.expected
    if expected is None:
        try:
            expected = normalize_expected(first_present(record, expected_keys))
        except KeyError:
            expected = None

    max_new_tokens = None
    try:
        max_new_tokens = int(get_nested(record, "request.generation_kwargs.max_gen_toks"))
    except (KeyError, TypeError, ValueError):
        pass

    return prompt, expected, max_new_tokens


def build_madly_packet_example() -> tuple[str, str, int]:
    filler = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
    needle = "One of the special magic numbers for madly-packet is: 5449368."
    context = "\n".join([filler] * 70 + [needle] + [filler] * 95)
    prompt = (
        "A special magic number is hidden within the following text. Make sure to memorize it. "
        "I will quiz you about the number afterwards.\n"
        f"{context}\n"
        "What is the special magic number for madly-packet mentioned in the provided text?\n"
        "The special magic number for madly-packet mentioned in the provided text is"
    )
    return prompt, "5449368", 50


def read_prompt_and_expected(args: argparse.Namespace) -> tuple[str, str | None, int | None]:
    if args.example == "madly-packet-5449368":
        return build_madly_packet_example()

    expected = args.expected
    if args.json_file is not None:
        record = json.loads(args.json_file.read_text(encoding="utf-8"))
        return record_to_prompt_expected(record, args)

    if args.jsonl_file is not None:
        with args.jsonl_file.open(encoding="utf-8") as f:
            for index, line in enumerate(f):
                if index == args.jsonl_index:
                    record = json.loads(line)
                    return record_to_prompt_expected(record, args)
        raise ValueError(f"{args.jsonl_file} has no record at index {args.jsonl_index}")

    if args.prompt_file is not None:
        return args.prompt_file.read_text(encoding="utf-8"), expected, None
    return args.prompt, expected, None


def iter_linear_attn_modules(model):
    for name, module in model.named_modules():
        if hasattr(module, "chunk_gated_delta_rule") and hasattr(module, "recurrent_gated_delta_rule"):
            yield name, module


def callable_name(fn) -> str:
    module = getattr(fn, "__module__", type(fn).__module__)
    name = getattr(fn, "__qualname__", getattr(fn, "__name__", type(fn).__name__))
    return f"{module}.{name}"


def configure_fallback(model, mode: str) -> list[dict[str, str]]:
    records = []
    for name, module in iter_linear_attn_modules(model):
        if mode == "force":
            module.chunk_gated_delta_rule = olmo_hybrid.torch_chunk_gated_delta_rule
            module.recurrent_gated_delta_rule = olmo_hybrid.torch_recurrent_gated_delta_rule

        chunk_name = callable_name(module.chunk_gated_delta_rule)
        recurrent_name = callable_name(module.recurrent_gated_delta_rule)
        records.append({"layer": name, "chunk": chunk_name, "recurrent": recurrent_name})

        if mode == "check":
            if module.chunk_gated_delta_rule is not olmo_hybrid.torch_chunk_gated_delta_rule:
                raise RuntimeError(f"{name} chunk GDN is not torch fallback: {chunk_name}")
            if module.recurrent_gated_delta_rule is not olmo_hybrid.torch_recurrent_gated_delta_rule:
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


def only_digits(text: str) -> str:
    return re.sub(r"\D", "", text)


def main() -> None:
    args = parse_args()
    prompt, expected, record_max_new_tokens = read_prompt_and_expected(args)
    if "<<truncated" in prompt:
        raise ValueError(
            "The selected prompt contains a literal truncation marker. Use the original untruncated request/data file "
            "or --example madly-packet-5449368."
        )
    max_new_tokens = record_max_new_tokens if record_max_new_tokens is not None else args.max_new_tokens
    device = pick_device(args.device)
    dtype = pick_dtype(args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        token=args.token,
        trust_remote_code=args.trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        attn_implementation=args.attn_implementation,
        revision=args.revision,
        token=args.token,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    )
    model.to(device)
    model.eval()

    fallback_records = configure_fallback(model, args.fallback)
    gdn_calls = install_call_counters(model)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=not args.no_cache,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        )

    new_tokens = generated[:, input_len:]
    continuation = tokenizer.decode(new_tokens[0], skip_special_tokens=True)
    full_text = tokenizer.decode(generated[0], skip_special_tokens=True)

    result = {
        "model": args.model,
        "device": str(device),
        "dtype": str(dtype),
        "input_tokens": input_len,
        "new_tokens": new_tokens.shape[-1],
        "max_new_tokens": max_new_tokens,
        "gdn_calls": gdn_calls,
        "fallback_layers": fallback_records,
        "continuation": continuation,
    }

    if expected is not None:
        result["expected"] = expected
        result["exact_in_continuation"] = expected in continuation
        result["expected_digits"] = only_digits(expected)
        result["continuation_digits"] = only_digits(continuation)
        result["digit_match"] = only_digits(expected) in only_digits(continuation)

    if args.print_json:
        print(json.dumps(result, indent=2))
        return

    print(f"model: {args.model}")
    print(f"device: {device}")
    print(f"dtype: {dtype}")
    print(f"input tokens: {input_len}")
    print(f"new tokens: {new_tokens.shape[-1]}")
    print(f"max new tokens: {max_new_tokens}")
    print(f"GDN calls: {gdn_calls}")
    print("first fallback layer:")
    print(f"  {fallback_records[0]['layer']}")
    print(f"  chunk: {fallback_records[0]['chunk']}")
    print(f"  recurrent: {fallback_records[0]['recurrent']}")
    if expected is not None:
        print(f"expected: {expected}")
        print(f"exact in continuation: {result['exact_in_continuation']}")
        print(f"digit match: {result['digit_match']}")
        print(f"continuation digits: {result['continuation_digits']}")
    print("\n--- continuation ---")
    print(continuation)
    print("--- full text suffix ---")
    print(full_text[-1000:])


if __name__ == "__main__":
    main()
