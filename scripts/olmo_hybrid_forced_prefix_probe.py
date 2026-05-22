#!/usr/bin/env python
# Copyright 2026 the HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Probe OLMo Hybrid full-vs-cached next-token logits after a forced prefix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import run_olmo_hybrid_ruler as ruler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", nargs="?", help="Model name or local path.")
    parser.add_argument("--model", dest="model", help="Model name or local path.")
    parser.add_argument(
        "--tokenizer",
        help="Tokenizer name or local path. Defaults to --model; useful for checking tokenizer packaging issues.",
    )
    parser.add_argument("--prompt", help="Prompt text. Mutually exclusive with --prompt-file.")
    parser.add_argument("--prompt-file", type=Path, help="Plain-text file containing the prompt.")
    parser.add_argument(
        "--examples-json",
        type=Path,
        default=ruler.DEFAULT_EXAMPLES_JSON,
        help="RULER examples JSON. Defaults to the same file used by scripts/run_olmo_hybrid_ruler.py.",
    )
    parser.add_argument("--example-name", help="Example name from --examples-json.")
    parser.add_argument(
        "--example-index",
        type=int,
        default=0,
        help="Example index from --examples-json. Defaults to the shared failing doc0 case.",
    )
    parser.add_argument("--expected", help="Expected answer. Defaults to the selected example's expected field.")
    parser.add_argument(
        "--forced-prefix",
        help=(
            "Known answer prefix to append before probing the next-token logits. "
            "If omitted, the prefix is built from the selected tokenizer's target-answer tokens."
        ),
    )
    parser.add_argument(
        "--continuation",
        help=(
            "Generated continuation text to compare against the expected answer in --prefix-mode divergence. "
            "If omitted, the script runs greedy generation with the same basic settings as run_olmo_hybrid_ruler.py."
        ),
    )
    parser.add_argument(
        "--prefix-mode",
        choices=["divergence", "tokens", "chars"],
        default="divergence",
        help=(
            "'divergence' probes the first token where the generated continuation differs from the expected answer. "
            "'tokens' uses a tokenizer-valid prefix of the expected answer. "
            "'chars' preserves the old leading-space plus first --prefix-chars behavior."
        ),
    )
    parser.add_argument("--prefix-chars", type=int, default=5)
    parser.add_argument(
        "--prefix-token-count",
        type=int,
        help=(
            "Number of target-answer tokens to force in --prefix-mode tokens. "
            "Defaults to all but the final target-answer token."
        ),
    )
    parser.add_argument(
        "--expected-next",
        help="Expected next-token text. Defaults to the suffix of --expected after --forced-prefix.",
    )
    parser.add_argument(
        "--bad-next",
        default=".",
        help="Known bad next-token text to compare against. Use an empty string to omit it.",
    )
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="Extra next-token candidate text. May be passed more than once.",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--diff-top-k", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=ruler.MAX_NEW_TOKENS)
    parser.add_argument("--no-cache", action="store_true", help="Run generate(use_cache=False) in divergence mode.")
    parser.add_argument("--print-json", action="store_true")

    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, mps, etc.")
    parser.add_argument("--revision", help="Branch/tag/commit for --model.")
    parser.add_argument("--token", help="HF token, if needed. Usually HF_TOKEN env var is simpler.")
    parser.add_argument("--attn-implementation", help="Full-attention backend to request, e.g. eager or sdpa.")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--rope",
        choices=["auto", "disable", "default", "config"],
        default="config",
        help="RoPE handling, matching scripts/run_olmo_hybrid_ruler.py.",
    )
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument(
        "--add-special-tokens",
        choices=["default", "true", "false"],
        default="default",
        help="Pass add_special_tokens to the prompt tokenizer call, or leave tokenizer default unchanged.",
    )
    parser.add_argument(
        "--fallback",
        choices=["force", "check", "auto"],
        default="force",
        help="'force' replaces FLA GDN callables with torch fallback.",
    )
    parser.add_argument("--torch-conv", action="store_true", help="Replace short conv modules with torch fallback.")
    parser.add_argument(
        "--torch-gated-norm",
        action="store_true",
        help="Replace gated RMSNorm modules with torch fallback.",
    )
    args = parser.parse_args()
    args.model = args.model or args.model_path
    if args.model is None:
        parser.error("Pass a model path positionally or with --model.")
    if args.prompt and args.prompt_file:
        parser.error("--prompt and --prompt-file are mutually exclusive.")
    return args


def resolve_case(args: argparse.Namespace) -> dict:
    example = None
    if args.prompt_file is not None:
        prompt = args.prompt_file.read_text(encoding="utf-8")
        name = str(args.prompt_file)
    elif args.prompt is not None:
        prompt = args.prompt
        name = "inline_prompt"
    else:
        examples = ruler.load_examples(args.examples_json)
        if args.example_name is not None:
            matches = [item for item in examples if item.get("name") == args.example_name]
            if not matches:
                raise ValueError(f"No example named {args.example_name!r} in {args.examples_json}")
            example = matches[0]
        else:
            if args.example_index < 0 or args.example_index >= len(examples):
                raise ValueError(f"--example-index must be in [0, {len(examples) - 1}]")
            example = examples[args.example_index]
        prompt = example["prompt"]
        name = example.get("name", f"example_{args.example_index}")

    expected = args.expected if args.expected is not None else (example or {}).get("expected")
    return {
        "name": name,
        "prompt": prompt,
        "expected": expected,
    }


def token_texts(tokenizer, token_ids: list[int]) -> list[str]:
    return [tokenizer.decode([int(token_id)]) for token_id in token_ids]


def encode_continuation(tokenizer, text: str, device):
    token_ids = tokenizer(
        text,
        return_tensors="pt",
        return_token_type_ids=False,
        add_special_tokens=False,
    ).input_ids.to(device)
    if token_ids.shape[-1] == 0:
        raise ValueError(f"Text {text!r} tokenized to no tokens")
    return token_ids


def tensor_from_token_ids(token_ids: list[int], device):
    torch = ruler.torch
    return torch.tensor([token_ids], dtype=torch.long, device=device)


def first_token_divergence(left: list[int], right: list[int]) -> int | None:
    for index, (left_token, right_token) in enumerate(zip(left, right)):
        if left_token != right_token:
            return index
    if len(left) == len(right):
        return None
    return min(len(left), len(right))


def generate_continuation(args: argparse.Namespace, model, tokenizer, inputs) -> str:
    gen_config = ruler.GenerationConfig(
        do_sample=False,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=1,
        use_cache=not args.no_cache,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
    )
    generated = model.generate(**inputs, generation_config=gen_config)
    input_len = inputs["input_ids"].shape[-1]
    new_tokens = generated[:, input_len:]
    return tokenizer.decode(new_tokens[0], skip_special_tokens=True)


def build_token_candidate(tokenizer, label: str, token_id: int) -> dict:
    token_text = tokenizer.decode([int(token_id)])
    return {
        "label": label,
        "text": token_text,
        "token_ids": [int(token_id)],
        "scored_token_id": int(token_id),
        "scored_token_text": token_text,
        "uses_first_token_only": False,
    }


def resolve_prefix_and_candidates(
    args: argparse.Namespace,
    tokenizer,
    case: dict,
    device,
    model=None,
    inputs=None,
) -> dict:
    expected = case["expected"]
    target_text = f" {expected}" if expected is not None else None
    target_ids = encode_continuation(tokenizer, target_text, device) if target_text is not None else None

    if args.forced_prefix is not None:
        prefix_text = args.forced_prefix
        prefix_ids = encode_continuation(tokenizer, prefix_text, device)
        expected_next = args.expected_next
        if expected_next is None and expected is not None:
            stripped_prefix = prefix_text.strip()
            if expected.startswith(stripped_prefix):
                expected_next = expected[len(stripped_prefix) :]
        prefix_mode = "explicit"
        candidates = build_candidate_infos(tokenizer, [candidate for candidate in (expected_next, args.bad_next, *args.candidate) if candidate])
    elif args.prefix_mode == "chars":
        if expected is None:
            raise ValueError("Pass --forced-prefix, or pass/choose an example with --expected.")
        prefix_text = " " + expected[: args.prefix_chars]
        prefix_ids = encode_continuation(tokenizer, prefix_text, device)
        expected_next = args.expected_next or expected[args.prefix_chars :]
        prefix_mode = "chars"
        candidates = build_candidate_infos(tokenizer, [candidate for candidate in (expected_next, args.bad_next, *args.candidate) if candidate])
    elif args.prefix_mode == "tokens":
        if target_ids is None:
            raise ValueError("Pass --forced-prefix, or pass/choose an example with --expected.")
        target_len = target_ids.shape[-1]
        prefix_token_count = args.prefix_token_count if args.prefix_token_count is not None else target_len - 1
        if prefix_token_count < 0 or prefix_token_count >= target_len:
            raise ValueError(f"--prefix-token-count must be in [0, {target_len - 1}] for this target answer.")
        prefix_ids = target_ids[:, :prefix_token_count]
        next_token_id = int(target_ids[0, prefix_token_count].item())
        prefix_text = tokenizer.decode(prefix_ids[0].tolist()) if prefix_token_count else ""
        expected_next = args.expected_next or tokenizer.decode([next_token_id])
        prefix_mode = "tokens"
        candidates = build_candidate_infos(tokenizer, [candidate for candidate in (expected_next, args.bad_next, *args.candidate) if candidate])
    else:
        if target_ids is None:
            raise ValueError("Pass/choose an example with --expected for --prefix-mode divergence.")
        if args.continuation is None:
            if model is None or inputs is None:
                raise ValueError("Internal error: divergence mode needs model and inputs when --continuation is omitted.")
            continuation = generate_continuation(args, model, tokenizer, inputs)
        else:
            continuation = args.continuation
        generated_text = ruler.first_answer_line(continuation)
        target_token_ids = [int(token_id) for token_id in target_ids[0].tolist()]
        generated_token_ids = tokenizer(generated_text, add_special_tokens=False).input_ids
        divergence = first_token_divergence(target_token_ids, generated_token_ids)
        target_is_generated_prefix = (
            len(generated_token_ids) >= len(target_token_ids)
            and generated_token_ids[: len(target_token_ids)] == target_token_ids
        )
        if divergence is None or target_is_generated_prefix:
            return {
                "mode": "divergence",
                "status": "no_divergence",
                "text": None,
                "ids": None,
                "tokens": None,
                "expected_next": None,
                "target_text": target_text,
                "target_token_ids": target_token_ids,
                "target_tokens": token_texts(tokenizer, target_token_ids),
                "generated_continuation": continuation,
                "generated_answer_text": generated_text,
                "generated_token_ids": generated_token_ids,
                "generated_tokens": token_texts(tokenizer, generated_token_ids),
                "target_is_generated_prefix": target_is_generated_prefix,
                "candidates": [],
            }

        prefix_token_ids = target_token_ids[:divergence]
        prefix_ids = tensor_from_token_ids(prefix_token_ids, device)
        prefix_text = tokenizer.decode(prefix_token_ids) if prefix_token_ids else ""
        expected_next_token_id = target_token_ids[divergence]
        candidates = [build_token_candidate(tokenizer, "expected", expected_next_token_id)]
        if divergence < len(generated_token_ids):
            candidates.append(build_token_candidate(tokenizer, "generated", generated_token_ids[divergence]))
        for extra_candidate in (args.bad_next, *args.candidate):
            if extra_candidate:
                candidates.extend(build_candidate_infos(tokenizer, [extra_candidate]))
        expected_next = tokenizer.decode([expected_next_token_id])
        prefix_mode = "divergence"

    return {
        "mode": prefix_mode,
        "status": "probe",
        "text": prefix_text,
        "ids": prefix_ids,
        "tokens": token_texts(tokenizer, prefix_ids[0].tolist()),
        "expected_next": expected_next,
        "target_text": target_text,
        "target_token_ids": [int(token_id) for token_id in target_ids[0].tolist()] if target_ids is not None else None,
        "target_tokens": token_texts(tokenizer, target_ids[0].tolist()) if target_ids is not None else None,
        "generated_continuation": None,
        "generated_answer_text": None,
        "generated_token_ids": None,
        "generated_tokens": None,
        "target_is_generated_prefix": None,
        "candidates": candidates,
    }


def build_candidate_infos(tokenizer, candidate_texts: list[str]) -> list[dict]:
    infos = []
    seen_token_ids = set()
    for text in candidate_texts:
        token_ids = tokenizer(text, add_special_tokens=False).input_ids
        if not token_ids:
            raise ValueError(f"Candidate {text!r} tokenized to no tokens")
        first_id = int(token_ids[0])
        if first_id in seen_token_ids:
            continue
        seen_token_ids.add(first_id)
        infos.append(
            {
                "label": text,
                "text": text,
                "token_ids": [int(token_id) for token_id in token_ids],
                "scored_token_id": first_id,
                "scored_token_text": tokenizer.decode([first_id]),
                "uses_first_token_only": len(token_ids) > 1,
            }
        )
    return infos


def score_logits(tokenizer, logits, candidate_infos: list[dict], top_k: int) -> dict:
    torch = ruler.torch
    log_probs = logits.float().log_softmax(dim=-1)
    candidates = {}
    for info in candidate_infos:
        token_id = torch.tensor([[info["scored_token_id"]]], device=log_probs.device)
        token_logprob, rank = ruler.logprob_rank(log_probs, token_id)
        candidates[info["label"]] = {
            "token_id": info["scored_token_id"],
            "text": info["text"],
            "token_text": info["scored_token_text"],
            "logprob": token_logprob,
            "rank": rank,
        }

    top = []
    if top_k > 0:
        top_logprobs, top_indices = torch.topk(log_probs, k=min(top_k, log_probs.shape[-1]), dim=-1)
        for rank, (token_id, logprob) in enumerate(zip(top_indices[0].tolist(), top_logprobs[0].tolist()), start=1):
            top.append(
                {
                    "rank": rank,
                    "token_id": int(token_id),
                    "token_text": tokenizer.decode([int(token_id)]),
                    "logprob": float(logprob),
                }
            )
    return {"candidates": candidates, "top": top}


def input_ids_and_mask(inputs, prefix_ids):
    torch = ruler.torch
    input_ids = torch.cat([inputs["input_ids"], prefix_ids], dim=-1)
    if "attention_mask" in inputs:
        prefix_mask = torch.ones_like(prefix_ids, device=inputs["attention_mask"].device)
        attention_mask = torch.cat([inputs["attention_mask"], prefix_mask], dim=-1)
    else:
        attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


def full_logits(model, inputs, prefix_ids, use_cache: bool):
    input_ids, attention_mask = input_ids_and_mask(inputs, prefix_ids)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=use_cache)
    return outputs.logits[:, -1, :]


def cached_chunk_logits(model, inputs, prefix_ids):
    outputs = model(**inputs, use_cache=True)
    logits = outputs.logits[:, -1, :]
    if prefix_ids.shape[-1] > 0:
        outputs = model(input_ids=prefix_ids, past_key_values=outputs.past_key_values, use_cache=True)
        logits = outputs.logits[:, -1, :]
    return logits


def cached_single_logits(model, inputs, prefix_ids):
    outputs = model(**inputs, use_cache=True)
    logits = outputs.logits[:, -1, :]
    past_key_values = outputs.past_key_values
    for index in range(prefix_ids.shape[-1]):
        outputs = model(
            input_ids=prefix_ids[:, index : index + 1],
            past_key_values=past_key_values,
            use_cache=True,
        )
        logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values
    return logits


def compare_logits(tokenizer, reference, other, candidate_infos: list[dict], top_k: int) -> dict:
    torch = ruler.torch
    diff = other.float() - reference.float()
    abs_diff = diff.abs()
    flat_abs_diff = abs_diff.flatten()
    top = []
    if top_k > 0:
        top_values, flat_indices = torch.topk(flat_abs_diff, k=min(top_k, flat_abs_diff.shape[-1]))
        vocab_size = diff.shape[-1]
        for value, flat_index in zip(top_values.tolist(), flat_indices.tolist()):
            token_id = int(flat_index % vocab_size)
            top.append(
                {
                    "token_id": token_id,
                    "token_text": tokenizer.decode([token_id]),
                    "abs_logit_diff": float(value),
                    "signed_logit_diff": float(diff.flatten()[flat_index].item()),
                }
            )

    ref_log_probs = reference.float().log_softmax(dim=-1)
    other_log_probs = other.float().log_softmax(dim=-1)
    candidate_deltas = {}
    for info in candidate_infos:
        token_id = info["scored_token_id"]
        candidate_deltas[info["label"]] = float((other_log_probs[:, token_id] - ref_log_probs[:, token_id]).item())

    return {
        "max_abs_logit_diff": float(abs_diff.max().item()),
        "mean_abs_logit_diff": float(abs_diff.mean().item()),
        "candidate_logprob_deltas": candidate_deltas,
        "top_abs_logit_diffs": top,
    }


def load_model_and_tokenizer(args: argparse.Namespace):
    ruler._load_runtime_modules()
    device = ruler.pick_device(args.device)
    dtype = ruler.pick_dtype(args.dtype)

    tokenizer_source = args.tokenizer or args.model
    tokenizer = ruler.AutoTokenizer.from_pretrained(
        tokenizer_source,
        revision=args.revision,
        token=args.token,
        trust_remote_code=args.trust_remote_code,
    )
    config = ruler.AutoConfig.from_pretrained(
        args.model,
        revision=args.revision,
        token=args.token,
        trust_remote_code=args.trust_remote_code,
    )
    rope_should_be_disabled = ruler._configure_rope_parameters(config, args.rope, args.rope_theta)
    model_kwargs = {}
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    model = ruler.AutoModelForCausalLM.from_pretrained(
        args.model,
        config=config,
        revision=args.revision,
        token=args.token,
        trust_remote_code=args.trust_remote_code,
        **model_kwargs,
    )
    rope_disabled_count = ruler._disable_rope(model) if rope_should_be_disabled else 0
    if dtype != "auto":
        model = model.to(dtype=dtype)
    model = model.to(device)
    model.eval()

    runtime_replacements = ruler.force_torch_runtime_modules(model, args.torch_conv, args.torch_gated_norm)
    fallback_records = ruler.configure_fallback(model, args.fallback)
    runtime = {
        "device": str(device),
        "dtype": str(dtype),
        "rope": args.rope,
        "rope_parameters": str(getattr(model.config, "rope_parameters", None)),
        "rope_disabled_count": rope_disabled_count,
        "attn_implementation": getattr(model.config, "_attn_implementation", None),
        "runtime_replacements": runtime_replacements,
        "gdn_impl": ruler.summarize_gdn_impl(fallback_records),
        "tokenizer": {
            "source": tokenizer_source,
            "class": type(tokenizer).__name__,
            "is_fast": getattr(tokenizer, "is_fast", None),
            "vocab_size": getattr(tokenizer, "vocab_size", None),
            "len": len(tokenizer),
            "special_tokens_map": tokenizer.special_tokens_map,
        },
    }
    return model, tokenizer, device, runtime


def run_probe(args: argparse.Namespace) -> dict:
    case = resolve_case(args)
    model, tokenizer, device, runtime = load_model_and_tokenizer(args)
    torch = ruler.torch
    inputs = tokenizer([case["prompt"]], **ruler.tokenizer_call_kwargs(args)).to(device)
    prefix = resolve_prefix_and_candidates(args, tokenizer, case, device, model=model, inputs=inputs)
    prompt_len = int(inputs["input_ids"].shape[-1])
    base_result = {
        "model": args.model,
        "revision": args.revision,
        "case": {
            "name": case["name"],
            "examples_json": str(args.examples_json),
            "expected": case["expected"],
            "expected_next": prefix["expected_next"],
            "prompt_chars": len(case["prompt"]),
            "prompt_tokens": prompt_len,
            "target_text": prefix["target_text"],
            "target_token_ids": prefix["target_token_ids"],
            "target_tokens": prefix["target_tokens"],
            "generated_continuation": prefix["generated_continuation"],
            "generated_answer_text": prefix["generated_answer_text"],
            "generated_token_ids": prefix["generated_token_ids"],
            "generated_tokens": prefix["generated_tokens"],
            "target_is_generated_prefix": prefix["target_is_generated_prefix"],
            "forced_prefix_mode": prefix["mode"],
            "forced_prefix_status": prefix["status"],
            "forced_prefix": prefix["text"],
            "forced_prefix_token_ids": None,
            "forced_prefix_tokens": prefix["tokens"],
        },
        "runtime": runtime,
        "add_special_tokens": args.add_special_tokens,
        "candidates": prefix["candidates"],
        "scores": {},
        "comparisons_vs_full_no_cache": {},
    }
    if prefix["status"] != "probe":
        return base_result

    prefix_ids = prefix["ids"]
    candidate_infos = prefix["candidates"]

    with torch.no_grad():
        logits_by_path = {
            "full_no_cache": full_logits(model, inputs, prefix_ids, use_cache=False),
            "full_with_cache": full_logits(model, inputs, prefix_ids, use_cache=True),
            "cached_chunk": cached_chunk_logits(model, inputs, prefix_ids),
            "cached_single": cached_single_logits(model, inputs, prefix_ids),
        }

    scores = {
        name: score_logits(tokenizer, logits, candidate_infos, args.top_k)
        for name, logits in logits_by_path.items()
    }
    comparisons = {
        name: compare_logits(tokenizer, logits_by_path["full_no_cache"], logits, candidate_infos, args.diff_top_k)
        for name, logits in logits_by_path.items()
        if name != "full_no_cache"
    }

    prefix_token_ids = [int(token_id) for token_id in prefix_ids[0].tolist()]
    base_result["case"]["forced_prefix_token_ids"] = prefix_token_ids
    base_result["candidates"] = candidate_infos
    base_result["scores"] = scores
    base_result["comparisons_vs_full_no_cache"] = comparisons
    return base_result


def best_candidate(path_score: dict) -> str | None:
    candidates = path_score["candidates"]
    if not candidates:
        return None
    return min(candidates.items(), key=lambda item: item[1]["rank"])[0]


def print_result(result: dict) -> None:
    case = result["case"]
    runtime = result["runtime"]
    print(f"case: {case['name']}")
    print(f"examples json: {case['examples_json']}")
    print(f"model: {result['model']}")
    print(
        "runtime: "
        f"device={runtime['device']} dtype={runtime['dtype']} "
        f"attn={runtime['attn_implementation']} gdn={runtime['gdn_impl']}"
    )
    tokenizer_info = runtime["tokenizer"]
    print(
        "tokenizer: "
        f"{tokenizer_info['source']} class={tokenizer_info['class']} "
        f"vocab={tokenizer_info['vocab_size']} len={tokenizer_info['len']}"
    )
    print(f"prompt tokens: {case['prompt_tokens']}")
    print(f"target tokens: {case['target_tokens']}")
    if case["generated_answer_text"] is not None:
        print(f"generated answer: {case['generated_answer_text']!r}")
        print(f"generated tokens: {case['generated_tokens']}")
    print(f"forced prefix mode: {case['forced_prefix_mode']}")
    print(f"forced prefix status: {case['forced_prefix_status']}")
    if case["forced_prefix_status"] != "probe":
        print("verdict: generated answer does not diverge from the expected target prefix; no failure prefix to probe")
        return
    print(f"forced prefix: {case['forced_prefix']!r}")
    print(f"prefix tokens: {case['forced_prefix_tokens']}")
    print()
    print("candidates:")
    for candidate in result["candidates"]:
        note = " (first token only)" if candidate["uses_first_token_only"] else ""
        print(
            f"  {candidate['label']!r}: {candidate['text']!r} -> token {candidate['scored_token_text']!r} "
            f"id={candidate['scored_token_id']}{note}"
        )

    print()
    for path_name, path_score in result["scores"].items():
        print(f"{path_name}:")
        for candidate_text, score in path_score["candidates"].items():
            print(
                f"  {candidate_text!r:<12} rank={score['rank']:<6} "
                f"logp={score['logprob']:.3f} token={score['token_text']!r}"
            )
        top = path_score["top"]
        if top:
            top_summary = ", ".join(
                f"{entry['rank']}:{entry['token_text']!r}/{entry['logprob']:.2f}" for entry in top[:5]
            )
            print(f"  top: {top_summary}")

    print()
    print("comparisons vs full_no_cache:")
    for path_name, comparison in result["comparisons_vs_full_no_cache"].items():
        deltas = ", ".join(
            f"{candidate!r}:{delta:+.3f}"
            for candidate, delta in comparison["candidate_logprob_deltas"].items()
        )
        print(
            f"  {path_name}: max|dlogit|={comparison['max_abs_logit_diff']:.4f} "
            f"mean|dlogit|={comparison['mean_abs_logit_diff']:.4f} candidate_dlogp=[{deltas}]"
        )

    winners = {path_name: best_candidate(path_score) for path_name, path_score in result["scores"].items()}
    print()
    print("best scored candidate by path:")
    for path_name, winner in winners.items():
        print(f"  {path_name}: {winner!r}")
    if len(set(winners.values())) > 1:
        print("verdict: path disagreement")
    else:
        print("verdict: candidate ordering agrees across paths")


def main() -> None:
    args = parse_args()
    result = run_probe(args)
    if args.print_json:
        print(json.dumps(result, indent=2))
    else:
        print_result(result)


if __name__ == "__main__":
    main()
