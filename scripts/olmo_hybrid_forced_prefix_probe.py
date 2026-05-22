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
    parser.add_argument("--prompt", help="Prompt text. Mutually exclusive with --prompt-file.")
    parser.add_argument("--prompt-file", type=Path, help="Plain-text file containing the prompt.")
    parser.add_argument("--examples-json", type=Path, default=ruler.DEFAULT_EXAMPLES_JSON)
    parser.add_argument("--example-name", help="Example name from --examples-json.")
    parser.add_argument("--example-index", type=int, default=0, help="Example index from --examples-json.")
    parser.add_argument("--expected", help="Expected answer. Defaults to the selected example's expected field.")
    parser.add_argument(
        "--forced-prefix",
        help=(
            "Known answer prefix to append before probing the next-token logits. "
            "Defaults to a leading space plus the first --prefix-chars of --expected."
        ),
    )
    parser.add_argument("--prefix-chars", type=int, default=5)
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
    forced_prefix = args.forced_prefix
    if forced_prefix is None:
        if expected is None:
            raise ValueError("Pass --forced-prefix, or pass/choose an example with --expected.")
        forced_prefix = " " + expected[: args.prefix_chars]

    expected_next = args.expected_next
    if expected_next is None and expected is not None:
        stripped_prefix = forced_prefix.strip()
        if expected.startswith(stripped_prefix):
            expected_next = expected[len(stripped_prefix) :]

    candidates = []
    for candidate in (expected_next, args.bad_next, *args.candidate):
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    if not candidates:
        raise ValueError("No candidates to score. Pass --expected-next, --bad-next, or --candidate.")

    return {
        "name": name,
        "prompt": prompt,
        "expected": expected,
        "forced_prefix": forced_prefix,
        "expected_next": expected_next,
        "candidates": candidates,
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


def build_candidate_infos(tokenizer, candidate_texts: list[str]) -> list[dict]:
    infos = []
    for text in candidate_texts:
        token_ids = tokenizer(text, add_special_tokens=False).input_ids
        if not token_ids:
            raise ValueError(f"Candidate {text!r} tokenized to no tokens")
        first_id = int(token_ids[0])
        infos.append(
            {
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
        candidates[info["text"]] = {
            "token_id": info["scored_token_id"],
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
        candidate_deltas[info["text"]] = float((other_log_probs[:, token_id] - ref_log_probs[:, token_id]).item())

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

    tokenizer = ruler.AutoTokenizer.from_pretrained(
        args.model,
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
    }
    return model, tokenizer, device, runtime


def run_probe(args: argparse.Namespace) -> dict:
    case = resolve_case(args)
    model, tokenizer, device, runtime = load_model_and_tokenizer(args)
    torch = ruler.torch
    inputs = tokenizer([case["prompt"]], **ruler.tokenizer_call_kwargs(args)).to(device)
    prefix_ids = encode_continuation(tokenizer, case["forced_prefix"], device)
    candidate_infos = build_candidate_infos(tokenizer, case["candidates"])

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

    prompt_len = int(inputs["input_ids"].shape[-1])
    prefix_token_ids = [int(token_id) for token_id in prefix_ids[0].tolist()]
    return {
        "model": args.model,
        "revision": args.revision,
        "case": {
            "name": case["name"],
            "expected": case["expected"],
            "expected_next": case["expected_next"],
            "prompt_chars": len(case["prompt"]),
            "prompt_tokens": prompt_len,
            "forced_prefix": case["forced_prefix"],
            "forced_prefix_token_ids": prefix_token_ids,
            "forced_prefix_tokens": token_texts(tokenizer, prefix_token_ids),
        },
        "runtime": runtime,
        "add_special_tokens": args.add_special_tokens,
        "candidates": candidate_infos,
        "scores": scores,
        "comparisons_vs_full_no_cache": comparisons,
    }


def best_candidate(path_score: dict) -> str | None:
    candidates = path_score["candidates"]
    if not candidates:
        return None
    return min(candidates.items(), key=lambda item: item[1]["rank"])[0]


def print_result(result: dict) -> None:
    case = result["case"]
    runtime = result["runtime"]
    print(f"case: {case['name']}")
    print(f"model: {result['model']}")
    print(
        "runtime: "
        f"device={runtime['device']} dtype={runtime['dtype']} "
        f"attn={runtime['attn_implementation']} gdn={runtime['gdn_impl']}"
    )
    print(f"prompt tokens: {case['prompt_tokens']}")
    print(f"forced prefix: {case['forced_prefix']!r}")
    print(f"prefix tokens: {case['forced_prefix_tokens']}")
    print()
    print("candidates:")
    for candidate in result["candidates"]:
        note = " (first token only)" if candidate["uses_first_token_only"] else ""
        print(
            f"  {candidate['text']!r} -> token {candidate['scored_token_text']!r} "
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
