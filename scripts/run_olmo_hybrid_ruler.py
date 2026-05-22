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
import functools
import hashlib
import importlib
import inspect
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


torch = None
AutoConfig = None
AutoModelForCausalLM = None
AutoTokenizer = None
GenerationConfig = None


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


def _load_runtime_modules() -> None:
    global torch, AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    if torch is not None:
        return

    import torch as torch_mod

    from transformers import AutoConfig as AutoConfigCls
    from transformers import AutoModelForCausalLM as AutoModelForCausalLMCls
    from transformers import AutoTokenizer as AutoTokenizerCls
    from transformers import GenerationConfig as GenerationConfigCls

    torch = torch_mod
    AutoConfig = AutoConfigCls
    AutoModelForCausalLM = AutoModelForCausalLMCls
    AutoTokenizer = AutoTokenizerCls
    GenerationConfig = GenerationConfigCls


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
HASH_CHUNK_BYTES = 16 * 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", nargs="?", help="Model name or local path.")
    parser.add_argument("--model", dest="model", help="Model name or local path.")
    parser.add_argument("--examples-json", type=Path, default=DEFAULT_EXAMPLES_JSON)
    parser.add_argument("--fork", help="Transformers fork to run, as URL@BRANCH.")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, mps, etc.")
    parser.add_argument("--revision", help="Branch/tag/commit for --model.")
    parser.add_argument("--token", help="HF token, if needed. Usually HF_TOKEN env var is simpler.")
    parser.add_argument("--attn-implementation", help="Full-attention backend to request, e.g. eager or sdpa.")
    parser.add_argument(
        "--rope",
        choices=["auto", "disable", "default", "config"],
        default="auto",
        help=(
            "RoPE handling for full-attention layers. 'auto' preserves the current behavior: inject a default theta "
            "if absent and then disable RoPE. 'default' injects theta but leaves RoPE active. 'config' leaves the "
            "loaded config untouched."
        ),
    )
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument(
        "--add-special-tokens",
        choices=["default", "true", "false"],
        default="default",
        help="Pass add_special_tokens to the tokenizer, or leave tokenizer default behavior unchanged.",
    )
    parser.add_argument(
        "--fallback",
        choices=["force", "check", "auto"],
        default="force",
        help="'force' replaces FLA GDN callables with torch fallback; 'check' errors unless fallback is already active.",
    )
    parser.add_argument(
        "--torch-conv",
        action="store_true",
        help="Replace linear-attention short conv modules with torch fallback.",
    )
    parser.add_argument(
        "--torch-gated-norm",
        action="store_true",
        help="Replace linear-attention gated RMSNorm modules with torch fallback.",
    )
    parser.add_argument("--max-examples", type=int, help="Only run the first N examples.")
    parser.add_argument("--ablation-suite", action="store_true", help="Run a predefined set of OLMo Hybrid ablations.")
    parser.add_argument("--ablation-output-dir", type=Path, help="Optional directory for per-ablation JSON outputs.")
    parser.add_argument(
        "--include-slow-ablations",
        action="store_true",
        help="Include expensive diagnostic ablations that are skipped by the default suite.",
    )
    parser.add_argument(
        "--score-answers",
        action="store_true",
        help="Teacher-force score gold/generated answer candidates for each example.",
    )
    parser.add_argument(
        "--score-mode",
        choices=["cached", "full", "both"],
        default="cached",
        help="Use cached decode, full no-cache forward, or both for teacher-forced answer scoring.",
    )
    parser.add_argument(
        "--score-rank-by",
        choices=["mean", "sum", "char_mean"],
        default="mean",
        help="Metric used for compact best-candidate summaries.",
    )
    parser.add_argument(
        "--answer-top-k",
        type=int,
        default=0,
        help="Record the top-k next tokens at the answer boundary.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--no-cache", action="store_true", help="Run generate(use_cache=False).")
    parser.add_argument("--print-json", action="store_true")
    parser.add_argument("--_child-run", action="store_true", help=argparse.SUPPRESS)
    l2norm = parser.add_mutually_exclusive_group()
    l2norm.add_argument("--l2norm", dest="l2norm", action="store_true", default=None, help="Enable linear_use_qk_l2norm.")
    l2norm.add_argument("--no-l2norm", dest="l2norm", action="store_false", help="Disable linear_use_qk_l2norm (default).")
    g_clamp = parser.add_mutually_exclusive_group()
    g_clamp.add_argument(
        "--g-clamp",
        "--torch-g-clamp",
        dest="g_clamp",
        action="store_true",
        default=None,
        help="Enable the torch GatedDeltaNet fallback clamp for g in [-20, 20].",
    )
    g_clamp.add_argument(
        "--no-g-clamp",
        "--no-torch-g-clamp",
        dest="g_clamp",
        action="store_false",
        help="Disable the torch GatedDeltaNet fallback clamp for g.",
    )
    args = parser.parse_args()
    args.model = args.model or args.model_path
    if args.model is None:
        parser.error("a model path is required, either positionally or with --model")
    return args


def make_child_args(args: argparse.Namespace) -> list[str]:
    child_args = [
        "--model",
        args.model,
        "--examples-json",
        str(args.examples_json),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--dtype",
        args.dtype,
        "--device",
        args.device,
        "--fallback",
        args.fallback,
        "--_child-run",
        "--print-json",
    ]
    if args.revision:
        child_args.extend(["--revision", args.revision])
    if args.token:
        child_args.extend(["--token", args.token])
    if args.trust_remote_code:
        child_args.append("--trust-remote-code")
    if args.attn_implementation:
        child_args.extend(["--attn-implementation", args.attn_implementation])
    if args.rope != "auto":
        child_args.extend(["--rope", args.rope])
    if args.rope_theta != 10000.0:
        child_args.extend(["--rope-theta", str(args.rope_theta)])
    if args.add_special_tokens != "default":
        child_args.extend(["--add-special-tokens", args.add_special_tokens])
    if args.no_cache:
        child_args.append("--no-cache")
    if args.torch_conv:
        child_args.append("--torch-conv")
    if args.torch_gated_norm:
        child_args.append("--torch-gated-norm")
    if args.max_examples is not None:
        child_args.extend(["--max-examples", str(args.max_examples)])
    if args.include_slow_ablations:
        child_args.append("--include-slow-ablations")
    if args.score_answers:
        child_args.append("--score-answers")
    if args.score_mode != "cached":
        child_args.extend(["--score-mode", args.score_mode])
    if args.score_rank_by != "mean":
        child_args.extend(["--score-rank-by", args.score_rank_by])
    if args.answer_top_k:
        child_args.extend(["--answer-top-k", str(args.answer_top_k)])
    if args.l2norm is True:
        child_args.append("--l2norm")
    elif args.l2norm is False:
        child_args.append("--no-l2norm")
    if args.g_clamp is True:
        child_args.append("--g-clamp")
    elif args.g_clamp is False:
        child_args.append("--no-g-clamp")
    return child_args


def split_fork(fork: str) -> tuple[str, str]:
    if "@" not in fork:
        raise ValueError("--fork must be formatted as URL@BRANCH")
    fork_url, fork_branch = fork.rsplit("@", 1)
    if not fork_url or not fork_branch:
        raise ValueError("--fork must be formatted as URL@BRANCH")
    return fork_url, fork_branch


def run_json_command(command: list[str], output_path: Path, env: dict[str, str] | None = None) -> bool:
    with output_path.open("w", encoding="utf-8") as handle:
        return subprocess.run(command, stdout=handle, env=env, check=False).returncode == 0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def hash_model_safetensors(model_path: str) -> dict:
    path = Path(model_path).expanduser()
    if path.is_file() and path.name.endswith(".safetensors"):
        files = [path]
    elif path.is_dir():
        files = sorted(path.glob("*.safetensors"))
    else:
        files = []

    entries = []
    manifest = hashlib.sha256()
    for file_path in files:
        file_sha256 = sha256_file(file_path)
        file_size = file_path.stat().st_size
        name = file_path.name
        entries.append({"name": name, "size": file_size, "sha256": file_sha256})
        manifest.update(name.encode("utf-8"))
        manifest.update(b"\0")
        manifest.update(str(file_size).encode("ascii"))
        manifest.update(b"\0")
        manifest.update(file_sha256.encode("ascii"))
        manifest.update(b"\n")

    return {
        "sha256": manifest.hexdigest() if entries else None,
        "num_files": len(entries),
        "files": entries,
    }


def grade(result: dict) -> str:
    return "PASS" if result["exact_matches"] else ("digit-match" if result["digit_matches"] else "FAIL")


def unique_values(values):
    unique = []
    for value in values:
        if value not in unique:
            unique.append(value)
    return unique


def summarize_g_clamp(fallback_records: list[dict], config_value, requested_value) -> dict:
    chunk_records = [record for record in fallback_records if record["chunk_torch_fallback"]]
    recurrent_records = [record for record in fallback_records if record["recurrent_torch_fallback"]]
    return {
        "requested": requested_value,
        "config": config_value,
        "chunk_torch_layers": len(chunk_records),
        "chunk_values": unique_values([record["chunk_g_clamp"] for record in chunk_records]),
        "chunk_sources": unique_values([record["chunk_g_clamp_source"] for record in chunk_records]),
        "recurrent_torch_layers": len(recurrent_records),
        "recurrent_values": unique_values([record["recurrent_g_clamp"] for record in recurrent_records]),
        "recurrent_sources": unique_values([record["recurrent_g_clamp_source"] for record in recurrent_records]),
    }


def format_g_clamp_summary(summary: dict) -> str:
    return (
        f"requested={summary['requested']} config={summary['config']} "
        f"chunk={summary['chunk_values']} via={summary['chunk_sources']} "
        f"({summary['chunk_torch_layers']} torch layers) "
        f"recurrent={summary['recurrent_values']} via={summary['recurrent_sources']} "
        f"({summary['recurrent_torch_layers']} torch layers)"
    )


def print_compact_result(result: dict, label: str) -> None:
    print(f"\n{'=' * 60}\n  {label}\n{'=' * 60}")
    print(f"  device:       {result['device']}")
    print(f"  dtype:        {result['dtype']}")
    print(f"  l2norm:       {result['l2norm']}")
    print(f"  g clamp:      {format_g_clamp_summary(result['g_clamp'])}")
    print(f"  rope:         {result['rope']} disabled={result['rope_disabled_count']}")
    print(f"  attn impl:    {result['attn_implementation']}")
    print(f"  special toks: {result['add_special_tokens']}")
    print(f"  replacements: {result['runtime_replacements']}")
    print(
        f"  scoring:      answers={result['score_answers']} "
        f"mode={result['score_mode']} rank_by={result['score_rank_by']} top_k={result['answer_top_k']}"
    )
    print(f"  model hash:   {result['model_safetensors']['sha256']} ({result['model_safetensors']['num_files']} safetensors)")
    print(f"  examples:     {result['num_examples']}")
    print(f"  exact match:  {result['exact_matches']}/{result['num_examples']}  ({grade(result)})")
    print(f"  digit match:  {result['digit_matches']}/{result['num_examples']}")
    for example in result["examples"]:
        print(f"  {example['name']}: exact={example['exact_match']} digit={example['digit_match']}")
        if "answer_scores" in example:
            best = best_answer_score(example["answer_scores"], result["score_mode"], result["score_rank_by"])
            if best is not None:
                print(f"    best score:   {best[0]} ({best[1]:.3f})")
            summary = answer_score_summary(example["answer_scores"], result["score_mode"], result["score_rank_by"])
            if summary:
                print(f"    score cmp:    {summary}")
        if "answer_boundary_topk" in example:
            gold_first = example["answer_boundary_topk"]["gold_first_token"]
            print(f"    gold first:   rank={gold_first['rank']} logp={gold_first['logprob']:.3f}")
        print(f"    continuation: {example['continuation'][:120]!r}")


def run_with_fork(args: argparse.Namespace) -> int:
    fork_url, fork_branch = split_fork(args.fork)
    child_args = make_child_args(args)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        venv = tmp_path / "venv"
        fork_json = tmp_path / "fork.json"

        print(f"[fork] Creating venv and installing {fork_url} @ {fork_branch} ...", file=sys.stderr)
        subprocess.run(["uv", "venv", "--system-site-packages", str(venv)], check=True)
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(venv / "bin" / "python"),
                f"transformers @ git+{fork_url}@{fork_branch}",
            ],
            check=True,
        )

        print(f"[fork] Running with {args.fork} ...", file=sys.stderr)
        fork_env = os.environ.copy()
        fork_env.pop("PYTHONPATH", None)
        fork_ok = run_json_command([str(venv / "bin" / "python"), str(Path(__file__).resolve()), *child_args], fork_json, fork_env)
        if not fork_ok:
            return 1
        print("[fork] Done.", file=sys.stderr)

        fork_result = json.loads(fork_json.read_text(encoding="utf-8"))
        if args.print_json:
            print(json.dumps(fork_result, indent=2))
        else:
            print_compact_result(fork_result, f"FORK  {args.fork}")

    return 0


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


def resolve_examples(args: argparse.Namespace) -> list[dict]:
    examples = load_examples(args.examples_json)
    if args.max_examples is not None:
        examples = examples[: args.max_examples]
    return examples


def tokenizer_call_kwargs(args: argparse.Namespace) -> dict:
    kwargs = {"return_tensors": "pt", "return_token_type_ids": False}
    if args.add_special_tokens != "default":
        kwargs["add_special_tokens"] = args.add_special_tokens == "true"
    return kwargs


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


def _ensure_rope_parameters(config, default_theta: float) -> bool:
    """Ensure config.rope_parameters has a usable rope_theta.

    Returns True when the theta was injected rather than loaded from config.
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
    rope_params["rope_theta"] = default_theta
    return True


def _configure_rope_parameters(config, mode: str, default_theta: float) -> bool:
    """Configure RoPE for ablations.

    Returns True when rotary embeddings should be zeroed out after model load.
    """
    if mode == "config":
        return False
    injected = _ensure_rope_parameters(config, default_theta)
    if mode == "disable":
        return True
    if mode == "default":
        return False
    # Current compatibility behavior: missing theta means this checkpoint expected
    # no RoPE, so construct with a placeholder then zero the rotary buffers.
    return injected


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


def _copy_conv_weight(source: torch.nn.Module, target: torch.nn.Module) -> None:
    source_weight = source.weight.detach()
    if source_weight.shape == target.weight.shape:
        target.weight.copy_(source_weight)
    elif source_weight.ndim == 2 and target.weight.ndim == 3 and source_weight.shape == target.weight[:, 0, :].shape:
        target.weight[:, 0, :].copy_(source_weight)
    else:
        raise ValueError(f"Unsupported short-conv weight shape {tuple(source_weight.shape)}")
    source_bias = getattr(source, "bias", None)
    if source_bias is not None:
        target.bias.copy_(source_bias.detach())


def force_torch_runtime_modules(model, force_conv: bool, force_gated_norm: bool) -> dict[str, int]:
    if not force_conv and not force_gated_norm:
        return {"conv": 0, "gated_norm": 0}

    olmo_mod = _import_olmo_mod()
    replacements = {"conv": 0, "gated_norm": 0}
    with torch.no_grad():
        for _, module in iter_linear_attn_modules(model):
            if force_conv:
                for attr in ("q_conv1d", "k_conv1d", "v_conv1d"):
                    old_conv = getattr(module, attr)
                    old_bias = getattr(old_conv, "bias", None)
                    new_conv = olmo_mod.OlmoHybridShortConvolution(
                        hidden_size=old_conv.weight.shape[0],
                        kernel_size=old_conv.weight.shape[-1],
                        bias=old_bias is not None,
                        activation="silu",
                    )
                    new_conv = new_conv.to(device=old_conv.weight.device, dtype=old_conv.weight.dtype)
                    _copy_conv_weight(old_conv, new_conv)
                    setattr(module, attr, new_conv)
                    replacements["conv"] += 1

            if force_gated_norm:
                old_norm = module.o_norm
                new_norm = olmo_mod.OlmoHybridRMSNormGated(module.head_v_dim, eps=1e-5)
                new_norm = new_norm.to(device=old_norm.weight.device, dtype=old_norm.weight.dtype)
                new_norm.weight.copy_(old_norm.weight.detach())
                module.o_norm = new_norm
                replacements["gated_norm"] += 1

    return replacements


def iter_linear_attn_modules(model):
    for name, module in model.named_modules():
        if hasattr(module, "chunk_gated_delta_rule") and hasattr(module, "recurrent_gated_delta_rule"):
            yield name, module


def callable_name(fn) -> str:
    module = getattr(fn, "__module__", type(fn).__module__)
    name = getattr(fn, "__qualname__", getattr(fn, "__name__", type(fn).__name__))
    return f"{module}.{name}"


def callable_base(fn):
    return getattr(fn, "_olmo_hybrid_base_callable", fn)


def callable_param(fn, param_name: str):
    try:
        return inspect.signature(callable_base(fn)).parameters.get(param_name)
    except (TypeError, ValueError):
        return None


def callable_kwarg_default(fn, param_name: str):
    param = callable_param(fn, param_name)
    if param is None or param.default is inspect.Signature.empty:
        return None
    return param.default


def source_mentions_g_clamp(fn) -> bool | None:
    try:
        source = inspect.getsource(callable_base(fn))
    except (OSError, TypeError):
        return None
    return "g.clamp" in source or ".clamp(min=-20" in source


def wrap_torch_gdn_callable(fn, g_clamp: bool, label: str):
    if callable_param(fn, "clamp_g") is None:
        source_has_g_clamp = source_mentions_g_clamp(fn)
        raise RuntimeError(
            f"Cannot set g_clamp={g_clamp} for {label} torch GDN fallback because {callable_name(fn)} "
            f"does not expose a clamp_g argument. source_has_g_clamp={source_has_g_clamp}"
        )

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        kwargs["clamp_g"] = g_clamp
        return fn(*args, **kwargs)

    wrapped._olmo_hybrid_base_callable = fn
    wrapped._olmo_hybrid_g_clamp = g_clamp
    return wrapped


def torch_g_clamp_info(module, fn, torch_fn) -> dict:
    is_torch_fallback = callable_base(fn) is torch_fn
    source_has_g_clamp = source_mentions_g_clamp(torch_fn) if is_torch_fallback else None
    info = {
        "is_torch_fallback": is_torch_fallback,
        "g_clamp": None,
        "g_clamp_source": "not_torch_fallback",
        "source_has_g_clamp": source_has_g_clamp,
    }
    if not is_torch_fallback:
        return info

    if hasattr(fn, "_olmo_hybrid_g_clamp"):
        info["g_clamp"] = fn._olmo_hybrid_g_clamp
        info["g_clamp_source"] = "runner_override"
        return info

    module_clamp_g = getattr(module, "clamp_g", None)
    if module_clamp_g is not None and callable_param(torch_fn, "clamp_g") is not None:
        info["g_clamp"] = module_clamp_g
        info["g_clamp_source"] = "module.clamp_g"
        return info

    default = callable_kwarg_default(torch_fn, "clamp_g")
    if default is not None:
        info["g_clamp"] = default
        info["g_clamp_source"] = "callable_default"
    elif source_has_g_clamp is not None:
        info["g_clamp"] = source_has_g_clamp
        info["g_clamp_source"] = "source_scan"
    else:
        info["g_clamp_source"] = "unknown"
    return info


def configure_fallback(model, mode: str, g_clamp: bool | None = None) -> list[dict]:
    olmo_mod = _import_olmo_mod()
    records = []
    for name, module in iter_linear_attn_modules(model):
        if g_clamp is not None and hasattr(module, "clamp_g"):
            module.clamp_g = g_clamp

        if mode == "force":
            module.chunk_gated_delta_rule = (
                wrap_torch_gdn_callable(olmo_mod.torch_chunk_gated_delta_rule, g_clamp, "chunk")
                if g_clamp is not None
                else olmo_mod.torch_chunk_gated_delta_rule
            )
            module.recurrent_gated_delta_rule = (
                wrap_torch_gdn_callable(olmo_mod.torch_recurrent_gated_delta_rule, g_clamp, "recurrent")
                if g_clamp is not None
                else olmo_mod.torch_recurrent_gated_delta_rule
            )
        elif g_clamp is not None:
            if module.chunk_gated_delta_rule is olmo_mod.torch_chunk_gated_delta_rule:
                module.chunk_gated_delta_rule = wrap_torch_gdn_callable(
                    olmo_mod.torch_chunk_gated_delta_rule, g_clamp, "chunk"
                )
            if module.recurrent_gated_delta_rule is olmo_mod.torch_recurrent_gated_delta_rule:
                module.recurrent_gated_delta_rule = wrap_torch_gdn_callable(
                    olmo_mod.torch_recurrent_gated_delta_rule, g_clamp, "recurrent"
                )

        chunk_name = callable_name(module.chunk_gated_delta_rule)
        recurrent_name = callable_name(module.recurrent_gated_delta_rule)
        chunk_clamp = torch_g_clamp_info(module, module.chunk_gated_delta_rule, olmo_mod.torch_chunk_gated_delta_rule)
        recurrent_clamp = torch_g_clamp_info(
            module, module.recurrent_gated_delta_rule, olmo_mod.torch_recurrent_gated_delta_rule
        )
        records.append(
            {
                "layer": name,
                "chunk": chunk_name,
                "recurrent": recurrent_name,
                "chunk_torch_fallback": chunk_clamp["is_torch_fallback"],
                "chunk_g_clamp": chunk_clamp["g_clamp"],
                "chunk_g_clamp_source": chunk_clamp["g_clamp_source"],
                "chunk_source_has_g_clamp": chunk_clamp["source_has_g_clamp"],
                "recurrent_torch_fallback": recurrent_clamp["is_torch_fallback"],
                "recurrent_g_clamp": recurrent_clamp["g_clamp"],
                "recurrent_g_clamp_source": recurrent_clamp["g_clamp_source"],
                "recurrent_source_has_g_clamp": recurrent_clamp["source_has_g_clamp"],
            }
        )

        if mode == "check":
            if callable_base(module.chunk_gated_delta_rule) is not olmo_mod.torch_chunk_gated_delta_rule:
                raise RuntimeError(f"{name} chunk GDN is not torch fallback: {chunk_name}")
            if callable_base(module.recurrent_gated_delta_rule) is not olmo_mod.torch_recurrent_gated_delta_rule:
                raise RuntimeError(f"{name} recurrent GDN is not torch fallback: {recurrent_name}")

    if not records:
        raise RuntimeError("No OLMo Hybrid linear-attention modules with GDN callables were found.")

    return records


def first_answer_line(continuation: str) -> str:
    for line in continuation.splitlines():
        if line.strip():
            return line
    return continuation


def comma_format_digits(value: str) -> str | None:
    if not value.isdigit():
        return None
    return f"{int(value):,}"


def add_unique_candidate(candidates: list[dict[str, str]], label: str, text: str | None) -> None:
    if not text:
        return
    if any(candidate["text"] == text for candidate in candidates):
        return
    candidates.append({"label": label, "text": text})


def build_answer_candidates(expected: str, continuation: str) -> list[dict[str, str]]:
    candidates = []
    add_unique_candidate(candidates, "gold", f" {expected}")
    add_unique_candidate(candidates, "gold_period", f" {expected}.")
    comma_expected = comma_format_digits(expected)
    if comma_expected is not None:
        add_unique_candidate(candidates, "gold_commas", f" {comma_expected}")
        add_unique_candidate(candidates, "gold_commas_period", f" {comma_expected}.")
    generated_line = first_answer_line(continuation)
    add_unique_candidate(candidates, "generated_first_line", generated_line)
    return candidates


def token_entries(tokenizer, token_ids, token_logprobs, token_ranks) -> list[dict]:
    entries = []
    for token_id, logprob, rank in zip(token_ids, token_logprobs, token_ranks):
        entries.append(
            {
                "id": int(token_id),
                "text": tokenizer.decode([int(token_id)]),
                "logprob": float(logprob),
                "rank": int(rank),
            }
        )
    return entries


def logprob_rank(log_probs: torch.Tensor, token_id: torch.Tensor) -> tuple[float, int]:
    token_logprob = log_probs.gather(-1, token_id.view(1, 1)).squeeze()
    rank = (log_probs > token_logprob).sum() + 1
    return float(token_logprob.item()), int(rank.item())


def build_score_payload(continuation: str, tokenizer, target_ids, token_logprobs, token_ranks) -> dict:
    sum_logprob = sum(token_logprobs)
    return {
        "num_tokens": target_ids.shape[-1],
        "num_chars": len(continuation),
        "sum_logprob": sum_logprob,
        "mean_logprob": sum_logprob / len(token_logprobs),
        "char_mean_logprob": sum_logprob / max(len(continuation), 1),
        "tokens": token_entries(tokenizer, target_ids[0].tolist(), token_logprobs, token_ranks),
    }


def score_continuation_cached(model, tokenizer, inputs, continuation: str, device: torch.device) -> dict:
    target_ids = tokenizer(continuation, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    if target_ids.shape[-1] == 0:
        return {
            "num_tokens": 0,
            "num_chars": 0,
            "sum_logprob": 0.0,
            "mean_logprob": None,
            "char_mean_logprob": None,
            "tokens": [],
        }

    outputs = model(**inputs, use_cache=True)
    logits = outputs.logits[:, -1, :]
    past_key_values = outputs.past_key_values
    token_logprobs = []
    token_ranks = []

    for index in range(target_ids.shape[-1]):
        log_probs = logits.float().log_softmax(dim=-1)
        token_logprob, rank = logprob_rank(log_probs, target_ids[:, index])
        token_logprobs.append(token_logprob)
        token_ranks.append(rank)
        if index + 1 < target_ids.shape[-1]:
            outputs = model(
                input_ids=target_ids[:, index : index + 1],
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values

    return build_score_payload(continuation, tokenizer, target_ids, token_logprobs, token_ranks)


def score_continuation_full(model, tokenizer, inputs, continuation: str, device: torch.device) -> dict:
    target_ids = tokenizer(continuation, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    if target_ids.shape[-1] == 0:
        return {
            "num_tokens": 0,
            "num_chars": 0,
            "sum_logprob": 0.0,
            "mean_logprob": None,
            "char_mean_logprob": None,
            "tokens": [],
        }

    input_ids = torch.cat([inputs["input_ids"], target_ids], dim=-1)
    attention_mask = torch.ones_like(input_ids, device=device)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    input_len = inputs["input_ids"].shape[-1]
    logits = outputs.logits[:, input_len - 1 : input_len + target_ids.shape[-1] - 1, :]
    log_probs = logits.float().log_softmax(dim=-1)

    token_logprobs = []
    token_ranks = []
    for index in range(target_ids.shape[-1]):
        token_logprob, rank = logprob_rank(log_probs[:, index, :], target_ids[:, index])
        token_logprobs.append(token_logprob)
        token_ranks.append(rank)

    return build_score_payload(continuation, tokenizer, target_ids, token_logprobs, token_ranks)


def score_answer_candidates(
    args, model, tokenizer, inputs, expected: str, continuation: str, device: torch.device
) -> dict:
    scores = {}
    for candidate in build_answer_candidates(expected, continuation):
        candidate_scores = {"text": candidate["text"]}
        if args.score_mode in {"cached", "both"}:
            candidate_scores["cached"] = score_continuation_cached(
                model, tokenizer, inputs, candidate["text"], device
            )
        if args.score_mode in {"full", "both"}:
            candidate_scores["full"] = score_continuation_full(model, tokenizer, inputs, candidate["text"], device)
        scores[candidate["label"]] = candidate_scores
    return scores


def answer_boundary_topk(model, tokenizer, inputs, expected: str, device: torch.device, top_k: int) -> dict | None:
    if top_k <= 0:
        return None

    outputs = model(**inputs, use_cache=False)
    log_probs = outputs.logits[:, -1, :].float().log_softmax(dim=-1)
    top_logprobs, top_indices = torch.topk(log_probs, k=top_k, dim=-1)
    gold_first_ids = tokenizer(f" {expected}", return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    gold_logprob, gold_rank = logprob_rank(log_probs, gold_first_ids[:, 0])
    return {
        "gold_first_token": {
            "id": int(gold_first_ids[0, 0]),
            "text": tokenizer.decode([int(gold_first_ids[0, 0])]),
            "logprob": gold_logprob,
            "rank": gold_rank,
        },
        "top_tokens": token_entries(
            tokenizer,
            top_indices[0].tolist(),
            top_logprobs[0].tolist(),
            list(range(1, top_k + 1)),
        ),
    }


def score_metric_value(score_payload: dict, metric: str) -> float | None:
    key = {
        "sum": "sum_logprob",
        "mean": "mean_logprob",
        "char_mean": "char_mean_logprob",
    }[metric]
    return score_payload.get(key)


def preferred_score_payload(payload: dict, mode: str) -> dict | None:
    preferred_mode = "cached" if mode != "full" else "full"
    return payload.get(preferred_mode) or payload.get("full") or payload.get("cached")


def best_answer_score(answer_scores: dict, mode: str, metric: str) -> tuple[str, float] | None:
    best_label = None
    best_score = None
    for label, payload in answer_scores.items():
        score_payload = preferred_score_payload(payload, mode)
        if score_payload is None:
            continue
        score = score_metric_value(score_payload, metric)
        if score is None:
            continue
        if best_score is None or score > best_score:
            best_label = label
            best_score = score
    if best_label is None:
        return None
    return best_label, best_score


def answer_score_summary(answer_scores: dict, mode: str, metric: str) -> str:
    labels = ["gold", "gold_period", "gold_commas", "gold_commas_period", "generated_first_line"]
    parts = []
    for label in labels:
        payload = answer_scores.get(label)
        if payload is None:
            continue
        score_payload = preferred_score_payload(payload, mode)
        if score_payload is None:
            continue
        score = score_metric_value(score_payload, metric)
        if score is not None:
            parts.append(f"{label}={score:.3f}")
    return ", ".join(parts)


def run_one_example(args, model, tokenizer, device: torch.device, example: dict) -> dict:
    inputs = tokenizer([example["prompt"]], **tokenizer_call_kwargs(args)).to(device)
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

    result = {
        "name": example["name"],
        "key": example["key"],
        "expected": expected,
        "input_tokens": input_len,
        "new_tokens": new_tokens.shape[-1],
        "continuation": continuation,
        "full_text_suffix": full_text[-1000:],
        "exact_match": expected in continuation,
        "digit_match": re.sub(r"\D", "", expected) in re.sub(r"\D", "", continuation),
    }
    with torch.no_grad():
        if args.score_answers:
            result["answer_scores"] = score_answer_candidates(
                args, model, tokenizer, inputs, expected, continuation, device
            )
        if args.answer_top_k:
            result["answer_boundary_topk"] = answer_boundary_topk(
                model, tokenizer, inputs, expected, device, args.answer_top_k
            )
    return result


def run_one(args, model_path: str, revision: str | None) -> dict:
    _load_runtime_modules()
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
    rope_should_be_disabled = _configure_rope_parameters(config, args.rope, args.rope_theta)
    if args.l2norm is not None:
        config.linear_use_qk_l2norm = args.l2norm
    if args.g_clamp is not None:
        config.linear_clamp_g = args.g_clamp
    model_kwargs = {}
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        revision=revision,
        token=args.token,
        trust_remote_code=args.trust_remote_code,
        **model_kwargs,
    )
    rope_disabled_count = _disable_rope(model) if rope_should_be_disabled else 0
    if dtype != "auto":
        model = model.to(dtype=dtype)
    model = model.to(device)
    model.eval()

    runtime_replacements = force_torch_runtime_modules(model, args.torch_conv, args.torch_gated_norm)
    fallback_records = configure_fallback(model, args.fallback, args.g_clamp)
    g_clamp_summary = summarize_g_clamp(
        fallback_records, getattr(model.config, "linear_clamp_g", None), args.g_clamp
    )
    examples = [
        run_one_example(args, model, tokenizer, device, example) for example in resolve_examples(args)
    ]

    return {
        "model": model_path,
        "revision": revision,
        "model_safetensors": hash_model_safetensors(model_path),
        "device": str(device),
        "dtype": str(dtype),
        "l2norm": getattr(model.config, "linear_use_qk_l2norm", None),
        "g_clamp": g_clamp_summary,
        "rope": args.rope,
        "rope_parameters": str(getattr(model.config, "rope_parameters", None)),
        "rope_disabled_count": rope_disabled_count,
        "attn_implementation": getattr(model.config, "_attn_implementation", None),
        "add_special_tokens": args.add_special_tokens,
        "tokenizer": {
            "class": type(tokenizer).__name__,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        },
        "runtime_replacements": runtime_replacements,
        "score_answers": args.score_answers,
        "score_mode": args.score_mode,
        "score_rank_by": args.score_rank_by,
        "answer_top_k": args.answer_top_k,
        "max_new_tokens": args.max_new_tokens,
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
    print(f"g clamp:      {format_g_clamp_summary(result['g_clamp'])}")
    print(
        f"rope:         {result['rope']} disabled={result['rope_disabled_count']} "
        f"params={result['rope_parameters']}"
    )
    print(f"attn impl:    {result['attn_implementation']}")
    print(f"special toks: {result['add_special_tokens']} tokenizer={result['tokenizer']}")
    print(f"replacements: {result['runtime_replacements']}")
    print(
        f"scoring:      answers={result['score_answers']} "
        f"mode={result['score_mode']} rank_by={result['score_rank_by']} top_k={result['answer_top_k']}"
    )
    print(f"model hash:   {result['model_safetensors']['sha256']} ({result['model_safetensors']['num_files']} safetensors)")
    print(f"examples:     {result['num_examples']}")
    print(f"exact match:  {result['exact_matches']}/{result['num_examples']}")
    print(f"digit match:  {result['digit_matches']}/{result['num_examples']}")
    for example in result["examples"]:
        print(f"\n--- {example['name']} ({example['key']}) ---")
        print(f"input tokens: {example['input_tokens']}")
        print(f"new tokens:   {example['new_tokens']}")
        print(f"expected:     {example['expected']}")
        print(f"exact match:  {example['exact_match']}")
        print(f"digit match:  {example['digit_match']}")
        print("continuation:")
        print(example["continuation"])
        if "answer_scores" in example:
            print("answer scores:")
            best = best_answer_score(example["answer_scores"], result["score_mode"], result["score_rank_by"])
            if best is not None:
                print(f"  best ({result['score_rank_by']}): {best[0]} ({best[1]:.3f})")
            for label, payload in example["answer_scores"].items():
                score_parts = []
                for mode in ("cached", "full"):
                    if mode in payload:
                        score = payload[mode]
                        score_parts.append(
                            f"{mode}=sum {score['sum_logprob']:.3f} "
                            f"mean {score['mean_logprob']:.3f} char {score['char_mean_logprob']:.3f}"
                        )
                print(f"  {label}: {payload['text']!r}  {'; '.join(score_parts)}")
        if "answer_boundary_topk" in example:
            print("answer-boundary top tokens:")
            gold_first = example["answer_boundary_topk"]["gold_first_token"]
            print(
                f"  gold first token: {gold_first['text']!r} "
                f"rank={gold_first['rank']} logp={gold_first['logprob']:.3f}"
            )
            for token in example["answer_boundary_topk"]["top_tokens"]:
                print(f"  #{token['rank']}: {token['text']!r} id={token['id']} logp={token['logprob']:.3f}")


ABLATION_CASES = [
    ("baseline", []),
    ("explicit_no_l2norm", ["--no-l2norm"]),
    ("l2norm_on", ["--l2norm"]),
    ("g_clamp_on", ["--g-clamp"]),
    ("g_clamp_off", ["--no-g-clamp"]),
    ("fallback_auto", ["--fallback", "auto"]),
    ("attn_eager", ["--attn-implementation", "eager"]),
    ("attn_sdpa", ["--attn-implementation", "sdpa"]),
    ("rope_disabled", ["--rope", "disable"]),
    ("rope_default_active", ["--rope", "default"]),
    ("rope_config_raw", ["--rope", "config"]),
    ("no_special_tokens", ["--add-special-tokens", "false"]),
    ("torch_conv", ["--torch-conv"]),
    ("torch_gated_norm", ["--torch-gated-norm"]),
]


SLOW_ABLATION_CASES = [
    ("torch_conv_gated_norm", ["--torch-conv", "--torch-gated-norm"]),
]


def make_ablation_base_args(args: argparse.Namespace) -> list[str]:
    base_args = [
        "--model",
        args.model,
        "--examples-json",
        str(args.examples_json),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--dtype",
        args.dtype,
        "--device",
        args.device,
        "--_child-run",
        "--print-json",
    ]
    if args.revision:
        base_args.extend(["--revision", args.revision])
    if args.token:
        base_args.extend(["--token", args.token])
    if args.trust_remote_code:
        base_args.append("--trust-remote-code")
    if args.max_examples is not None:
        base_args.extend(["--max-examples", str(args.max_examples)])
    if args.score_answers:
        base_args.append("--score-answers")
    if args.score_mode != "cached":
        base_args.extend(["--score-mode", args.score_mode])
    if args.score_rank_by != "mean":
        base_args.extend(["--score-rank-by", args.score_rank_by])
    if args.answer_top_k:
        base_args.extend(["--answer-top-k", str(args.answer_top_k)])
    return base_args


def run_ablation_suite(args: argparse.Namespace) -> int:
    script_path = str(Path(__file__).resolve())
    base_args = make_ablation_base_args(args)
    output_dir = args.ablation_output_dir
    tempdir = None
    if output_dir is None:
        tempdir = tempfile.TemporaryDirectory()
        output_dir = Path(tempdir.name)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    try:
        failures = 0
        ablation_cases = ABLATION_CASES + (SLOW_ABLATION_CASES if args.include_slow_ablations else [])
        for label, extra_args in ablation_cases:
            output_path = output_dir / f"{label}.json"
            command = [sys.executable, script_path, *base_args, *extra_args]
            print(f"\n[ablation] {label}: {' '.join(extra_args) if extra_args else '(baseline)'}", file=sys.stderr)
            with output_path.open("w", encoding="utf-8") as handle:
                proc = subprocess.run(command, stdout=handle, stderr=subprocess.PIPE, text=True, check=False)
            if proc.returncode != 0:
                failures += 1
                print(f"[ablation] {label} failed with exit code {proc.returncode}", file=sys.stderr)
                if proc.stderr:
                    print(proc.stderr[-4000:], file=sys.stderr)
                continue

            result = json.loads(output_path.read_text(encoding="utf-8"))
            print_compact_result(result, label)

        if args.ablation_output_dir is not None:
            print(f"\n[ablation] JSON outputs written to {output_dir}", file=sys.stderr)
        return 1 if failures else 0
    finally:
        if tempdir is not None:
            tempdir.cleanup()


def main() -> None:
    args = parse_args()
    if args.ablation_suite:
        raise SystemExit(run_ablation_suite(args))
    if args.fork and not args._child_run:
        raise SystemExit(run_with_fork(args))

    result = run_one(args, args.model, args.revision)

    if args.print_json:
        print(json.dumps(result, indent=2))
        return

    print_result(result)


if __name__ == "__main__":
    main()
