#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "transformers==5.7.0",
#     "flash-linear-attention==0.5.0",
#     "torch==2.7.1",
#     "accelerate",
# ]
#
# [tool.uv.sources]
# torch = [
#     { index = "pytorch-cu128", marker = "sys_platform == 'linux'" },
# ]
#
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
# ///

"""Minimal NIAH reproducer for Olmo-Hybrid-Instruct-SFT-7B running in DroPE mode.

The released SFT checkpoint was trained with DroPE (no RoPE), but transformers
5.7.0 silently mutates the config's `rope_parameters: null` into RoPE-θ=10000
during loading. The fix is to override `cfg.rope_parameters = {"rope_theta": None}`
*before* `from_pretrained` — the explicit None inside the dict survives the
standardization chain, and the modeling code's NoPE branch fires correctly.

Usage:
    uv run --script scripts/simple_run_ruler.py                   # L=4096 smoke
    uv run --script scripts/simple_run_ruler.py --length 32768    # longer run

Expected output: all 5 fractions return the correct magic number ("hit=True").
With the bug present (no override), all return things like "10." or "8.".
"""

import argparse
import random

import torch

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "allenai/Olmo-Hybrid-7B"
SEED = 12345
FILLERS = [
    "The Atlantic Ocean covers a vast area between continents.",
    "Many species of birds migrate thousands of miles each year.",
    "Mountains form by tectonic plate movements over geological time.",
    "Coffee beans are roasted to develop their flavor profiles.",
    "Solar panels convert sunlight into usable electrical energy.",
    "Glaciers carve U-shaped valleys as they slowly advance.",
    "Honeybees communicate the location of nectar through waggle dances.",
    "Limestone caves form from the slow dissolution of carbonate rock.",
]


def load_drope(model_id):
    """Load the SFT checkpoint in true DroPE / NoPE mode."""
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

    # KEY OVERRIDE: explicit None inside the dict, not a top-level null.
    # transformers' setdefault chain preserves explicit None but injects
    # default_theta=10000 if the value is missing entirely.
    cfg.rope_parameters = {"rope_theta": None}

    model = (
        AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
            config=cfg,
        )
        .to("cuda:0")
        .eval()
    )

    # Sanity: this MUST print "rotary_emb is None: True" for DroPE.
    assert model.model.rotary_emb is None, (
        f"NoPE override failed — rotary_emb is {type(model.model.rotary_emb).__name__}. "
        f"rope_parameters in model config: {model.config.rope_parameters}"
    )
    print(f"✓ DroPE active: rotary_emb is None, rope_parameters={model.config.rope_parameters}")
    return model


def build_niah_prompt(tokenizer, target_len, magic, insert_frac, keyword="Atlantis"):
    needle = f"The magic number for {keyword} is {magic}."
    rng = random.Random(SEED + int(magic))
    parts = []
    while True:
        parts.append(rng.choice(FILLERS))
        if len(tokenizer.encode(" ".join(parts))) > target_len + 100:
            break
    haystack = " ".join(parts)
    ids = tokenizer.encode(haystack)[: target_len - 80]
    haystack = tokenizer.decode(ids, skip_special_tokens=True)
    p = int(len(haystack) * insert_frac)
    haystack = haystack[:p] + " " + needle + " " + haystack[p:]
    user_msg = haystack + f"\n\nWhat is the magic number for {keyword}? Reply with just the number."
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_msg}],
        tokenize=False,
        add_generation_prompt=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=int, default=4096)
    ap.add_argument("--fractions", type=float, nargs="+", default=[0.1, 0.25, 0.5, 0.75, 0.9])
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = load_drope(MODEL_ID)

    rng = random.Random(SEED)
    correct = 0
    for frac in args.fractions:
        magic = rng.randint(100000, 999999)
        prompt = build_niah_prompt(tokenizer, args.length, magic, frac)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
        n_in = inputs.input_ids.shape[1]
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=24,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        gen = tokenizer.decode(out[0, n_in:], skip_special_tokens=True).strip()
        hit = str(magic) in gen
        correct += int(hit)
        print(f"  frac={frac:.2f} len={n_in} magic={magic} gen={gen[:50]!r} hit={hit}")

    print(f"\nNIAH @ L={args.length}: {correct}/{len(args.fractions)}")


if __name__ == "__main__":
    main()
