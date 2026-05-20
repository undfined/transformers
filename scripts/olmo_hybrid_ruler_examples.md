# OLMo Hybrid RULER One-Off Examples

Run the built-in madly-packet repro prompt:

```bash
scripts/run_olmo_hybrid_ruler_madly_packet_example.sh /path/to/olmo-hybrid --device cuda --dtype bfloat16
```

Equivalent direct invocation:

```bash
PYTHONPATH=src python scripts/run_olmo_hybrid_ruler_one.py \
  --model /path/to/olmo-hybrid \
  --example madly-packet-5449368 \
  --device cuda \
  --dtype bfloat16 \
  --fallback force
```

The downloaded failure continuations are kept in:

```text
scripts/olmo_hybrid_ruler_niah_s_1_4096_predictions_failures.example.jsonl
```

The matching downloaded request file had literal `<<truncated ...>>` markers in `request.context`,
so the built-in prompt reconstructs a full RULER/NIAH-style prompt with the same key and label:
`madly-packet -> 5449368`.
