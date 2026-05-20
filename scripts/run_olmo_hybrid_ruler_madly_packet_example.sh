#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 MODEL_OR_PATH [extra run_olmo_hybrid_ruler_one.py args...]" >&2
  echo "example: $0 /path/to/olmo-hybrid --device cuda --dtype bfloat16" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
MODEL="$1"
shift

cd "${REPO_ROOT}"
PYTHONPATH=src python scripts/run_olmo_hybrid_ruler_one.py \
  --model "${MODEL}" \
  --example madly-packet-5449368 \
  --fallback force \
  "$@"
