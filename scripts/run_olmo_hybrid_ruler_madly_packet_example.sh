#!/usr/bin/env bash
set -euo pipefail

USE_QK_L2NORM=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --l2norm) USE_QK_L2NORM=1; shift ;;
    --no-l2norm) USE_QK_L2NORM=0; shift ;;
    *) break ;;
  esac
done

if [[ $# -lt 1 ]]; then
  echo "usage: $0 [--l2norm|--no-l2norm] MODEL_OR_PATH [extra run_olmo_hybrid_ruler_one.py args...]" >&2
  echo "example: $0 --l2norm /path/to/olmo-hybrid --device cuda --dtype bfloat16" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
MODEL="$1"
shift

L2NORM_ARG=()
if [[ "${USE_QK_L2NORM}" -eq 1 ]]; then
  L2NORM_ARG=(--l2norm)
fi

cd "${REPO_ROOT}"
PYTHONPATH=src python scripts/run_olmo_hybrid_ruler_one.py \
  --model "${MODEL}" \
  --example madly-packet-5449368 \
  --fallback force \
  "${L2NORM_ARG[@]}" \
  "$@"
