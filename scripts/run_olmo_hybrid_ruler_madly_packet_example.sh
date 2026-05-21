#!/usr/bin/env bash
set -euo pipefail

USE_QK_L2NORM=0
REVISION=""
FORK=""
FORK_REVISION=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --l2norm) USE_QK_L2NORM=1; shift ;;
    --no-l2norm) USE_QK_L2NORM=0; shift ;;
    --revision) REVISION="$2"; shift 2 ;;
    --fork) FORK="$2"; shift 2 ;;
    --fork-revision) FORK_REVISION="$2"; shift 2 ;;
    *) break ;;
  esac
done

if [[ $# -lt 1 ]]; then
  echo "usage: $0 [--l2norm|--no-l2norm] [--revision BRANCH] [--fork MODEL [--fork-revision BRANCH]] MODEL_OR_PATH [extra args...]" >&2
  echo "example: $0 --fork user/olmo-fork --fork-revision my-branch /path/to/olmo-hybrid --device cuda" >&2
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

REVISION_ARG=()
if [[ -n "${REVISION}" ]]; then
  REVISION_ARG=(--revision "${REVISION}")
fi

FORK_ARG=()
if [[ -n "${FORK}" ]]; then
  FORK_ARG=(--fork "${FORK}")
  if [[ -n "${FORK_REVISION}" ]]; then
    FORK_ARG+=(--fork-revision "${FORK_REVISION}")
  fi
fi

cd "${REPO_ROOT}"
PYTHONPATH=src python scripts/run_olmo_hybrid_ruler_one.py \
  --model "${MODEL}" \
  --fallback force \
  "${L2NORM_ARG[@]}" \
  "${REVISION_ARG[@]}" \
  "${FORK_ARG[@]}" \
  "$@"

