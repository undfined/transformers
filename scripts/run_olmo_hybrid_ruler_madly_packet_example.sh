#!/usr/bin/env bash
set -euo pipefail

USE_QK_L2NORM=0
REVISION=""
FORK=""
MODEL=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --l2norm) USE_QK_L2NORM=1; shift ;;
    --no-l2norm) USE_QK_L2NORM=0; shift ;;
    --revision) REVISION="$2"; shift 2 ;;
    --fork) FORK="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    *) break ;;
  esac
done

if [[ -z "${MODEL}" ]]; then
  if [[ $# -lt 1 ]]; then
    echo "usage: $0 [--l2norm|--no-l2norm] [--revision BRANCH] [--fork URL@BRANCH] --model MODEL_OR_PATH [extra args...]" >&2
    echo "example: $0 --fork https://github.com/yanhong-lbh/transformers.git@olmo-3.5-hybrid --model /path/to/olmo-hybrid --device cuda" >&2
    exit 2
  fi
  MODEL="$1"
  shift
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

L2NORM_ARG=(); [[ "${USE_QK_L2NORM}" -eq 1 ]] && L2NORM_ARG=(--l2norm)
REVISION_ARG=(); [[ -n "${REVISION}" ]] && REVISION_ARG=(--revision "${REVISION}")

BASE_ARGS=(
  --model "${MODEL}"
  --fallback force
  ${L2NORM_ARG[@]+"${L2NORM_ARG[@]}"}
  ${REVISION_ARG[@]+"${REVISION_ARG[@]}"}
  "$@"
)

# No fork: run normally against local src/
if [[ -z "${FORK}" ]]; then
  PYTHONPATH=src python scripts/run_olmo_hybrid_ruler_one.py "${BASE_ARGS[@]}"
  exit 0
fi

# Fork mode: clone fork, run both with different PYTHONPATH, compare
FORK_URL="${FORK%@*}"
FORK_BRANCH="${FORK##*@}"

TMPDIR=$(mktemp -d)
trap "rm -rf '${TMPDIR}'" EXIT

VENV="${TMPDIR}/venv"
echo "[fork] Creating venv and installing ${FORK_URL} @ ${FORK_BRANCH} ..." >&2
uv venv --system-site-packages "${VENV}" >&2
uv pip install --python "${VENV}/bin/python" \
    "transformers @ git+${FORK_URL}@${FORK_BRANCH}" >&2

LOCAL_JSON="${TMPDIR}/local.json"
FORK_JSON="${TMPDIR}/fork.json"

# Local run (may fail if model type not supported in local transformers)
echo "[local] Running with local src/ ..." >&2
if PYTHONPATH=src python scripts/run_olmo_hybrid_ruler_one.py \
    "${BASE_ARGS[@]}" --print-json > "${LOCAL_JSON}"; then
  echo "[local] Done." >&2
else
  echo "[local] Failed (model type likely not in local transformers — skipping local run)." >&2
  LOCAL_JSON=""
fi

# Fork run
echo "[fork] Running with ${FORK} ..." >&2
"${VENV}/bin/python" scripts/run_olmo_hybrid_ruler_one.py \
    "${BASE_ARGS[@]}" --print-json > "${FORK_JSON}"
echo "[fork] Done." >&2

# Print results and comparison
python3 - "${LOCAL_JSON:-}" "${FORK_JSON}" "${FORK}" <<'PYEOF'
import json, sys

local_file, fork_file, fork_label = sys.argv[1], sys.argv[2], sys.argv[3]

def grade(r):
    return "PASS" if r["exact_match"] else ("digit-match" if r["digit_match"] else "FAIL")

def print_result(r, label):
    print(f"\n{'='*60}\n  {label}\n{'='*60}")
    print(f"  device:       {r['device']}")
    print(f"  dtype:        {r['dtype']}")
    print(f"  l2norm:       {r['l2norm']}")
    print(f"  input tokens: {r['input_tokens']}")
    print(f"  GDN calls:    {r['gdn_calls']}")
    print(f"  exact match:  {r['exact_match']}  ({grade(r)})")
    print(f"  continuation: {repr(r['continuation'][:120])}")

local_result = None
if local_file:
    with open(local_file) as f:
        local_result = json.load(f)
    print_result(local_result, "LOCAL")

with open(fork_file) as f:
    fork_result = json.load(f)
print_result(fork_result, f"FORK  {fork_label}")

if local_result:
    print(f"\n{'='*60}\n  COMPARISON\n{'='*60}")
    print(f"  local: {grade(local_result)}")
    print(f"  fork:  {grade(fork_result)}")
PYEOF
