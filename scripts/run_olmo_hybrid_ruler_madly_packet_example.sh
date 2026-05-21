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

# Fork mode: clone fork, run in isolated venv
FORK_URL="${FORK%@*}"
FORK_BRANCH="${FORK##*@}"

TMPDIR=$(mktemp -d)
trap "rm -rf '${TMPDIR}'" EXIT

VENV="${TMPDIR}/venv"
echo "[fork] Creating venv and installing ${FORK_URL} @ ${FORK_BRANCH} ..." >&2
uv venv --system-site-packages "${VENV}" >&2
uv pip install --python "${VENV}/bin/python" \
    "transformers @ git+${FORK_URL}@${FORK_BRANCH}" >&2

echo "[fork] Running with ${FORK} ..." >&2
"${VENV}/bin/python" scripts/run_olmo_hybrid_ruler_one.py "${BASE_ARGS[@]}"
