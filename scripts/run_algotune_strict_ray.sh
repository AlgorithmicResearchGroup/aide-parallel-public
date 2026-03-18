#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TASK_NAME="${1:-base64_encoding}"
RESULT_JSON="${2:-/tmp/algotune-strict-e2e-${TASK_NAME}-ray.json}"

# Load local secrets/config if present.
if [[ -f "$ROOT_DIR/.env.local" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.env.local"
  set +a
elif [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.env"
  set +a
fi

if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
  echo "ANTHROPIC_API_KEY must be set, or provided in .env.local / .env." >&2
  exit 1
fi

export PATH="$ROOT_DIR/.venv-algotune/bin:$PATH"
export AIDE_MODEL="${AIDE_MODEL:-claude-sonnet-4-20250514}"
export AIDE_FEEDBACK_MODEL="${AIDE_FEEDBACK_MODEL:-claude-sonnet-4-20250514}"
export ALGOTUNE_HF_REVISION="${ALGOTUNE_HF_REVISION:-fc3744ffd7eebaa9e9b55427e2cda440955fdd2d}"
export AIDE_ALGOTUNE_DIRECT_SINGLE_EXPERIMENT=0
if [[ -n "${MLFLOW_TRACKING_URI:-}" || -n "${MLFLOW_EXPERIMENT_NAME:-}" ]]; then
  export AIDE_ENABLE_MLFLOW="${AIDE_ENABLE_MLFLOW:-1}"
fi

TRACKING_EXPERIMENT="${TRACKING_EXPERIMENT:-${MLFLOW_EXPERIMENT_NAME:-}}"

cmd=(
  ./cli/aide-run
  --local
  --task algotune
  --at-task "$TASK_NAME"
  --num-experiments 4
  --num-iterations 1
  --steps 4
  --cpus-per-experiment 1
  --result-json "$RESULT_JSON"
)

if [[ -n "$TRACKING_EXPERIMENT" ]]; then
  cmd+=(--tracking-experiment "$TRACKING_EXPERIMENT")
fi

"${cmd[@]}"
