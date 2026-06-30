#!/usr/bin/env bash
set -euo pipefail

# Compare PROMIS external-test metrics for baseline vs ruleout models across
# matched random seeds.
#
# Usage from the code/project directory:
#   bash script/run_analyze_gleason_promis_external_seed_paired.sh
#
# Common overrides:
#   TASK_TYPE=binary SELECTION=best_auc bash script/run_analyze_gleason_promis_external_seed_paired.sh
#   TASK_TYPE=ordinal SELECTION=best_qwk bash script/run_analyze_gleason_promis_external_seed_paired.sh
#   TARGETS=ruleout_psa_value,ruleout_age bash script/run_analyze_gleason_promis_external_seed_paired.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PROJECT_DIR=${PROJECT_DIR:-${DEFAULT_PROJECT_DIR}}
OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT_DIR}/output_cls}
BASELINE=${BASELINE:-ruleout_none}
MODEL=${MODEL:-profound_conv}
TRAIN_MODE=${TRAIN_MODE:-fintune}
TASK_TYPE=${TASK_TYPE:-binary}
BINARY_POSITIVE_MIN=${BINARY_POSITIVE_MIN:-2}
ORDINAL_LEVELS=${ORDINAL_LEVELS:-5}
SEED_BOOTSTRAP_ITERATIONS=${SEED_BOOTSTRAP_ITERATIONS:-10000}
SEED_BOOTSTRAP_SEED=${SEED_BOOTSTRAP_SEED:-0}
BOOTSTRAP_ITERATIONS=${BOOTSTRAP_ITERATIONS:-10000}
BOOTSTRAP_SEED=${BOOTSTRAP_SEED:-0}
PAIR_KEY=${PAIR_KEY:-image_npy_path}
CLUSTER_KEY=${CLUSTER_KEY:-person_id}
PREDICTION_COLUMN=${PREDICTION_COLUMN:-pred_threshold_default_0_5}
TARGETS=${TARGETS:-}
ANALYSIS=${ANALYSIS:-aggregate}

if [[ "${TASK_TYPE}" == "binary" ]]; then
  ROOT=${ROOT:-${OUTPUT_DIR}/gleason/binary/grade_group_ge_${BINARY_POSITIVE_MIN}}
  SELECTION=${SELECTION:-best_auc}
  METRICS=${METRICS:-test_auc test_balanced_acc test_sens_at_80_spec}
elif [[ "${TASK_TYPE}" == "ordinal" ]]; then
  ROOT=${ROOT:-${OUTPUT_DIR}/gleason/ordinal/grade_group_ordinal_${ORDINAL_LEVELS}levels}
  SELECTION=${SELECTION:-best_qwk}
  METRICS=${METRICS:-test_qwk test_auc test_balanced_acc}
else
  echo "Unsupported TASK_TYPE=${TASK_TYPE}; expected binary or ordinal" >&2
  exit 1
fi

CSV_OUTPUT=${CSV_OUTPUT:-${PROJECT_DIR}/results/promis_external_ruleout_stats_${TASK_TYPE}_${SELECTION}_seed_paired.csv}

cd "${PROJECT_DIR}"

if [[ ! -f "analyze_gleason_ruleout_stats.py" ]]; then
  echo "Could not find analyze_gleason_ruleout_stats.py in PROJECT_DIR=${PROJECT_DIR}" >&2
  exit 1
fi

if [[ -z "${TARGETS}" ]]; then
  TARGETS=$(
    ROOT="${ROOT}" \
    BASELINE="${BASELINE}" \
    MODEL="${MODEL}" \
    TRAIN_MODE="${TRAIN_MODE}" \
    SELECTION="${SELECTION}" \
    python - <<'PY'
import os
from pathlib import Path

from analyze_gleason_ruleout_stats import metric_file_candidates

root = Path(os.environ["ROOT"])
baseline = os.environ["BASELINE"]
model = os.environ["MODEL"]
train_mode = os.environ["TRAIN_MODE"]
selection = os.environ["SELECTION"]
targets = []

for ruleout_dir in sorted(root.glob("ruleout_*")):
    if not ruleout_dir.is_dir() or ruleout_dir.name == baseline:
        continue
    seed_root = ruleout_dir / model / train_mode
    has_promis_metrics = False
    for seed_dir in sorted(seed_root.iterdir() if seed_root.exists() else []):
        if not seed_dir.is_dir() or not seed_dir.name.isdigit():
            continue
        candidates = metric_file_candidates(
            root,
            ruleout_dir.name,
            model,
            train_mode,
            seed_dir.name,
            selection,
            "promis_external_metrics",
        )
        if any(path.exists() for path in candidates):
            has_promis_metrics = True
            break
    if has_promis_metrics:
        targets.append(ruleout_dir.name)

print(",".join(targets))
PY
  )
fi
if [[ -z "${TARGETS}" ]]; then
  echo "No PROMIS external target metric files found under ROOT=${ROOT}" >&2
  echo "Run PROMIS external testing first, or set TARGETS=ruleout_x,ruleout_y explicitly." >&2
  exit 1
fi

CMD=(
  python -u analyze_gleason_ruleout_stats.py
  --analysis "${ANALYSIS}"
  --root "${ROOT}"
  --baseline "${BASELINE}"
  --model "${MODEL}"
  --train_mode "${TRAIN_MODE}"
  --selection "${SELECTION}"
  --metric_prefix promis_external_metrics
  --prediction_prefix promis_external_predictions
  --metrics ${METRICS}
  --seed_bootstrap_iterations "${SEED_BOOTSTRAP_ITERATIONS}"
  --seed_bootstrap_seed "${SEED_BOOTSTRAP_SEED}"
  --bootstrap_iterations "${BOOTSTRAP_ITERATIONS}"
  --bootstrap_seed "${BOOTSTRAP_SEED}"
  --pair_key "${PAIR_KEY}"
  --cluster_key "${CLUSTER_KEY}"
  --prediction_column "${PREDICTION_COLUMN}"
  --table_mode holistic
  --csv_output "${CSV_OUTPUT}"
)

if [[ -n "${TARGETS}" ]]; then
  IFS=',' read -r -a TARGET_LIST <<< "${TARGETS}"
  CMD+=(--targets)
  for TARGET in "${TARGET_LIST[@]}"; do
    TARGET="$(echo "${TARGET}" | xargs)"
    if [[ -n "${TARGET}" ]]; then
      CMD+=("${TARGET}")
    fi
  done
fi

echo "Running:"
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"

echo "Wrote: ${CSV_OUTPUT}"
