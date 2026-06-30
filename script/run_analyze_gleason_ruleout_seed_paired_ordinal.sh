#!/usr/bin/env bash
set -euo pipefail

# Compare ordinal baseline vs ordinal ruleout models across matched random seeds.
#
# This uses metric JSON files by default. For each matched seed, it computes:
#   delta = ruleout_metric - baseline_metric
# and runs seed-paired bootstrap tests over seed-level deltas.
#
# Usage from the code/project directory:
#   bash script/run_analyze_gleason_ruleout_seed_paired_ordinal.sh
#
# Common overrides:
#   SELECTION=best_qwk bash script/run_analyze_gleason_ruleout_seed_paired_ordinal.sh
#   TARGETS=ruleout_bmi,ruleout_age bash script/run_analyze_gleason_ruleout_seed_paired_ordinal.sh
#   METRICS="test_qwk test_auc" bash script/run_analyze_gleason_ruleout_seed_paired_ordinal.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PROJECT_DIR=${PROJECT_DIR:-${DEFAULT_PROJECT_DIR}}
OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT_DIR}/output_cls}
ROOT=${ROOT:-${OUTPUT_DIR}/gleason/ordinal/grade_group_ordinal_5levels}
BASELINE=${BASELINE:-ruleout_none}
MODEL=${MODEL:-profound_conv}
TRAIN_MODE=${TRAIN_MODE:-fintune}
SELECTION=${SELECTION:-best_qwk}
METRICS=${METRICS:-test_qwk test_auc test_balanced_acc}
SEED_BOOTSTRAP_ITERATIONS=${SEED_BOOTSTRAP_ITERATIONS:-10000}
SEED_BOOTSTRAP_SEED=${SEED_BOOTSTRAP_SEED:-0}
TARGETS=${TARGETS:-}
CSV_OUTPUT=${CSV_OUTPUT:-${PROJECT_DIR}/results/ruleout_stats_ordinal_${SELECTION}_seed_paired.csv}

cd "${PROJECT_DIR}"

if [[ ! -f "analyze_gleason_ruleout_stats.py" ]]; then
  echo "Could not find analyze_gleason_ruleout_stats.py in PROJECT_DIR=${PROJECT_DIR}" >&2
  exit 1
fi

CMD=(
  python -u analyze_gleason_ruleout_stats.py
  --analysis aggregate
  --root "${ROOT}"
  --baseline "${BASELINE}"
  --model "${MODEL}"
  --train_mode "${TRAIN_MODE}"
  --selection "${SELECTION}"
  --metrics ${METRICS}
  --seed_bootstrap_iterations "${SEED_BOOTSTRAP_ITERATIONS}"
  --seed_bootstrap_seed "${SEED_BOOTSTRAP_SEED}"
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
