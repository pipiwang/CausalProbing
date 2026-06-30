#!/usr/bin/env bash
set -euo pipefail

# Compare baseline vs ruleout models across matched random seeds.
#
# This uses metric JSON files, not per-scan prediction CSVs. For each matched
# seed, it computes:
#   delta = ruleout_metric - baseline_metric
# and runs paired tests over seed-level deltas.
#
# Usage from the code/project directory:
#   bash script/run_analyze_gleason_ruleout_seed_paired.sh
#
# Common overrides:
#   SELECTION=best_auc bash script/run_analyze_gleason_ruleout_seed_paired.sh
#   METRICS="test_auc test_balanced_acc test_sens_at_80_spec" bash script/run_analyze_gleason_ruleout_seed_paired.sh
#   TARGETS=ruleout_psa_value,ruleout_age bash script/run_analyze_gleason_ruleout_seed_paired.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PROJECT_DIR=${PROJECT_DIR:-${DEFAULT_PROJECT_DIR}}
OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT_DIR}/output_cls}
BINARY_POSITIVE_MIN=${BINARY_POSITIVE_MIN:-2}
LABEL_DEFINITION=${LABEL_DEFINITION:-grade_group_ge_${BINARY_POSITIVE_MIN}}
ROOT=${ROOT:-${OUTPUT_DIR}/gleason/binary/${LABEL_DEFINITION}}
BASELINE=${BASELINE:-ruleout_none}
MODEL=${MODEL:-profound_conv}
TRAIN_MODE=${TRAIN_MODE:-fintune}
SELECTION=${SELECTION:-best_auc}
METRICS=${METRICS:-test_auc test_balanced_acc test_sens_at_80_spec}
AGGREGATE_TEST=${AGGREGATE_TEST:-both}
SEED_BOOTSTRAP_ITERATIONS=${SEED_BOOTSTRAP_ITERATIONS:-10000}
SEED_BOOTSTRAP_SEED=${SEED_BOOTSTRAP_SEED:-0}
TARGETS=${TARGETS:-}
if [[ "${LABEL_DEFINITION}" == "grade_group_ge_2" ]]; then
  DEFAULT_CSV_OUTPUT=${PROJECT_DIR}/results/ruleout_stats_${SELECTION}_seed_paired_auc_balanced_acc_sens80spec.csv
else
  DEFAULT_CSV_OUTPUT=${PROJECT_DIR}/results/ruleout_stats_${LABEL_DEFINITION}_${SELECTION}_seed_paired_auc_balanced_acc_sens80spec.csv
fi
CSV_OUTPUT=${CSV_OUTPUT:-${DEFAULT_CSV_OUTPUT}}

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
  --aggregate_test "${AGGREGATE_TEST}"
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
