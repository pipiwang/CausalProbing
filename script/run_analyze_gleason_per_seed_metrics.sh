#!/usr/bin/env bash
set -euo pipefail

# Write one compact row per seed/run with main test metrics and same-seed
# deltas versus the baseline ruleout_none model.
#
# Usage from the code/project directory:
#   bash script/run_analyze_gleason_per_seed_metrics.sh
#
# Common overrides:
#   SELECTION=best_auc bash script/run_analyze_gleason_per_seed_metrics.sh
#   METRICS="test_auc test_balanced_acc test_sens_at_80_spec" bash script/run_analyze_gleason_per_seed_metrics.sh
#   METRICS=all bash script/run_analyze_gleason_per_seed_metrics.sh
#   TARGETS=ruleout_psa_value,ruleout_age bash script/run_analyze_gleason_per_seed_metrics.sh
#   BINARY_POSITIVE_MIN=3 bash script/run_analyze_gleason_per_seed_metrics.sh
#   METRIC_PREFIX=promis_external_metrics CSV_OUTPUT=results/promis_per_seed_metrics_best_auc.csv bash script/run_analyze_gleason_per_seed_metrics.sh

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
METRIC_PREFIX=${METRIC_PREFIX:-test_metrics}
METRICS=${METRICS:-test_auc test_balanced_acc test_sens_at_80_spec}
TARGETS=${TARGETS:-}

if [[ "${LABEL_DEFINITION}" == "grade_group_ge_2" ]]; then
  DEFAULT_CSV_OUTPUT=${PROJECT_DIR}/results/per_seed_metrics_${SELECTION}.csv
else
  DEFAULT_CSV_OUTPUT=${PROJECT_DIR}/results/per_seed_metrics_${LABEL_DEFINITION}_${SELECTION}.csv
fi
CSV_OUTPUT=${CSV_OUTPUT:-${DEFAULT_CSV_OUTPUT}}

cd "${PROJECT_DIR}"

if [[ ! -f "analyze_gleason_ruleout_stats.py" ]]; then
  echo "Could not find analyze_gleason_ruleout_stats.py in PROJECT_DIR=${PROJECT_DIR}" >&2
  exit 1
fi

CMD=(
  python -u analyze_gleason_ruleout_stats.py
  --analysis per_seed
  --root "${ROOT}"
  --baseline "${BASELINE}"
  --model "${MODEL}"
  --train_mode "${TRAIN_MODE}"
  --selection "${SELECTION}"
  --metric_prefix "${METRIC_PREFIX}"
  --metrics ${METRICS}
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
