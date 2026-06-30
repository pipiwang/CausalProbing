#!/usr/bin/env bash
set -euo pipefail

# Compare complete-case baseline vs ruleout runs for each variable.
#
# This reads from:
#   output_cls_complete_case/<variable>/gleason/binary/grade_group_ge_2/
# and writes separate CSVs under results_complete_case/ to avoid clashing with
# full-cohort result tables.
#
# Usage from the code directory after train/test jobs finish:
#   bash script/run_analyze_gleason_complete_case_seed_paired.sh
#   VARIABLES="psa_value scan_prostate_volume_ml" bash script/run_analyze_gleason_complete_case_seed_paired.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PROJECT_DIR=${PROJECT_DIR:-${DEFAULT_PROJECT_DIR}}
COMPLETE_CASE_OUTPUT_ROOT=${COMPLETE_CASE_OUTPUT_ROOT:-${PROJECT_DIR}/output_cls_complete_case}
RESULTS_DIR=${RESULTS_DIR:-${PROJECT_DIR}/results_complete_case}

VARIABLES=${VARIABLES:-"bmi psa_value scan_prostate_volume_ml max_pirads pirads_high age dre_abnormal smoking_encoded alcohol_encoded cardio_any respiratory_any diabetes renal_metabolic_any"}
BASELINE=${BASELINE:-ruleout_none}
MODEL=${MODEL:-profound_conv}
TRAIN_MODE=${TRAIN_MODE:-fintune}
SELECTION=${SELECTION:-best_auc}
METRICS=${METRICS:-test_auc test_balanced_acc test_sens_at_80_spec}
AGGREGATE_TEST=${AGGREGATE_TEST:-both}
SEED_BOOTSTRAP_ITERATIONS=${SEED_BOOTSTRAP_ITERATIONS:-10000}
SEED_BOOTSTRAP_SEED=${SEED_BOOTSTRAP_SEED:-0}
TABLE_MODE=${TABLE_MODE:-holistic}

cd "${PROJECT_DIR}"
mkdir -p "${RESULTS_DIR}"

if [[ ! -f "analyze_gleason_ruleout_stats.py" ]]; then
  echo "Could not find analyze_gleason_ruleout_stats.py in PROJECT_DIR=${PROJECT_DIR}" >&2
  exit 1
fi

read -r -a VARIABLE_LIST <<< "${VARIABLES}"
for VARIABLE in "${VARIABLE_LIST[@]}"; do
  ROOT=${COMPLETE_CASE_OUTPUT_ROOT}/${VARIABLE}/gleason/binary/grade_group_ge_2
  TARGET=ruleout_${VARIABLE}
  CSV_OUTPUT=${RESULTS_DIR}/ruleout_stats_complete_case_${VARIABLE}_${SELECTION}_seed_paired_auc_balanced_acc_sens80spec.csv

  CMD=(
    python -u analyze_gleason_ruleout_stats.py
    --analysis aggregate
    --root "${ROOT}"
    --baseline "${BASELINE}"
    --model "${MODEL}"
    --train_mode "${TRAIN_MODE}"
    --selection "${SELECTION}"
    --metrics ${METRICS}
    --targets "${TARGET}"
    --aggregate_test "${AGGREGATE_TEST}"
    --seed_bootstrap_iterations "${SEED_BOOTSTRAP_ITERATIONS}"
    --seed_bootstrap_seed "${SEED_BOOTSTRAP_SEED}"
    --table_mode "${TABLE_MODE}"
    --csv_output "${CSV_OUTPUT}"
  )

  echo "Running complete-case analysis for ${VARIABLE}:"
  printf ' %q' "${CMD[@]}"
  echo
  "${CMD[@]}"
  echo "Wrote: ${CSV_OUTPUT}"
done
