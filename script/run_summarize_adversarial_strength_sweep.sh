#!/usr/bin/env bash
set -euo pipefail

# Summarize adversarial loss-weight sweep metrics.
#
# Usage from the code/project directory:
#   bash script/run_summarize_adversarial_strength_sweep.sh
#
# Common overrides:
#   SEEDS="0 1" bash script/run_summarize_adversarial_strength_sweep.sh
#   SELECTION=best_balanced_acc bash script/run_summarize_adversarial_strength_sweep.sh
#   RULEOUTS="ruleout_psa_value" bash script/run_summarize_adversarial_strength_sweep.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PROJECT_DIR=${PROJECT_DIR:-${DEFAULT_PROJECT_DIR}}
OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT_DIR}/output_cls}
ROOT=${ROOT:-${OUTPUT_DIR}/adversarial_strength_sweep}
SELECTION=${SELECTION:-best_auc}
VARIANT=${VARIANT:-mri_only}
METRICS=${METRICS:-test_auc test_balanced_acc test_acc test_sensitivity test_specificity test_f1 test_loss}
PRIMARY_METRIC=${PRIMARY_METRIC:-test_auc}
TASK_TYPE=${TASK_TYPE:-binary}
LABEL_DEFINITION=${LABEL_DEFINITION:-grade_group_ge_2}
MODEL=${MODEL:-profound_conv}
TRAIN_MODE=${TRAIN_MODE:-fintune}
WEIGHTS=${WEIGHTS:-}
SEEDS=${SEEDS:-}
RULEOUTS=${RULEOUTS:-}

RESULTS_DIR=${RESULTS_DIR:-${PROJECT_DIR}/results}
LONG_CSV=${LONG_CSV:-${RESULTS_DIR}/adversarial_strength_sweep_${SELECTION}_${VARIANT}_long.csv}
SUMMARY_CSV=${SUMMARY_CSV:-${RESULTS_DIR}/adversarial_strength_sweep_${SELECTION}_${VARIANT}_summary.csv}
MISSING_CSV=${MISSING_CSV:-${RESULTS_DIR}/adversarial_strength_sweep_${SELECTION}_${VARIANT}_missing.csv}

cd "${PROJECT_DIR}"

if [[ ! -f "summarize_adversarial_strength_sweep.py" ]]; then
  echo "Could not find summarize_adversarial_strength_sweep.py in PROJECT_DIR=${PROJECT_DIR}" >&2
  exit 1
fi

CMD=(
  python -u summarize_adversarial_strength_sweep.py
  --root "${ROOT}"
  --task_type "${TASK_TYPE}"
  --label_definition "${LABEL_DEFINITION}"
  --model "${MODEL}"
  --train_mode "${TRAIN_MODE}"
  --selection "${SELECTION}"
  --variant "${VARIANT}"
  --metrics ${METRICS}
  --primary_metric "${PRIMARY_METRIC}"
  --long_csv "${LONG_CSV}"
  --summary_csv "${SUMMARY_CSV}"
  --missing_csv "${MISSING_CSV}"
)

if [[ -n "${WEIGHTS}" ]]; then
  CMD+=(--weights)
  read -r -a WEIGHT_LIST <<< "${WEIGHTS}"
  for WEIGHT in "${WEIGHT_LIST[@]}"; do
    CMD+=("${WEIGHT}")
  done
fi

if [[ -n "${SEEDS}" ]]; then
  CMD+=(--seeds)
  read -r -a SEED_LIST <<< "${SEEDS}"
  for SEED in "${SEED_LIST[@]}"; do
    CMD+=("${SEED}")
  done
fi

if [[ -n "${RULEOUTS}" ]]; then
  CMD+=(--ruleouts)
  read -r -a RULEOUT_LIST <<< "${RULEOUTS}"
  for RULEOUT in "${RULEOUT_LIST[@]}"; do
    CMD+=("${RULEOUT}")
  done
fi

echo "Running:"
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"
