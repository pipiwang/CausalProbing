#!/usr/bin/env bash
set -euo pipefail

# Paired-seed analysis for ResNet18 Gleason binary Grade Group >= 2 vs other.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PROJECT_DIR=${PROJECT_DIR:-${DEFAULT_PROJECT_DIR}}
VARIABLES=${VARIABLES:-"bmi psa_value scan_prostate_volume_ml max_pirads pirads_high age dre_abnormal smoking_encoded alcohol_encoded cardio_any respiratory_any diabetes renal_metabolic_any"}

export PROJECT_DIR
export BINARY_POSITIVE_MIN=2
export LABEL_DEFINITION=${LABEL_DEFINITION:-grade_group_ge_2}
export MODEL=${MODEL:-resnet18}
export TRAIN_MODE=${TRAIN_MODE:-scratch}
export CSV_OUTPUT=${CSV_OUTPUT:-${PROJECT_DIR}/results/ruleout_stats_resnet18_best_auc_seed_paired_auc_balanced_acc_sens80spec.csv}

if [[ -z "${TARGETS:-}" ]]; then
  read -r -a VARIABLE_LIST <<< "${VARIABLES}"
  TARGET_LIST=()
  for VARIABLE in "${VARIABLE_LIST[@]}"; do
    if [[ "${VARIABLE}" == "none" || "${VARIABLE}" == "baseline" || "${VARIABLE}" == "ruleout_none" ]]; then
      continue
    fi
    TARGET_LIST+=("ruleout_${VARIABLE}")
  done
  if [[ ${#TARGET_LIST[@]} -gt 0 ]]; then
    TARGETS="$(IFS=,; echo "${TARGET_LIST[*]}")"
    export TARGETS
  fi
fi

bash "${SCRIPT_DIR}/run_analyze_gleason_ruleout_seed_paired.sh"
