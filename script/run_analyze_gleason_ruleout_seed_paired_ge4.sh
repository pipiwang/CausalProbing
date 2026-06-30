#!/usr/bin/env bash
set -euo pipefail

# Paired-seed analysis for Gleason Grade Group >= 4 vs Grade Group 1-3.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VARIABLES=${VARIABLES:-"bmi psa_value scan_prostate_volume_ml max_pirads pirads_high age dre_abnormal smoking_encoded alcohol_encoded cardio_any respiratory_any diabetes renal_metabolic_any"}

export BINARY_POSITIVE_MIN=4
export LABEL_DEFINITION=${LABEL_DEFINITION:-grade_group_ge_4}

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
