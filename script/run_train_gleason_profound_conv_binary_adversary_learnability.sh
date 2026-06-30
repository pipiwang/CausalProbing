#!/usr/bin/env bash
set -euo pipefail

# Direct compute-node runner for adversary learnability controls.
#
# This trains the adversarial head for each selected variable while disabling
# gradient reversal into the encoder:
#   GRL_LAMBDA=0
# The purpose is to test whether Z can predict each clinical variable before
# interpreting a full adversarial-removal experiment.
#
# Usage from the code/project directory:
#   bash script/run_train_gleason_profound_conv_binary_adversary_learnability.sh
#
# Common overrides:
#   EPOCHS=50 BATCH_SIZE=4 bash script/run_train_gleason_profound_conv_binary_adversary_learnability.sh
#   VARIABLES=psa_value,age,bmi bash script/run_train_gleason_profound_conv_binary_adversary_learnability.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PROJECT_DIR=${PROJECT_DIR:-${DEFAULT_PROJECT_DIR}}
RUNNER=${RUNNER:-${SCRIPT_DIR}/run_train_gleason_profound_conv_binary_adversarial_llrd.sh}

VARIABLES=${VARIABLES:-psa_value,age,bmi,cardio_any,diabetes,renal_metabolic_any}
IFS=',' read -r -a VARIABLE_LIST <<< "${VARIABLES}"

SEED=${SEED:-0}
EPOCHS=${EPOCHS:-100}
BATCH_SIZE=${BATCH_SIZE:-8}
NUM_WORKERS=${NUM_WORKERS:-8}
PIN_MEM=${PIN_MEM:-false}
DEVICE=${DEVICE:-cuda}

CSV_PATH=${CSV_PATH:-data/gleason_classification.csv}
PRETRAIN=${PRETRAIN:-checkpoints/profound_conv_checkpoint-799.pth}

OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT_DIR}/output_cls_adversary_learnability}
LOG_DIR=${LOG_DIR:-${PROJECT_DIR}/output_cls_adversary_learnability}
RUN_LOG_DIR=${RUN_LOG_DIR:-${PROJECT_DIR}/logs/adversary_learnability}

ADVERSARIAL_LOSS_WEIGHT=${ADVERSARIAL_LOSS_WEIGHT:-1.0}
GRL_LAMBDA=0.0
GRL_SCHEDULE=${GRL_SCHEDULE:-constant}
GRL_GAMMA=${GRL_GAMMA:-10.0}

mkdir -p "${RUN_LOG_DIR}"
cd "${PROJECT_DIR}"

if [[ ! -x "${RUNNER}" ]]; then
  echo "Runner is missing or not executable: ${RUNNER}" >&2
  exit 1
fi

MASTER_RUN_ID="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG="${RUN_LOG_DIR}/adversary_learnability_${MASTER_RUN_ID}.log"
exec > >(tee -a "${MASTER_LOG}") 2>&1

echo "Project directory: ${PROJECT_DIR}"
echo "Runner: ${RUNNER}"
echo "Variables: ${VARIABLES}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Run log directory: ${RUN_LOG_DIR}"
echo "Master log: ${MASTER_LOG}"
echo "Seed: ${SEED}"
echo "GRL_LAMBDA: ${GRL_LAMBDA}"
echo "GRL_SCHEDULE: ${GRL_SCHEDULE}"
echo "Host: $(hostname)"
nvidia-smi || true

for VARIABLE in "${VARIABLE_LIST[@]}"; do
  if [[ -z "${VARIABLE}" ]]; then
    continue
  fi

  VARIABLE="$(echo "${VARIABLE}" | xargs)"
  echo
  echo "============================================================"
  echo "Adversary learnability run: ${VARIABLE}"
  echo "============================================================"

  ADVERSARIAL_VARIABLE="${VARIABLE}" \
  ADVERSARIAL_COLUMN="" \
  ADVERSARIAL_OBSERVED_COLUMN="" \
  ADVERSARIAL_NUM_CLASSES="" \
  ADVERSARIAL_LOSS_WEIGHT="${ADVERSARIAL_LOSS_WEIGHT}" \
  GRL_LAMBDA="${GRL_LAMBDA}" \
  GRL_SCHEDULE="${GRL_SCHEDULE}" \
  GRL_GAMMA="${GRL_GAMMA}" \
  PROJECT_DIR="${PROJECT_DIR}" \
  CSV_PATH="${CSV_PATH}" \
  PRETRAIN="${PRETRAIN}" \
  OUTPUT_DIR="${OUTPUT_DIR}" \
  LOG_DIR="${LOG_DIR}" \
  RUN_LOG_DIR="${RUN_LOG_DIR}" \
  SEED="${SEED}" \
  EPOCHS="${EPOCHS}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  NUM_WORKERS="${NUM_WORKERS}" \
  PIN_MEM="${PIN_MEM}" \
  DEVICE="${DEVICE}" \
  bash "${RUNNER}"
done

echo
echo "All adversary learnability runs completed."
