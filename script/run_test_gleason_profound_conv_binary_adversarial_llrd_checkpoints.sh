#!/usr/bin/env bash
set -euo pipefail

# Direct node checkpoint tester for a binary Gleason adversarial run.
#
# Usage:
#   bash script/run_test_gleason_profound_conv_binary_adversarial_llrd_checkpoints.sh
#
# Common overrides:
#   SEED=1 BATCH_SIZE=4 bash script/run_test_gleason_profound_conv_binary_adversarial_llrd_checkpoints.sh
#   ADVERSARIAL_VARIABLE=none bash script/run_test_gleason_profound_conv_binary_adversarial_llrd_checkpoints.sh
#   CHECKPOINTS="best last" bash script/run_test_gleason_profound_conv_binary_adversarial_llrd_checkpoints.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PROJECT_DIR=${PROJECT_DIR:-${DEFAULT_PROJECT_DIR}}

CSV_PATH=${CSV_PATH:-data/gleason_classification.csv}
PRETRAIN=${PRETRAIN:-checkpoints/profound_conv_checkpoint-799.pth}
OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT_DIR}/output_cls}
LOG_DIR=${LOG_DIR:-${PROJECT_DIR}/output_cls}
RUN_LOG_DIR=${RUN_LOG_DIR:-${PROJECT_DIR}/logs}

SEED=${SEED:-0}
BATCH_SIZE=${BATCH_SIZE:-8}
NUM_WORKERS=${NUM_WORKERS:-8}
CROP_SPATIAL_SIZE=${CROP_SPATIAL_SIZE:-64,256,256}
PIN_MEM=${PIN_MEM:-false}
DEVICE=${DEVICE:-cuda}
BINARY_POSITIVE_MIN=${BINARY_POSITIVE_MIN:-2}
LABEL_DEFINITION=grade_group_ge_${BINARY_POSITIVE_MIN}

# Must match the training run. Set to "none" to test a baseline run.
ADVERSARIAL_VARIABLE=${ADVERSARIAL_VARIABLE:-psa_value}
ADVERSARIAL_COLUMN=${ADVERSARIAL_COLUMN:-}
ADVERSARIAL_OBSERVED_COLUMN=${ADVERSARIAL_OBSERVED_COLUMN:-}
ADVERSARIAL_NUM_CLASSES=${ADVERSARIAL_NUM_CLASSES:-}

if [[ "${ADVERSARIAL_VARIABLE}" == "none" ]]; then
  ADVERSARIAL_NAME=ruleout_none
else
  ADVERSARIAL_NAME=ruleout_${ADVERSARIAL_VARIABLE}
fi

EXP_DIR=${OUTPUT_DIR}/gleason/binary/${LABEL_DEFINITION}/${ADVERSARIAL_NAME}/profound_conv/fintune/${SEED}

if [[ -n "${CHECKPOINTS:-}" ]]; then
  read -r -a CHECKPOINT_NAMES <<< "${CHECKPOINTS}"
else
  CHECKPOINT_NAMES=(best best_auc best_balanced_acc best_loss last)
fi

mkdir -p "${RUN_LOG_DIR}"
cd "${PROJECT_DIR}"

if [[ ! -f "test_gleason_classification.py" ]]; then
  echo "Could not find test_gleason_classification.py in PROJECT_DIR=${PROJECT_DIR}" >&2
  echo "Set PROJECT_DIR to your code directory, for example:" >&2
  echo "  PROJECT_DIR=/path/to/CausalProbing bash ${BASH_SOURCE[0]}" >&2
  exit 1
fi

RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="${RUN_LOG_DIR}/test_pconv_binary_adv_${RUN_ID}.log"
exec > >(tee -a "${RUN_LOG}") 2>&1

echo "Project directory: ${PROJECT_DIR}"
echo "Run log: ${RUN_LOG}"
echo "Binary label definition: grade_group >= ${BINARY_POSITIVE_MIN}"
echo "Experiment directory: ${EXP_DIR}"
echo "Checkpoints: ${CHECKPOINT_NAMES[*]}"
echo "Host: $(hostname)"
nvidia-smi || true

for CKPT_NAME in "${CHECKPOINT_NAMES[@]}"; do
  CKPT_PATH=${EXP_DIR}/${CKPT_NAME}.pth.tar
  if [[ ! -f "${CKPT_PATH}" ]]; then
    echo "Skipping missing checkpoint: ${CKPT_PATH}"
    continue
  fi

  CMD=(
    python -u test_gleason_classification.py
    --csv_path "${CSV_PATH}"
    --split_col split
    --image_path_col image_npy_path
    --model profound_conv
    --train fintune
    --pretrain "${PRETRAIN}"
    --task_type binary
    --label_col grade_group
    --binary_positive_min "${BINARY_POSITIVE_MIN}"
    --crop_spatial_size "${CROP_SPATIAL_SIZE}"
    --batch_size "${BATCH_SIZE}"
    --num_workers "${NUM_WORKERS}"
    --device "${DEVICE}"
    --lr 1e-4
    --weight_decay 0.05
    --warmup_epochs 5
    --layer_decay 0.6
    --layer_decay_type group
    --seed "${SEED}"
    --output_dir "${OUTPUT_DIR}"
    --log_dir "${LOG_DIR}"
    --checkpoint "${CKPT_PATH}"
    --metrics_output "${EXP_DIR}/test_metrics_${CKPT_NAME}_mri_only.json"
  )

  if [[ "${PIN_MEM}" == "true" ]]; then
    CMD+=(--pin_mem)
  else
    CMD+=(--no_pin_mem)
  fi

  if [[ "${ADVERSARIAL_VARIABLE}" != "none" ]]; then
    CMD+=(
      --adversarial_variable "${ADVERSARIAL_VARIABLE}"
      --drop_adversarial_head
    )
    if [[ -n "${ADVERSARIAL_COLUMN}" ]]; then
      CMD+=(--adversarial_column "${ADVERSARIAL_COLUMN}")
    fi
    if [[ -n "${ADVERSARIAL_OBSERVED_COLUMN}" ]]; then
      CMD+=(--adversarial_observed_column "${ADVERSARIAL_OBSERVED_COLUMN}")
    fi
    if [[ -n "${ADVERSARIAL_NUM_CLASSES}" ]]; then
      CMD+=(--adversarial_num_classes "${ADVERSARIAL_NUM_CLASSES}")
    fi
  fi

  echo "Running:"
  printf ' %q' "${CMD[@]}"
  echo
  "${CMD[@]}"
done
