#!/usr/bin/env bash
set -euo pipefail

# Direct node runner for one binary Gleason adversarial run.
#
# Usage from the code/project directory:
#   bash script/run_train_gleason_profound_conv_binary_adversarial_llrd.sh
#
# Common overrides:
#   SEED=1 BATCH_SIZE=4 bash script/run_train_gleason_profound_conv_binary_adversarial_llrd.sh
#   ADVERSARIAL_VARIABLE=bmi bash script/run_train_gleason_profound_conv_binary_adversarial_llrd.sh
#   ADVERSARIAL_VARIABLE=none bash script/run_train_gleason_profound_conv_binary_adversarial_llrd.sh

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
EPOCHS=${EPOCHS:-100}
NUM_WORKERS=${NUM_WORKERS:-8}
CROP_SPATIAL_SIZE=${CROP_SPATIAL_SIZE:-64,256,256}
PIN_MEM=${PIN_MEM:-false}
DEVICE=${DEVICE:-cuda}

# Set ADVERSARIAL_VARIABLE=none for a baseline-shaped control run.
# Common choices: psa_value, age, bmi, cardio_any, respiratory_any, diabetes, renal_metabolic_any.
ADVERSARIAL_VARIABLE=${ADVERSARIAL_VARIABLE:-psa_value}
ADVERSARIAL_COLUMN=${ADVERSARIAL_COLUMN:-}
ADVERSARIAL_OBSERVED_COLUMN=${ADVERSARIAL_OBSERVED_COLUMN:-}
ADVERSARIAL_NUM_CLASSES=${ADVERSARIAL_NUM_CLASSES:-}
ADVERSARIAL_LOSS_WEIGHT=${ADVERSARIAL_LOSS_WEIGHT:-1.0}
GRL_LAMBDA=${GRL_LAMBDA:-1.0}
GRL_SCHEDULE=${GRL_SCHEDULE:-dann}
GRL_GAMMA=${GRL_GAMMA:-10.0}

mkdir -p "${RUN_LOG_DIR}"
cd "${PROJECT_DIR}"

if [[ ! -f "main_gleason_classification.py" ]]; then
  echo "Could not find main_gleason_classification.py in PROJECT_DIR=${PROJECT_DIR}" >&2
  echo "Set PROJECT_DIR to your code directory, for example:" >&2
  echo "  PROJECT_DIR=/path/to/CausalProbing bash ${BASH_SOURCE[0]}" >&2
  exit 1
fi

if [[ "${ACTIVATE_VENV:-false}" == "true" ]]; then
  if [[ ! -f ".venv/bin/activate" ]]; then
    echo "ACTIVATE_VENV=true, but ${PROJECT_DIR}/.venv/bin/activate was not found" >&2
    exit 1
  fi
  source ".venv/bin/activate"
fi

RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="${RUN_LOG_DIR}/gleason_pconv_binary_adv_${RUN_ID}.log"
exec > >(tee -a "${RUN_LOG}") 2>&1

echo "Project directory: ${PROJECT_DIR}"
echo "Run log: ${RUN_LOG}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Log directory: ${LOG_DIR}"
echo "Seed: ${SEED}"
echo "Adversarial variable: ${ADVERSARIAL_VARIABLE}"
echo "Host: $(hostname)"
nvidia-smi || true

CMD=(
  python -u main_gleason_classification.py
  --csv_path "${CSV_PATH}"
  --split_col split
  --image_path_col image_npy_path
  --model profound_conv
  --train fintune
  --pretrain "${PRETRAIN}"
  --task_type binary
  --label_col grade_group
  --binary_positive_min 2
  --crop_spatial_size "${CROP_SPATIAL_SIZE}"
  --batch_size "${BATCH_SIZE}"
  --epochs "${EPOCHS}"
  --num_workers "${NUM_WORKERS}"
  --weighted_sampling
  --device "${DEVICE}"
  --lr 1e-4
  --weight_decay 0.05
  --warmup_epochs 5
  --layer_decay 0.6
  --layer_decay_type group
  --primary_metric auc
  --save_ckpt_interval 10
  --seed "${SEED}"
  --output_dir "${OUTPUT_DIR}"
  --log_dir "${LOG_DIR}"
)

if [[ "${PIN_MEM}" == "true" ]]; then
  CMD+=(--pin_mem)
else
  CMD+=(--no_pin_mem)
fi

if [[ "${ADVERSARIAL_VARIABLE}" != "none" ]]; then
  CMD+=(
    --adversarial_variable "${ADVERSARIAL_VARIABLE}"
    --adversarial_loss_weight "${ADVERSARIAL_LOSS_WEIGHT}"
    --grl_lambda "${GRL_LAMBDA}"
    --grl_schedule "${GRL_SCHEDULE}"
    --grl_gamma "${GRL_GAMMA}"
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
