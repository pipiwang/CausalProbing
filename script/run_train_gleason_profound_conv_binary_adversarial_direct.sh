#!/usr/bin/env bash
set -euo pipefail

# Direct compute-node runner for MRI -> adversarial-variable training.
#
# This trains an MRI encoder plus adversarial head end-to-end to predict one
# clinical/adversarial variable. No Gleason head or Gleason loss is used.
#
# Usage on an Isambard compute node from the code/project directory:
#   bash script/run_train_gleason_profound_conv_binary_adversarial_direct.sh
#
# Common overrides:
#   ADVERSARIAL_VARIABLE=age bash script/run_train_gleason_profound_conv_binary_adversarial_direct.sh
#   EPOCHS=30 BATCH_SIZE=4 NUM_WORKERS=4 bash script/run_train_gleason_profound_conv_binary_adversarial_direct.sh
#   ADVERSARIAL_WEIGHTED_SAMPLING=false bash script/run_train_gleason_profound_conv_binary_adversarial_direct.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PROJECT_DIR=${PROJECT_DIR:-${DEFAULT_PROJECT_DIR}}
CONDA_ENV=${CONDA_ENV:-mae}
CONDA_SH=${CONDA_SH:-${HOME}/miniforge3/etc/profile.d/conda.sh}

CSV_PATH=${CSV_PATH:-data/gleason_classification.csv}
PRETRAIN=${PRETRAIN:-checkpoints/profound_conv_checkpoint-799.pth}

SEED=${SEED:-0}
BATCH_SIZE=${BATCH_SIZE:-8}
EPOCHS=${EPOCHS:-50}
NUM_WORKERS=${NUM_WORKERS:-8}
CROP_SPATIAL_SIZE=${CROP_SPATIAL_SIZE:-64,256,256}
PIN_MEM=${PIN_MEM:-false}
DEVICE=${DEVICE:-cuda}

ADVERSARIAL_VARIABLE=${ADVERSARIAL_VARIABLE:-psa_value}
ADVERSARIAL_COLUMN=${ADVERSARIAL_COLUMN:-}
ADVERSARIAL_OBSERVED_COLUMN=${ADVERSARIAL_OBSERVED_COLUMN:-}
ADVERSARIAL_NUM_CLASSES=${ADVERSARIAL_NUM_CLASSES:-}
ADVERSARIAL_LOSS_WEIGHT=${ADVERSARIAL_LOSS_WEIGHT:-1.0}
ADVERSARIAL_WEIGHTED_SAMPLING=${ADVERSARIAL_WEIGHTED_SAMPLING:-true}

LR=${LR:-1e-4}
MIN_LR=${MIN_LR:-0.0}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-5}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.05}
LAYER_DECAY=${LAYER_DECAY:-0.6}
LAYER_DECAY_TYPE=${LAYER_DECAY_TYPE:-group}
PRIMARY_METRIC=${PRIMARY_METRIC:-balanced_acc}
SAVE_CKPT_INTERVAL=${SAVE_CKPT_INTERVAL:-10}

OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT_DIR}/output_cls_adversarial_direct}
LOG_DIR=${LOG_DIR:-${PROJECT_DIR}/output_cls_adversarial_direct}
RUN_LOG_DIR=${RUN_LOG_DIR:-${PROJECT_DIR}/logs/adversarial_direct}

mkdir -p "${RUN_LOG_DIR}"
cd "${PROJECT_DIR}"

if [[ ! -f "train_adversarial_direct.py" ]]; then
  echo "Could not find train_adversarial_direct.py in PROJECT_DIR=${PROJECT_DIR}" >&2
  echo "Set PROJECT_DIR to your code directory, for example:" >&2
  echo "  PROJECT_DIR=/path/to/CausalProbing bash ${BASH_SOURCE[0]}" >&2
  exit 1
fi

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "Conda activation script not found: ${CONDA_SH}" >&2
  echo "Set CONDA_SH=/path/to/conda.sh if your Isambard conda install differs." >&2
  exit 1
fi
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="${RUN_LOG_DIR}/gleason_pconv_binary_adv_direct_${ADVERSARIAL_VARIABLE}_${RUN_ID}.log"
exec > >(tee -a "${RUN_LOG}") 2>&1

echo "Project directory: ${PROJECT_DIR}"
echo "Run log: ${RUN_LOG}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Log directory: ${LOG_DIR}"
echo "Seed: ${SEED}"
echo "Adversarial variable: ${ADVERSARIAL_VARIABLE}"
echo "Adversarial weighted sampling: ${ADVERSARIAL_WEIGHTED_SAMPLING}"
echo "Conda env: ${CONDA_ENV}"
echo "Host: $(hostname)"
nvidia-smi || true

CMD=(
  python -u train_adversarial_direct.py
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
  --device "${DEVICE}"
  --lr "${LR}"
  --min_lr "${MIN_LR}"
  --warmup_epochs "${WARMUP_EPOCHS}"
  --weight_decay "${WEIGHT_DECAY}"
  --layer_decay "${LAYER_DECAY}"
  --layer_decay_type "${LAYER_DECAY_TYPE}"
  --primary_metric "${PRIMARY_METRIC}"
  --save_ckpt_interval "${SAVE_CKPT_INTERVAL}"
  --seed "${SEED}"
  --output_dir "${OUTPUT_DIR}"
  --log_dir "${LOG_DIR}"
  --adversarial_variable "${ADVERSARIAL_VARIABLE}"
  --adversarial_loss_weight "${ADVERSARIAL_LOSS_WEIGHT}"
)

if [[ "${PIN_MEM}" == "true" ]]; then
  CMD+=(--pin_mem)
else
  CMD+=(--no_pin_mem)
fi

if [[ "${ADVERSARIAL_WEIGHTED_SAMPLING}" == "true" ]]; then
  CMD+=(--adversarial_weighted_sampling)
fi
if [[ -n "${ADVERSARIAL_COLUMN}" ]]; then
  CMD+=(--adversarial_column "${ADVERSARIAL_COLUMN}")
fi
if [[ -n "${ADVERSARIAL_OBSERVED_COLUMN}" ]]; then
  CMD+=(--adversarial_observed_column "${ADVERSARIAL_OBSERVED_COLUMN}")
fi
if [[ -n "${ADVERSARIAL_NUM_CLASSES}" ]]; then
  CMD+=(--adversarial_num_classes "${ADVERSARIAL_NUM_CLASSES}")
fi

echo "Running:"
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"
