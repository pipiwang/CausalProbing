#!/usr/bin/env bash
set -euo pipefail

# Direct compute-node runner for adversarial-only probe training.
#
# This loads a trained Gleason checkpoint, freezes the encoder and Gleason head,
# then trains only the adversarial probe head. No Gleason loss is used.
#
# Usage on an Isambard compute node from the code/project directory:
#   bash script/run_train_gleason_profound_conv_binary_adversarial_probe.sh
#
# Common overrides:
#   CHECKPOINT=/path/to/best.pth.tar bash script/run_train_gleason_profound_conv_binary_adversarial_probe.sh
#   ADVERSARIAL_VARIABLE=age EPOCHS=30 bash script/run_train_gleason_profound_conv_binary_adversarial_probe.sh
#   BATCH_SIZE=4 NUM_WORKERS=4 bash script/run_train_gleason_profound_conv_binary_adversarial_probe.sh
#
# Test an already trained probe checkpoint without further training:
#   EPOCHS=0 CHECKPOINT=/path/to/adversarial_probe/.../best.pth.tar LOAD_ADVERSARIAL_HEAD=true \
#     bash script/run_train_gleason_profound_conv_binary_adversarial_probe.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PROJECT_DIR=${PROJECT_DIR:-${DEFAULT_PROJECT_DIR}}
CONDA_ENV=${CONDA_ENV:-mae}
CONDA_SH=${CONDA_SH:-${HOME}/miniforge3/etc/profile.d/conda.sh}

CSV_PATH=${CSV_PATH:-data/gleason_classification.csv}
PRETRAIN=${PRETRAIN:-checkpoints/profound_conv_checkpoint-799.pth}

SEED=${SEED:-0}
BATCH_SIZE=${BATCH_SIZE:-8}
EPOCHS=${EPOCHS:-20}
NUM_WORKERS=${NUM_WORKERS:-8}
CROP_SPATIAL_SIZE=${CROP_SPATIAL_SIZE:-64,256,256}
PIN_MEM=${PIN_MEM:-false}
DEVICE=${DEVICE:-cuda}

ADVERSARIAL_VARIABLE=${ADVERSARIAL_VARIABLE:-psa_value}
ADVERSARIAL_COLUMN=${ADVERSARIAL_COLUMN:-}
ADVERSARIAL_OBSERVED_COLUMN=${ADVERSARIAL_OBSERVED_COLUMN:-}
ADVERSARIAL_NUM_CLASSES=${ADVERSARIAL_NUM_CLASSES:-}
ADVERSARIAL_LOSS_WEIGHT=${ADVERSARIAL_LOSS_WEIGHT:-1.0}
LOAD_ADVERSARIAL_HEAD=${LOAD_ADVERSARIAL_HEAD:-false}

LR=${LR:-1e-3}
MIN_LR=${MIN_LR:-0.0}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-0}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0}
PRIMARY_METRIC=${PRIMARY_METRIC:-balanced_acc}
SAVE_CKPT_INTERVAL=${SAVE_CKPT_INTERVAL:-0}

GLEASON_OUTPUT_DIR=${GLEASON_OUTPUT_DIR:-${PROJECT_DIR}/output_cls}
OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT_DIR}/output_cls_adversarial_probe}
LOG_DIR=${LOG_DIR:-${PROJECT_DIR}/output_cls_adversarial_probe}
RUN_LOG_DIR=${RUN_LOG_DIR:-${PROJECT_DIR}/logs/adversarial_probe}

DEFAULT_CHECKPOINT=${GLEASON_OUTPUT_DIR}/gleason/binary/grade_group_ge_2/ruleout_none/profound_conv/fintune/${SEED}/best.pth.tar
CHECKPOINT=${CHECKPOINT:-${DEFAULT_CHECKPOINT}}

mkdir -p "${RUN_LOG_DIR}"
cd "${PROJECT_DIR}"

if [[ ! -f "train_adversarial_probe.py" ]]; then
  echo "Could not find train_adversarial_probe.py in PROJECT_DIR=${PROJECT_DIR}" >&2
  echo "Set PROJECT_DIR to your code directory, for example:" >&2
  echo "  PROJECT_DIR=/path/to/CausalProbing bash ${BASH_SOURCE[0]}" >&2
  exit 1
fi

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT}" >&2
  echo "Set CHECKPOINT=/path/to/a/Gleason-or-probe-checkpoint.pth.tar" >&2
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
RUN_LOG="${RUN_LOG_DIR}/gleason_pconv_binary_adv_probe_${ADVERSARIAL_VARIABLE}_${RUN_ID}.log"
exec > >(tee -a "${RUN_LOG}") 2>&1

echo "Project directory: ${PROJECT_DIR}"
echo "Run log: ${RUN_LOG}"
echo "Source checkpoint: ${CHECKPOINT}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Log directory: ${LOG_DIR}"
echo "Seed: ${SEED}"
echo "Adversarial variable: ${ADVERSARIAL_VARIABLE}"
echo "Conda env: ${CONDA_ENV}"
echo "Host: $(hostname)"
nvidia-smi || true

CMD=(
  python -u train_adversarial_probe.py
  --checkpoint "${CHECKPOINT}"
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
  --lr "${LR}"
  --min_lr "${MIN_LR}"
  --warmup_epochs "${WARMUP_EPOCHS}"
  --weight_decay "${WEIGHT_DECAY}"
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

if [[ -n "${ADVERSARIAL_COLUMN}" ]]; then
  CMD+=(--adversarial_column "${ADVERSARIAL_COLUMN}")
fi
if [[ -n "${ADVERSARIAL_OBSERVED_COLUMN}" ]]; then
  CMD+=(--adversarial_observed_column "${ADVERSARIAL_OBSERVED_COLUMN}")
fi
if [[ -n "${ADVERSARIAL_NUM_CLASSES}" ]]; then
  CMD+=(--adversarial_num_classes "${ADVERSARIAL_NUM_CLASSES}")
fi
if [[ "${LOAD_ADVERSARIAL_HEAD}" == "true" ]]; then
  CMD+=(--load_adversarial_head)
fi

echo "Running:"
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"
