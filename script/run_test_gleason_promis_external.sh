#!/usr/bin/env bash
set -euo pipefail

# Test-only PROMIS external runner for Gleason checkpoints.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/../test_gleason_classification.py" ]]; then
  DEFAULT_PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
elif [[ -f "${SCRIPT_DIR}/../../code/test_gleason_classification.py" ]]; then
  DEFAULT_PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
else
  DEFAULT_PROJECT_DIR=/path/to/CausalProbing
fi

PROJECT_DIR=${PROJECT_DIR:-${DEFAULT_PROJECT_DIR}}
if [[ -d "${PROJECT_DIR}/code" && -f "${PROJECT_DIR}/code/test_gleason_classification.py" ]]; then
  CODE_DIR=${CODE_DIR:-${PROJECT_DIR}/code}
else
  CODE_DIR=${CODE_DIR:-${PROJECT_DIR}}
fi

INTERNAL_CSV=${INTERNAL_CSV:-data/gleason_classification.csv}
PROMIS_CSV=${PROMIS_CSV:-data/promis_external_gleason.csv}
PROMIS_CACHE_TEST_DIR=${PROMIS_CACHE_TEST_DIR:-data/promis_external_gleason/img/test}
PRETRAIN=${PRETRAIN:-checkpoints/profound_conv_checkpoint-799.pth}
OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT_DIR}/output_cls}
LOG_DIR=${LOG_DIR:-${PROJECT_DIR}/output_cls}
RUN_LOG_DIR=${RUN_LOG_DIR:-${PROJECT_DIR}/logs}
CONDA_ENV=${CONDA_ENV:-mae}

SEEDS=${SEEDS:-"0 1 2 3"}
read -r -a SEED_LIST <<< "${SEEDS}"
TASK_INDEX=${TASK_INDEX:-0}
if (( TASK_INDEX < 0 || TASK_INDEX >= ${#SEED_LIST[@]} )); then
  echo "TASK_INDEX=${TASK_INDEX} is outside SEEDS list length ${#SEED_LIST[@]}" >&2
  echo "SEEDS=${SEEDS}" >&2
  exit 1
fi
SEED=${SEED:-${SEED_LIST[${TASK_INDEX}]}}

BATCH_SIZE=${BATCH_SIZE:-8}
NUM_WORKERS=${NUM_WORKERS:-8}
CROP_SPATIAL_SIZE=${CROP_SPATIAL_SIZE:-64,256,256}
PIN_MEM=${PIN_MEM:-false}
DEVICE=${DEVICE:-cuda}
MODEL=${MODEL:-profound_conv}
TRAIN_MODE=${TRAIN_MODE:-fintune}
TASK_TYPE=${TASK_TYPE:-ordinal}
ORDINAL_LEVELS=${ORDINAL_LEVELS:-5}
LABEL_OFFSET=${LABEL_OFFSET:-1}
BINARY_POSITIVE_MIN=${BINARY_POSITIVE_MIN:-2}
MRI_ONLY_TEST=${MRI_ONLY_TEST:-true}
if [[ -z "${LR:-}" ]]; then
  if [[ "${MODEL}" == "resnet18" ]]; then
    LR=1e-3
  else
    LR=1e-4
  fi
fi
WEIGHT_DECAY=${WEIGHT_DECAY:-0.05}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-5}
LAYER_DECAY=${LAYER_DECAY:-0.6}
LAYER_DECAY_TYPE=${LAYER_DECAY_TYPE:-group}

ADVERSARIAL_VARIABLE=${ADVERSARIAL_VARIABLE:-none}
ADVERSARIAL_COLUMN=${ADVERSARIAL_COLUMN:-}
ADVERSARIAL_OBSERVED_COLUMN=${ADVERSARIAL_OBSERVED_COLUMN:-}
ADVERSARIAL_NUM_CLASSES=${ADVERSARIAL_NUM_CLASSES:-}

if [[ -z "${CHECKPOINTS:-}" ]]; then
  if [[ "${TASK_TYPE}" == "ordinal" ]]; then
    CHECKPOINTS="best best_qwk best_auc best_balanced_acc"
  else
    CHECKPOINTS="best best_auc best_balanced_acc"
  fi
fi
read -r -a CHECKPOINT_NAMES <<< "${CHECKPOINTS}"

if [[ "${ADVERSARIAL_VARIABLE}" == "none" ]]; then
  ADVERSARIAL_NAME=ruleout_none
else
  ADVERSARIAL_NAME=ruleout_${ADVERSARIAL_VARIABLE}
fi

if [[ "${TASK_TYPE}" == "ordinal" ]]; then
  EXP_DIR=${OUTPUT_DIR}/gleason/ordinal/grade_group_ordinal_${ORDINAL_LEVELS}levels/${ADVERSARIAL_NAME}/${MODEL}/${TRAIN_MODE}/${SEED}
elif [[ "${TASK_TYPE}" == "binary" ]]; then
  EXP_DIR=${OUTPUT_DIR}/gleason/binary/grade_group_ge_${BINARY_POSITIVE_MIN}/${ADVERSARIAL_NAME}/${MODEL}/${TRAIN_MODE}/${SEED}
else
  echo "Unsupported TASK_TYPE=${TASK_TYPE}; expected ordinal or binary" >&2
  exit 1
fi

mkdir -p "${RUN_LOG_DIR}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="${RUN_LOG_DIR}/promis_external_${TASK_TYPE}_seed${SEED}_${RUN_ID}.log"
exec > >(tee -a "${RUN_LOG}") 2>&1

cd "${CODE_DIR}"

if [[ ! -f test_gleason_classification.py ]]; then
  echo "Could not find test_gleason_classification.py in CODE_DIR=${CODE_DIR}" >&2
  exit 1
fi

source "${HOME}/miniforge3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

echo "Project directory: ${PROJECT_DIR}"
echo "Code directory: ${CODE_DIR}"
echo "Conda environment: ${CONDA_ENV}"
echo "Run log: ${RUN_LOG}"
echo "Internal train/val CSV: ${INTERNAL_CSV}"
echo "PROMIS test CSV: ${PROMIS_CSV}"
echo "Expected PROMIS cache folder: ${PROMIS_CACHE_TEST_DIR}"
echo "Seeds: ${SEEDS}"
echo "Task index: ${TASK_INDEX}"
echo "Seed: ${SEED}"
echo "Task type: ${TASK_TYPE}"
echo "Adversarial variable: ${ADVERSARIAL_VARIABLE}"
echo "Experiment directory: ${EXP_DIR}"
echo "Checkpoints: ${CHECKPOINT_NAMES[*]}"
echo "Host: $(hostname)"
nvidia-smi || true

if [[ ! -f "${PROMIS_CSV}" ]]; then
  echo "PROMIS CSV does not exist: ${PROMIS_CSV}" >&2
  exit 1
fi
if [[ ! -d "${PROMIS_CACHE_TEST_DIR}" ]]; then
  echo "Warning: PROMIS cache test directory does not exist: ${PROMIS_CACHE_TEST_DIR}" >&2
fi

BASE_ARGS=(
  --csv_path "${INTERNAL_CSV}"
  --test_csv "${PROMIS_CSV}"
  --split_col split
  --image_path_col image_npy_path
  --model "${MODEL}"
  --train "${TRAIN_MODE}"
  --pretrain "${PRETRAIN}"
  --task_type "${TASK_TYPE}"
  --label_col grade_group
  --binary_positive_min "${BINARY_POSITIVE_MIN}"
  --crop_spatial_size "${CROP_SPATIAL_SIZE}"
  --batch_size "${BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --device "${DEVICE}"
  --lr "${LR}"
  --weight_decay "${WEIGHT_DECAY}"
  --warmup_epochs "${WARMUP_EPOCHS}"
  --layer_decay "${LAYER_DECAY}"
  --layer_decay_type "${LAYER_DECAY_TYPE}"
  --seed "${SEED}"
  --output_dir "${OUTPUT_DIR}"
  --log_dir "${LOG_DIR}"
)

if [[ "${TASK_TYPE}" == "ordinal" ]]; then
  BASE_ARGS+=(
    --ordinal_levels "${ORDINAL_LEVELS}"
    --label_offset "${LABEL_OFFSET}"
  )
fi

if [[ "${PIN_MEM}" == "true" ]]; then
  BASE_ARGS+=(--pin_mem)
else
  BASE_ARGS+=(--no_pin_mem)
fi

ADVERSARIAL_ARGS=()
if [[ "${ADVERSARIAL_VARIABLE}" != "none" ]]; then
  ADVERSARIAL_ARGS+=(--adversarial_variable "${ADVERSARIAL_VARIABLE}")
  if [[ "${MRI_ONLY_TEST}" == "true" ]]; then
    ADVERSARIAL_ARGS+=(--drop_adversarial_head)
  fi
  if [[ -n "${ADVERSARIAL_COLUMN}" ]]; then
    ADVERSARIAL_ARGS+=(--adversarial_column "${ADVERSARIAL_COLUMN}")
  fi
  if [[ -n "${ADVERSARIAL_OBSERVED_COLUMN}" ]]; then
    ADVERSARIAL_ARGS+=(--adversarial_observed_column "${ADVERSARIAL_OBSERVED_COLUMN}")
  fi
  if [[ -n "${ADVERSARIAL_NUM_CLASSES}" ]]; then
    ADVERSARIAL_ARGS+=(--adversarial_num_classes "${ADVERSARIAL_NUM_CLASSES}")
  fi
fi

for CKPT_NAME in "${CHECKPOINT_NAMES[@]}"; do
  CKPT_PATH=${EXP_DIR}/${CKPT_NAME}.pth.tar
  if [[ ! -f "${CKPT_PATH}" ]]; then
    echo "Skipping missing checkpoint: ${CKPT_PATH}"
    continue
  fi

  METRICS_SUFFIX=${CKPT_NAME}
  if [[ "${ADVERSARIAL_VARIABLE}" != "none" && "${MRI_ONLY_TEST}" == "true" ]]; then
    METRICS_SUFFIX=${CKPT_NAME}_mri_only
  fi

  METRICS_OUTPUT=${EXP_DIR}/promis_external_metrics_${METRICS_SUFFIX}.json
  PREDICTIONS_OUTPUT=${EXP_DIR}/promis_external_predictions_${METRICS_SUFFIX}.csv
  TEST_CMD=(
    python -u test_gleason_classification.py
    "${BASE_ARGS[@]}"
    "${ADVERSARIAL_ARGS[@]}"
    --checkpoint "${CKPT_PATH}"
    --metrics_output "${METRICS_OUTPUT}"
    --predictions_output "${PREDICTIONS_OUTPUT}"
  )

  echo "Testing command:"
  printf ' %q' "${TEST_CMD[@]}"
  echo
  "${TEST_CMD[@]}"
  echo "Wrote PROMIS metrics: ${METRICS_OUTPUT}"
  echo "Wrote PROMIS predictions: ${PREDICTIONS_OUTPUT}"
done
