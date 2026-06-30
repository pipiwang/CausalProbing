#!/usr/bin/env bash
set -euo pipefail

# Direct compute-node PROMIS external test for binary Grade Group >= 2,
# ruleout_diabetes, seeds 0-3. Assumes the Python environment is already active.
#
# Usage from /path/to/CausalProbing:
#   bash script/run_test_gleason_promis_external_binary_ge2_diabetes_seeds_0_3_node.sh
#
# Common overrides:
#   CHECKPOINTS=best_auc bash script/run_test_gleason_promis_external_binary_ge2_diabetes_seeds_0_3_node.sh
#   SEEDS="0 1" BATCH_SIZE=4 bash script/run_test_gleason_promis_external_binary_ge2_diabetes_seeds_0_3_node.sh

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

SEEDS=${SEEDS:-"0 1 2 3"}
BATCH_SIZE=${BATCH_SIZE:-8}
NUM_WORKERS=${NUM_WORKERS:-8}
CROP_SPATIAL_SIZE=${CROP_SPATIAL_SIZE:-64,256,256}
PIN_MEM=${PIN_MEM:-false}
DEVICE=${DEVICE:-cuda}
MODEL=${MODEL:-profound_conv}
TRAIN_MODE=${TRAIN_MODE:-fintune}
BINARY_POSITIVE_MIN=${BINARY_POSITIVE_MIN:-2}
ADVERSARIAL_VARIABLE=${ADVERSARIAL_VARIABLE:-diabetes}
MRI_ONLY_TEST=${MRI_ONLY_TEST:-true}
CHECKPOINTS=${CHECKPOINTS:-best_auc}
LR=${LR:-1e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.05}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-5}
LAYER_DECAY=${LAYER_DECAY:-0.6}
LAYER_DECAY_TYPE=${LAYER_DECAY_TYPE:-group}

read -r -a SEED_LIST <<< "${SEEDS}"
read -r -a CHECKPOINT_NAMES <<< "${CHECKPOINTS}"

if [[ "${ADVERSARIAL_VARIABLE}" == "none" ]]; then
  ADVERSARIAL_NAME=ruleout_none
else
  ADVERSARIAL_NAME=ruleout_${ADVERSARIAL_VARIABLE}
fi

mkdir -p "${RUN_LOG_DIR}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="${RUN_LOG_DIR}/promis_external_binary_ge2_${ADVERSARIAL_NAME}_seeds_0_3_${RUN_ID}.log"
exec > >(tee -a "${RUN_LOG}") 2>&1

cd "${CODE_DIR}"

if [[ ! -f test_gleason_classification.py ]]; then
  echo "Could not find test_gleason_classification.py in CODE_DIR=${CODE_DIR}" >&2
  exit 1
fi
if [[ ! -f "${PROMIS_CSV}" ]]; then
  echo "PROMIS CSV does not exist: ${PROMIS_CSV}" >&2
  exit 1
fi
if [[ ! -d "${PROMIS_CACHE_TEST_DIR}" ]]; then
  echo "Warning: PROMIS cache test directory does not exist: ${PROMIS_CACHE_TEST_DIR}" >&2
fi

echo "Project directory: ${PROJECT_DIR}"
echo "Code directory: ${CODE_DIR}"
echo "Run log: ${RUN_LOG}"
echo "PROMIS test CSV: ${PROMIS_CSV}"
echo "Expected PROMIS cache folder: ${PROMIS_CACHE_TEST_DIR}"
echo "Task type: binary"
echo "Binary label definition: grade_group >= ${BINARY_POSITIVE_MIN}"
echo "Adversarial variable: ${ADVERSARIAL_VARIABLE}"
echo "Seeds: ${SEEDS}"
echo "Checkpoints: ${CHECKPOINTS}"
echo "Host: $(hostname)"
nvidia-smi || true

BASE_ARGS=(
  --csv_path "${INTERNAL_CSV}"
  --test_csv "${PROMIS_CSV}"
  --split_col split
  --image_path_col image_npy_path
  --model "${MODEL}"
  --train "${TRAIN_MODE}"
  --pretrain "${PRETRAIN}"
  --task_type binary
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
  --output_dir "${OUTPUT_DIR}"
  --log_dir "${LOG_DIR}"
)

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
fi

for SEED in "${SEED_LIST[@]}"; do
  EXP_DIR=${OUTPUT_DIR}/gleason/binary/grade_group_ge_${BINARY_POSITIVE_MIN}/${ADVERSARIAL_NAME}/${MODEL}/${TRAIN_MODE}/${SEED}
  echo
  echo "=== PROMIS binary ge${BINARY_POSITIVE_MIN} ${ADVERSARIAL_NAME}, seed ${SEED} ==="
  echo "Experiment directory: ${EXP_DIR}"

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

    TEST_CMD=(
      python -u test_gleason_classification.py
      "${BASE_ARGS[@]}"
      "${ADVERSARIAL_ARGS[@]}"
      --seed "${SEED}"
      --checkpoint "${CKPT_PATH}"
      --metrics_output "${EXP_DIR}/promis_external_metrics_${METRICS_SUFFIX}.json"
      --predictions_output "${EXP_DIR}/promis_external_predictions_${METRICS_SUFFIX}.csv"
    )

    echo "Testing command:"
    printf ' %q' "${TEST_CMD[@]}"
    echo
    "${TEST_CMD[@]}"
  done
done
