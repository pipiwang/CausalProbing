# CausalProbing

Implementation for the paper **"Causal Representation Probing for Prostate
Cancer Grade Prediction from MRI"**.

This repository contains code for prostate MRI Gleason grade prediction and
causal representation probing. The core workflow trains prostate MRI
classification models, removes or probes clinical-variable information with
adversarial heads, evaluates internal and PROMIS external performance, and
summarizes rule-out effects with bootstrap statistics.

## Repository Structure

| Path | Purpose |
| --- | --- |
| `clean_meta_info.py` | Prepare cleaned clinical metadata and derived adversarial variables. |
| `preprocess_gleason_images.py` | Convert raw T2/DWI/ADC scans into cached tensors and a Gleason manifest. |
| `preprocess_promis_external.py` | Prepare PROMIS scans as an external-test manifest. |
| `main_gleason_classification.py` | Train binary or ordinal Gleason classifiers with optional adversarial rule-out loss. |
| `test_gleason_classification.py` | Evaluate checkpoints and export metrics/predictions. |
| `train_adversarial_probe.py` | Train frozen-representation adversarial probes. |
| `train_adversarial_direct.py` | Train direct MRI-to-clinical-variable predictability controls. |
| `summarize_adversarial_candidates.py` | Summarize candidate clinical variables for adversarial training. |
| `analyze_gleason_ruleout_stats.py` | Compare baseline and rule-out results with bootstrap analyses. |
| `dataset/`, `engine/`, `models/`, `util/` | Data loading, training loops, model builders, metrics, and utilities. |
| `script/*.sh` | Optional direct-run helper wrappers for training, testing, PROMIS evaluation, and analysis. |

Segmentation code, Slurm submit files, local data, logs, figures, checkpoints,
and generated result tables are intentionally not included.

## Setup

Use an existing Python environment with a CUDA-compatible PyTorch installation:

```bash
pip install -r requirements.txt
```

The code was developed for GPU training. CPU execution is mainly useful for
lightweight inspection or debugging.

## Data And Checkpoints

Clinical data, MRI scans, cached tensors, prediction files, checkpoints, and
result tables are not included in this repository. Provide your own internal
manifest and image paths, then pass them with command-line arguments such as
`--csv_path`, `--raw-image-root`, `--cache-dir`, `--output_dir`, and `--log_dir`.

PROMIS prostate MRI data can be downloaded from
[Zenodo record 15683922](https://zenodo.org/records/15683922).

The ProFound encoder code and pretrained checkpoints can be found at
[pipiwang/ProFound](https://github.com/pipiwang/ProFound). By default, this
repository expects ProFound checkpoints under:

```text
checkpoints/profound_conv_checkpoint-799.pth
checkpoints/profound_vit_checkpoint-799.pth
```

You can also pass a checkpoint path explicitly with `--pretrain`.

## Minimal Workflow

Prepare an internal Gleason manifest:

```bash
python clean_meta_info.py \
  --input data/meta_info_processed.csv \
  --output data/meta_info_cleaned.csv

python preprocess_gleason_images.py \
  --input data/meta_info_cleaned.csv \
  --raw-image-root data/raw_mri \
  --cache-dir data/img \
  --qc-dir data/qc \
  --output data/gleason_classification.csv
```

Summarize candidate clinical variables:

```bash
python summarize_adversarial_candidates.py \
  --csv_path data/gleason_classification.csv
```

Train a baseline binary Gleason model:

```bash
python main_gleason_classification.py \
  --csv_path data/gleason_classification.csv \
  --model profound_conv \
  --train fintune \
  --pretrain checkpoints/profound_conv_checkpoint-799.pth \
  --task_type binary \
  --label_col grade_group \
  --binary_positive_min 2 \
  --adversarial_variable none \
  --weighted_sampling \
  --output_dir output_cls \
  --log_dir output_cls
```

Train a rule-out model by changing `--adversarial_variable`, for example:

```bash
python main_gleason_classification.py \
  --csv_path data/gleason_classification.csv \
  --model profound_conv \
  --train fintune \
  --pretrain checkpoints/profound_conv_checkpoint-799.pth \
  --task_type binary \
  --label_col grade_group \
  --binary_positive_min 2 \
  --adversarial_variable psa_value \
  --adversarial_loss_weight 1.0 \
  --grl_lambda 1.0 \
  --grl_schedule dann \
  --weighted_sampling \
  --output_dir output_cls \
  --log_dir output_cls
```

Evaluate a trained checkpoint:

```bash
python test_gleason_classification.py \
  --csv_path data/gleason_classification.csv \
  --model profound_conv \
  --train fintune \
  --task_type binary \
  --label_col grade_group \
  --binary_positive_min 2 \
  --checkpoint output_cls/gleason/binary/grade_group_ge_2/ruleout_none/profound_conv/fintune/0/best.pth.tar \
  --output_dir output_cls \
  --log_dir output_cls
```

Compare baseline and rule-out results across matched seeds:

```bash
python analyze_gleason_ruleout_stats.py \
  --analysis aggregate \
  --root output_cls/gleason/binary/grade_group_ge_2 \
  --baseline ruleout_none \
  --selection best_auc \
  --metrics test_auc test_balanced_acc test_sens_at_80_spec \
  --seed_bootstrap_iterations 10000 \
  --csv_output results/ruleout_stats_best_auc.csv
```

The helper scripts in `script/` wrap common direct-run workflows. They are
optional and can be customized with environment variables such as `CSV_PATH`,
`PRETRAIN`, `OUTPUT_DIR`, `SEED`, `TARGETS`, and `SEED_BOOTSTRAP_ITERATIONS`.

## External PROMIS Evaluation

After preparing a PROMIS manifest with `preprocess_promis_external.py`, evaluate
trained checkpoints with:

```bash
bash script/run_test_gleason_promis_external.sh
```

Then summarize matched-seed PROMIS rule-out results with:

```bash
bash script/run_analyze_gleason_promis_external_seed_paired.sh
```

## Notes

Any local or cluster-specific paths in examples are placeholders. Edit them or
override them through command-line arguments and environment variables before
running on your system.
