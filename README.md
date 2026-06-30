# CausalProbing

Code for causal probing and adversarial rule-out experiments for prostate MRI
Gleason grade classification. The workflow includes metadata preparation, MRI
preprocessing, internal training/evaluation, PROMIS external evaluation, and
rule-out result analysis.

## Repository Structure

| Path | Purpose |
| --- | --- |
| `clean_meta_info.py` | Prepare cleaned metadata and derived clinical variables. |
| `preprocess_gleason_images.py` | Build cached MRI tensors and a Gleason training manifest from raw T2/DWI/ADC scans. |
| `preprocess_promis_external.py` | Prepare PROMIS scans as an external-test manifest. |
| `main_gleason_classification.py` | Train binary or ordinal Gleason classifiers, with optional adversarial rule-out loss. |
| `test_gleason_classification.py` | Evaluate checkpoints and write metrics/predictions. |
| `train_adversarial_probe.py` | Train a frozen-representation adversarial probe. |
| `train_adversarial_direct.py` | Train direct MRI-to-clinical-variable predictability controls. |
| `analyze_gleason_ruleout_stats.py` | Compare baseline and rule-out metrics across seeds or paired predictions. |
| `plot_gleason_ruleout_results.py` | Render rule-out summary figures from analysis tables. |
| `dataset/`, `engine/`, `models/`, `util/` | Data loading, training loops, model builders, metrics, and helpers. |
| `script/*.sh` | Optional direct-run helper wrappers. |

## Setup

With an existing Python environment:

```bash
pip install -r requirements.txt
```

GPU training expects a CUDA-capable PyTorch installation compatible with your
system.

## Data

Clinical data, MRI scans, cached tensors, prediction files, checkpoints, and
result tables are not included in this repository. Provide your own manifest and
image paths, then pass them with command-line arguments such as `--csv_path`,
`--raw-image-root`, `--cache-dir`, `--output_dir`, and `--log_dir`.

PROMIS prostate MRI data can be downloaded from
[Zenodo record 15683922](https://zenodo.org/records/15683922).

## Minimal Workflow

Prepare an internal Gleason manifest:

```bash
python clean_meta_info.py --input data/meta_info_processed.csv --output data/meta_info_cleaned.csv
python preprocess_gleason_images.py \
  --input data/meta_info_cleaned.csv \
  --raw-image-root data/raw_mri \
  --cache-dir data/img \
  --qc-dir data/qc \
  --output data/gleason_classification.csv
```

Check the dataset and model wiring:

```bash
python check_gleason_dataset.py --csv_path data/gleason_classification.csv
python check_gleason_model.py --csv_path data/gleason_classification.csv
```

Train and evaluate a baseline binary Gleason model:

```bash
python main_gleason_classification.py \
  --csv_path data/gleason_classification.csv \
  --task_type binary \
  --label_col grade_group \
  --binary_positive_min 2 \
  --adversarial_variable none \
  --output_dir output_cls \
  --log_dir output_cls

python test_gleason_classification.py \
  --csv_path data/gleason_classification.csv \
  --checkpoint output_cls/gleason/binary/grade_group_ge_2/ruleout_none/profound_conv/fintune/0/best.pth.tar \
  --output_dir output_cls
```

Run a rule-out experiment by setting `--adversarial_variable`, then compare
matched seeds:

```bash
python analyze_gleason_ruleout_stats.py \
  --analysis aggregate \
  --root output_cls/gleason/binary/grade_group_ge_2 \
  --baseline ruleout_none \
  --selection best_auc \
  --metrics test_auc test_balanced_acc test_sens_at_80_spec \
  --csv_output results/ruleout_stats_best_auc.csv
```

## Paths

Any local or cluster-specific paths in examples are placeholders. Edit them or
override them through command-line arguments and environment variables before
running on your system.
