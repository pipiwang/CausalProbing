import argparse
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from dataset.gleason_cls import (
    GleasonClassificationDataset,
    configure_gleason_task_args,
    get_gleason_transforms,
    parse_int_tuple,
)
from util.gleason_config import ADVERSARIAL_VARIABLE_CHOICES
from util.paths import DATA_ROOT, GLEASON_CLASSIFICATION_CSV


def get_args_parser():
    parser = argparse.ArgumentParser("Check Gleason classification dataset")
    parser.add_argument("--data_root", default=str(DATA_ROOT), type=str)
    parser.add_argument("--csv_path", default=str(GLEASON_CLASSIFICATION_CSV), type=str)
    parser.add_argument("--train_csv", default=None, type=str)
    parser.add_argument("--val_csv", default=None, type=str)
    parser.add_argument("--test_csv", default=None, type=str)
    parser.add_argument("--split_col", default="split", type=str)
    parser.add_argument("--patient_col", default="person_id", type=str)
    parser.add_argument("--scan_id_col", default="new_id", type=str)
    parser.add_argument("--image_path_col", default="image_npy_path", type=str)
    parser.add_argument("--in_channels", default=3, type=int)
    parser.add_argument("--crop_spatial_size", default=(64, 256, 256), type=parse_int_tuple)

    parser.add_argument("--task_type", choices=["binary", "ordinal"], required=True)
    parser.add_argument("--label_col", default="grade_group", type=str)
    parser.add_argument("--binary_label_col", default=None, type=str)
    parser.add_argument("--binary_positive_min", default=2, type=int)
    parser.add_argument("--ordinal_levels", default=5, type=int)
    parser.add_argument("--label_offset", default=1, type=int)

    parser.add_argument("--adversarial_specs", default="", type=str)
    parser.add_argument(
        "--adversarial_variable",
        default=None,
        choices=ADVERSARIAL_VARIABLE_CHOICES,
        type=str,
    )
    parser.add_argument("--adversarial_column", default=None, type=str)
    parser.add_argument("--adversarial_observed_column", default=None, type=str)
    parser.add_argument("--adversarial_num_classes", default=None, type=int)
    parser.add_argument("--adversarial_loss_weight", default=1.0, type=float)
    parser.add_argument("--grl_lambda", default=1.0, type=float)
    parser.add_argument(
        "--grl_schedule",
        choices=["constant", "dann"],
        default="constant",
        type=str,
    )
    parser.add_argument("--grl_gamma", default=10.0, type=float)

    parser.add_argument("--max_samples_per_split", default=16, type=int)
    parser.add_argument("--check_train_augmentations", action="store_true")
    return parser


def csv_for_phase(args, phase: str):
    phase_csv = getattr(args, f"{phase}_csv")
    return phase_csv if phase_csv else args.csv_path


def print_manifest_summary(args):
    df = pd.read_csv(args.csv_path)
    print(f"Manifest: {args.csv_path}")
    print(f"Rows: {len(df):,}")

    if args.split_col in df.columns:
        print("Scan counts by split:")
        print(df[args.split_col].value_counts(dropna=False).sort_index().to_string())

    if args.split_col in df.columns and args.patient_col in df.columns:
        print("Patient counts by split:")
        print(
            df.groupby(args.split_col)[args.patient_col]
            .nunique(dropna=True)
            .sort_index()
            .to_string()
        )

    for column in [args.image_path_col, args.label_col, "cache_shape", "cache_spacing"]:
        if column in df.columns:
            missing = int(df[column].isna().sum())
            print(f"Missing {column}: {missing:,}")

    if "all_zero_scan" in df.columns:
        print(f"Manifest all_zero_scan rows: {int(df['all_zero_scan'].fillna(0).sum()):,}")
    if "all_zero_modalities" in df.columns:
        nonempty = df["all_zero_modalities"].fillna("").astype(str).str.len() > 0
        print(f"Manifest all_zero_modalities rows: {int(nonempty.sum()):,}")


def check_sample(dataset, idx: int):
    sample = dataset[idx]
    if len(sample) == 3:
        image, label, sample_idx = sample
        adv_targets = {}
        adv_masks = {}
    elif len(sample) == 5:
        image, label, adv_targets, adv_masks, sample_idx = sample
    else:
        raise ValueError(f"Unexpected sample length: {len(sample)}")

    if tuple(image.shape) != tuple((dataset.args.in_channels, *dataset.args.crop_spatial_size)):
        raise ValueError(
            f"sample {idx} transformed shape {tuple(image.shape)} does not match "
            f"expected {(dataset.args.in_channels, *dataset.args.crop_spatial_size)}"
        )
    if not torch.isfinite(image).all():
        raise ValueError(f"sample {idx} contains NaN/Inf after transforms")
    if not torch.any(image != 0):
        raise ValueError(f"sample {idx} is all zero after transforms")

    return {
        "idx": int(sample_idx),
        "label": int(label),
        "shape": tuple(image.shape),
        "min": float(image.min()),
        "max": float(image.max()),
        "mean": float(image.mean()),
        "std": float(image.std()),
        "adv_targets": {name: int(value) for name, value in adv_targets.items()},
        "adv_masks": {name: bool(value) for name, value in adv_masks.items()},
    }


def check_split(args, phase: str, transforms):
    dataset = GleasonClassificationDataset(args, csv_for_phase(args, phase), phase, transforms)
    print(f"\n[{phase}] samples: {len(dataset):,}")
    print(f"[{phase}] label counts: {dataset.label_counts()}")
    if dataset.adversarial_specs:
        print(
            f"[{phase}] adversarial usable counts: "
            f"{dataset.adversarial_observed_counts()}"
        )

    n_check = min(args.max_samples_per_split, len(dataset))
    if n_check <= 0:
        raise ValueError(f"No samples to check for split {phase}")
    indices = np.linspace(0, len(dataset) - 1, num=n_check, dtype=int)

    summaries = []
    for idx in indices:
        summaries.append(check_sample(dataset, int(idx)))

    shapes = sorted({str(item["shape"]) for item in summaries})
    labels = pd.Series([item["label"] for item in summaries]).value_counts().sort_index()
    print(f"[{phase}] checked samples: {n_check}")
    print(f"[{phase}] checked transformed shapes: {shapes}")
    print(f"[{phase}] checked label counts: {labels.to_dict()}")
    print(
        f"[{phase}] checked intensity range: "
        f"min={min(item['min'] for item in summaries):.4g}, "
        f"max={max(item['max'] for item in summaries):.4g}"
    )


def main(args):
    configure_gleason_task_args(args)
    print_manifest_summary(args)
    print(f"Label definition: {args.label_definition}")
    print(f"Adversarial definition: {args.adversarial_definition}")
    print(f"Target transformed shape [C,Z,H,W]: {(args.in_channels, *args.crop_spatial_size)}")

    train_transforms, val_transforms, test_transforms = get_gleason_transforms(args)
    if not args.check_train_augmentations:
        train_transforms = val_transforms

    check_split(args, "train", train_transforms)
    check_split(args, "val", val_transforms)
    check_split(args, "test", test_transforms)
    print("\nDataset check passed.")


if __name__ == "__main__":
    parsed = get_args_parser().parse_args()
    main(SimpleNamespace(**vars(parsed)))
