"""Patient-wise split and image preprocessing for Gleason classification.

This script:
- builds a patient-wise 7:1:2 train/val/test split stratified by max grade_group
- constructs raw image paths from new_id
- aligns DWI/ADC to T2 physical space
- resamples all modalities to foundation-model spacing, default 0.5 x 0.5 x 1.0 mm
- saves one .npy cache per scan with shape [3, Z, H, W]
- writes a final manifest CSV, failure report CSV, split summary CSV, and QC PNGs
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from util.paths import (
    GLEASON_CLASSIFICATION_CSV,
    GLEASON_PREPROCESS_FAILURES_CSV,
    GLEASON_SPLIT_SUMMARY_CSV,
    IMAGE_CACHE_DIR,
    META_INFO_CLEANED_CSV,
    QC_DIR,
    RAW_IMAGE_ROOT,
)


MODALITIES = ("t2", "dwi", "adc")
TARGET_SPACING = (0.5, 0.5, 1.0)


def require_simpleitk():
    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise SystemExit(
            "SimpleITK is required for preprocessing. Install it in the HPC environment."
        ) from exc
    return sitk


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess Gleason MRI scans into .npy cache files."
    )
    parser.add_argument("--input", default=str(META_INFO_CLEANED_CSV), type=str)
    parser.add_argument("--raw-image-root", default=str(RAW_IMAGE_ROOT), type=str)
    parser.add_argument("--cache-dir", default=str(IMAGE_CACHE_DIR), type=str)
    parser.add_argument("--qc-dir", default=str(QC_DIR), type=str)
    parser.add_argument("--output", default=str(GLEASON_CLASSIFICATION_CSV), type=str)
    parser.add_argument(
        "--failure-output",
        default=str(GLEASON_PREPROCESS_FAILURES_CSV),
        type=str,
    )
    parser.add_argument(
        "--split-summary-output",
        default=str(GLEASON_SPLIT_SUMMARY_CSV),
        type=str,
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--train-ratio", default=0.7, type=float)
    parser.add_argument("--val-ratio", default=0.1, type=float)
    parser.add_argument("--test-ratio", default=0.2, type=float)
    parser.add_argument("--label-col", default="grade_group", type=str)
    parser.add_argument("--patient-col", default="person_id", type=str)
    parser.add_argument("--scan-id-col", default="new_id", type=str)
    parser.add_argument(
        "--target-spacing",
        default=TARGET_SPACING,
        nargs=3,
        type=float,
        metavar=("SX", "SY", "SZ"),
        help="Output voxel spacing in SimpleITK x y z order. Default: 0.5 0.5 1.0.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-qc", action="store_true")
    parser.add_argument(
        "--log-every",
        default=25,
        type=int,
        help="Print preprocessing progress every N scans. Use 0 to disable.",
    )
    parser.add_argument(
        "--allow-nonstratified-fallback",
        action="store_true",
        help="Use patient-wise random split if stratified split is impossible.",
    )
    return parser.parse_args()


def image_paths(raw_image_root: Path, scan_id: str) -> dict[str, Path]:
    scan_dir = raw_image_root / scan_id
    return {modality: scan_dir / f"{modality}.nii.gz" for modality in MODALITIES}


def format_tuple(values: Iterable[object]) -> str:
    return ";".join(str(value) for value in values)


def image_size(image) -> str:
    return format_tuple(int(v) for v in image.GetSize())


def image_spacing(image) -> str:
    return format_tuple(f"{float(v):.8g}" for v in image.GetSpacing())


def resample_to_reference(moving, reference):
    sitk = require_simpleitk()
    return sitk.Resample(
        moving,
        reference,
        sitk.Transform(reference.GetDimension(), sitk.sitkIdentity),
        sitk.sitkLinear,
        0.0,
        moving.GetPixelID(),
    )


def resample_to_spacing(image, out_spacing=TARGET_SPACING):
    sitk = require_simpleitk()
    in_spacing = np.asarray(image.GetSpacing(), dtype=float)
    in_size = np.asarray(image.GetSize(), dtype=float)
    out_spacing = np.asarray(out_spacing, dtype=float)
    out_size = np.maximum(np.round(in_size * in_spacing / out_spacing), 1).astype(int)

    return sitk.Resample(
        image,
        [int(v) for v in out_size.tolist()],
        sitk.Transform(image.GetDimension(), sitk.sitkIdentity),
        sitk.sitkLinear,
        image.GetOrigin(),
        tuple(float(v) for v in out_spacing.tolist()),
        image.GetDirection(),
        0.0,
        image.GetPixelID(),
    )


def zero_content_report(image: np.ndarray) -> tuple[int, str]:
    finite = np.isfinite(image)
    nonzero = finite & (image != 0)
    all_zero_scan = int(not nonzero.any())
    all_zero_modalities = [
        modality
        for modality, modality_image in zip(MODALITIES, image)
        if not (np.isfinite(modality_image) & (modality_image != 0)).any()
    ]
    return all_zero_scan, ";".join(all_zero_modalities)


def preprocess_scan(
    row,
    raw_image_root: Path,
    cache_dir: Path,
    overwrite: bool,
    scan_id_col: str,
    target_spacing: tuple[float, float, float],
):
    sitk = require_simpleitk()
    scan_id = str(row[scan_id_col])
    paths = image_paths(raw_image_root, scan_id)

    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing image file(s): " + ", ".join(missing))

    t2 = sitk.ReadImage(str(paths["t2"]), sitk.sitkFloat32)
    dwi = sitk.ReadImage(str(paths["dwi"]), sitk.sitkFloat32)
    adc = sitk.ReadImage(str(paths["adc"]), sitk.sitkFloat32)

    original_sizes = {
        "t2_original_size": image_size(t2),
        "dwi_original_size": image_size(dwi),
        "adc_original_size": image_size(adc),
        "t2_original_spacing": image_spacing(t2),
        "dwi_original_spacing": image_spacing(dwi),
        "adc_original_spacing": image_spacing(adc),
    }

    dwi_on_t2 = resample_to_reference(dwi, t2)
    adc_on_t2 = resample_to_reference(adc, t2)

    resampled = [
        resample_to_spacing(t2, target_spacing),
        resample_to_spacing(dwi_on_t2, target_spacing),
        resample_to_spacing(adc_on_t2, target_spacing),
    ]

    arrays = []
    for image in resampled:
        array = sitk.GetArrayFromImage(image).astype(np.float32)
        arrays.append(array)
    shapes = [array.shape for array in arrays]
    if len(set(shapes)) != 1:
        raise ValueError(f"resampled modality shapes do not match: {shapes}")

    split = str(row["split"])
    image = np.stack(arrays, axis=0).astype(np.float32)
    all_zero_scan, all_zero_modalities = zero_content_report(image)
    if all_zero_scan:
        raise ValueError("all-zero scan after preprocessing")
    if all_zero_modalities:
        raise ValueError(
            f"all-zero modality after preprocessing: {all_zero_modalities}"
        )

    cache_path = cache_dir / split / f"{scan_id}.npy"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite or not cache_path.exists():
        np.save(cache_path, image)

    z_size = int(image.shape[1])
    info = {
        **original_sizes,
        "image_npy_path": str(cache_path.resolve()),
        "t2_source_path": str(paths["t2"].resolve()),
        "dwi_source_path": str(paths["dwi"].resolve()),
        "adc_source_path": str(paths["adc"].resolve()),
        "cache_shape": format_tuple(image.shape),
        "cache_channels": int(image.shape[0]),
        "cache_z": z_size,
        "cache_height": int(image.shape[2]),
        "cache_width": int(image.shape[3]),
        "cache_spacing": format_tuple(target_spacing),
        "all_zero_scan": all_zero_scan,
        "all_zero_modalities": all_zero_modalities,
        "qc_slice_1": max(z_size // 3, 0),
        "qc_slice_2": min((2 * z_size) // 3, z_size - 1),
    }
    return image, info


def robust_slice_view(array: np.ndarray) -> np.ndarray:
    finite = np.isfinite(array)
    if not finite.any():
        return np.zeros_like(array, dtype=np.float32)
    values = array[finite]
    vmin, vmax = np.percentile(values, [1, 99])
    if np.isclose(vmin, vmax):
        return np.zeros_like(array, dtype=np.float32)
    return np.clip((array - vmin) / (vmax - vmin), 0, 1)


def write_qc_png(image: np.ndarray, qc_path: Path, scan_id: str, grade_group, split: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _, z_size, _, _ = image.shape
    slices = [max(z_size // 3, 0), min((2 * z_size) // 3, z_size - 1)]
    names = ["T2", "DWI", "ADC"]

    fig, axes = plt.subplots(2, 3, figsize=(9, 6), constrained_layout=True)
    for row_idx, z_idx in enumerate(slices):
        for channel_idx, name in enumerate(names):
            axes[row_idx, channel_idx].imshow(
                robust_slice_view(image[channel_idx, z_idx]),
                cmap="gray",
            )
            axes[row_idx, channel_idx].set_title(f"{name} z={z_idx}")
            axes[row_idx, channel_idx].axis("off")

    fig.suptitle(
        f"{scan_id} | split={split} | grade_group={grade_group} | shape={tuple(image.shape)}"
    )
    qc_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(qc_path, dpi=120)
    plt.close(fig)


def patient_stratification_table(df: pd.DataFrame, patient_col: str, label_col: str):
    patients = (
        df.groupby(patient_col, dropna=False)[label_col]
        .max()
        .reset_index()
        .rename(columns={label_col: "patient_stratify_grade_group"})
    )
    patients = patients.dropna(subset=["patient_stratify_grade_group"]).reset_index(drop=True)
    patients["patient_stratify_grade_group"] = (
        patients["patient_stratify_grade_group"].astype(int)
    )
    return patients


def safe_train_test_split(
    patients: pd.DataFrame,
    test_size: float,
    seed: int,
    allow_nonstratified_fallback: bool,
):
    stratify = patients["patient_stratify_grade_group"]
    try:
        return train_test_split(
            patients,
            test_size=test_size,
            random_state=seed,
            stratify=stratify,
        )
    except ValueError:
        if not allow_nonstratified_fallback:
            raise
        return train_test_split(patients, test_size=test_size, random_state=seed)


def make_patient_split(
    df: pd.DataFrame,
    patient_col: str,
    label_col: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    allow_nonstratified_fallback: bool,
) -> pd.DataFrame:
    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")

    patients = patient_stratification_table(df, patient_col, label_col)
    train_val, test = safe_train_test_split(
        patients,
        test_size=test_ratio,
        seed=seed,
        allow_nonstratified_fallback=allow_nonstratified_fallback,
    )
    val_fraction_of_train_val = val_ratio / (train_ratio + val_ratio)
    train, val = safe_train_test_split(
        train_val,
        test_size=val_fraction_of_train_val,
        seed=seed,
        allow_nonstratified_fallback=allow_nonstratified_fallback,
    )

    split_rows = []
    for split_name, split_df in [
        ("train", train),
        ("val", val),
        ("test", test),
    ]:
        split_rows.append(
            split_df[[patient_col, "patient_stratify_grade_group"]].assign(
                split=split_name
            )
        )
    return pd.concat(split_rows, axis=0, ignore_index=True)


def add_split(df: pd.DataFrame, patient_split: pd.DataFrame, patient_col: str):
    return df.merge(patient_split, on=patient_col, how="inner")


def split_summary(manifest: pd.DataFrame, patient_col: str, label_col: str) -> pd.DataFrame:
    rows = []
    for split_name, split_df in manifest.groupby("split", sort=False):
        rows.append(
            {
                "split": split_name,
                "patients": split_df[patient_col].nunique(dropna=True),
                "scans": len(split_df),
                "scan_grade_group_counts": split_df[label_col]
                .value_counts(dropna=False)
                .sort_index()
                .to_json(),
                "patient_stratify_grade_group_counts": split_df[
                    [patient_col, "patient_stratify_grade_group"]
                ]
                .drop_duplicates()["patient_stratify_grade_group"]
                .value_counts(dropna=False)
                .sort_index()
                .to_json(),
            }
        )
    return pd.DataFrame(rows)


def write_failures(path: Path, failures: list[dict[str, object]]) -> None:
    fieldnames = ["new_id", "person_id", "reason"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(failures)


def main():
    args = parse_args()
    input_path = Path(args.input)
    raw_image_root = Path(args.raw_image_root)
    cache_dir = Path(args.cache_dir)
    qc_dir = Path(args.qc_dir)
    output_path = Path(args.output)
    failure_output = Path(args.failure_output)
    split_summary_output = Path(args.split_summary_output)
    target_spacing = tuple(float(value) for value in args.target_spacing)
    if any(value <= 0 for value in target_spacing):
        raise ValueError(f"Target spacing must be positive, got {target_spacing}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)
    for split_name in ("train", "val", "test"):
        (cache_dir / split_name).mkdir(parents=True, exist_ok=True)
        (qc_dir / split_name).mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    failure_output.parent.mkdir(parents=True, exist_ok=True)
    split_summary_output.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    required = [args.scan_id_col, args.patient_col, args.label_col]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required input column(s): {missing}")

    required_missing = df[df[required].isna().any(axis=1)]
    failures = [
        {
            "new_id": row.get(args.scan_id_col, ""),
            "person_id": row.get(args.patient_col, ""),
            "reason": f"missing required column value among {required}",
        }
        for _, row in required_missing.iterrows()
    ]

    labeled = df.dropna(subset=[args.label_col, args.patient_col, args.scan_id_col]).copy()
    labeled[args.label_col] = labeled[args.label_col].astype(int)
    patient_split = make_patient_split(
        labeled,
        patient_col=args.patient_col,
        label_col=args.label_col,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        allow_nonstratified_fallback=args.allow_nonstratified_fallback,
    )
    split_df = add_split(labeled, patient_split, args.patient_col)

    success_rows = []
    start_time = time.time()
    total_scans = len(split_df)
    all_zero_scan_count = 0
    all_zero_modality_count = 0
    print(f"Input rows: {len(df):,}")
    print(f"Rows excluded before preprocessing: {len(required_missing):,}")
    print(f"Starting preprocessing for {total_scans:,} scans across all splits")
    print(f"Target spacing x/y/z: {format_tuple(target_spacing)}")
    preprocess_failures = 0
    for current_index, (_, row) in enumerate(split_df.iterrows(), start=1):
        scan_id = str(row[args.scan_id_col])
        try:
            image, info = preprocess_scan(
                row,
                raw_image_root,
                cache_dir,
                args.overwrite,
                args.scan_id_col,
                target_spacing,
            )
            qc_path = qc_dir / str(row["split"]) / f"{scan_id}.png"
            qc_ok = 0
            qc_error = ""
            if args.skip_qc:
                qc_error = "skip_qc"
            elif args.overwrite or not qc_path.exists():
                try:
                    write_qc_png(
                        image,
                        qc_path,
                        scan_id,
                        row[args.label_col],
                        row["split"],
                    )
                    qc_ok = 1
                except Exception as exc:
                    qc_error = str(exc)
            else:
                qc_ok = 1
            row_data = row.to_dict()
            row_data.update(info)
            row_data["qc_path"] = str(qc_path.resolve())
            row_data["qc_ok"] = qc_ok
            row_data["qc_error"] = qc_error
            row_data["preprocess_ok"] = 1
            success_rows.append(row_data)
        except Exception as exc:
            reason = str(exc)
            preprocess_failures += 1
            failures.append(
                {
                    "new_id": scan_id,
                    "person_id": row.get(args.patient_col, ""),
                    "reason": reason,
                }
            )
            if reason.startswith("all-zero scan"):
                all_zero_scan_count += 1
                print(
                    f"ALL_ZERO_SCAN_EXCLUDED scan={scan_id} split={row['split']} "
                    f"reason={reason}",
                    file=sys.stderr,
                    flush=True,
                )
            elif reason.startswith("all-zero modality"):
                all_zero_modality_count += 1
                print(
                    f"ALL_ZERO_MODALITY_EXCLUDED scan={scan_id} split={row['split']} "
                    f"reason={reason}",
                    file=sys.stderr,
                    flush=True,
                )
        if args.log_every > 0 and (
            current_index == 1
            or current_index % args.log_every == 0
            or current_index == total_scans
        ):
            elapsed = time.time() - start_time
            rate = current_index / elapsed if elapsed > 0 else 0.0
            remaining = total_scans - current_index
            eta_seconds = remaining / rate if rate > 0 else 0.0
            print(
                f"[{current_index:,}/{total_scans:,}] "
                f"scan={scan_id} "
                f"ok={len(success_rows):,} failed={preprocess_failures:,} "
                f"rate={rate:.2f} scans/s eta={eta_seconds / 60:.1f} min",
                flush=True,
            )

    manifest = pd.DataFrame(success_rows)
    if len(manifest) == 0:
        write_failures(failure_output, failures)
        raise SystemExit(f"No scans preprocessed successfully. See {failure_output}")

    manifest.to_csv(output_path, index=False)
    write_failures(failure_output, failures)
    summary = split_summary(manifest, args.patient_col, args.label_col)
    summary.to_csv(split_summary_output, index=False)

    print(f"Read {len(df):,} rows from {input_path}")
    print(f"Rows with usable label/patient/scan id: {len(labeled):,}")
    print(f"Rows entering patient-wise split/preprocessing: {len(split_df):,}")
    print(f"Preprocessed successfully: {len(manifest):,} scans")
    print(f"Failed or excluded scans reported: {len(failures):,}")
    print(f"All-zero scans excluded: {all_zero_scan_count:,}", file=sys.stderr)
    print(
        f"All-zero-modality scans excluded: {all_zero_modality_count:,}",
        file=sys.stderr,
    )
    qc_ok_count = int(manifest["qc_ok"].sum())
    qc_failed = manifest[manifest["qc_ok"] == 0]
    print(f"QC PNGs written/found: {qc_ok_count:,} scans")
    print(f"QC PNGs missing/failed/skipped: {len(qc_failed):,} scans")
    if len(qc_failed) > 0:
        print("QC issue examples:")
        examples = (
            qc_failed["qc_error"]
            .fillna("")
            .replace("", "unknown")
            .value_counts()
            .head(5)
        )
        for reason, count in examples.items():
            print(f"  {count:,} scans: {reason}")
    print(f"Cache directory split folders: {cache_dir / 'train'}, {cache_dir / 'val'}, {cache_dir / 'test'}")
    print(f"QC directory split folders: {qc_dir / 'train'}, {qc_dir / 'val'}, {qc_dir / 'test'}")
    print(f"Wrote manifest: {output_path}")
    print(f"Wrote failures: {failure_output}")
    print(f"Wrote split summary: {split_summary_output}")
    print("\nSplit summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
