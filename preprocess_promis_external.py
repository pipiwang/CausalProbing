"""Build an external-test Gleason manifest from the PROMIS NIfTI dataset.

The Gleason classifier consumes preprocessed .npy caches with shape
[3, Z, H, W] and a manifest column named image_npy_path. PROMIS rows contain
relative NIfTI paths instead, so this script stacks the already-resampled
T2/DWI/ADC volumes and writes all rows as an external test split.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


MODALITY_COLUMNS = {
    "t2": "t2w",
    "dwi": "dwi",
    "adc": "adc",
}
TARGET_SPACING = (0.5, 0.5, 1.0)


def require_simpleitk():
    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise SystemExit(
            "SimpleITK is required for preprocessing. Install it in the Python environment."
        ) from exc
    return sitk


def format_tuple(values) -> str:
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
        for modality, modality_image in zip(MODALITY_COLUMNS, image)
        if not (np.isfinite(modality_image) & (modality_image != 0)).any()
    ]
    return all_zero_scan, ";".join(all_zero_modalities)


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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess PROMIS scans into a Gleason external-test manifest."
    )
    parser.add_argument("--input", default="data/promis_all.csv", type=str)
    parser.add_argument(
        "--raw-image-root",
        default="data/promis_recurated_resampled",
        type=str,
    )
    parser.add_argument(
        "--cache-dir",
        default="data/promis_external_gleason/img",
        type=str,
    )
    parser.add_argument(
        "--qc-dir",
        default="data/promis_external_gleason/qc",
        type=str,
    )
    parser.add_argument(
        "--output",
        default="data/promis_external_gleason.csv",
        type=str,
    )
    parser.add_argument(
        "--failure-output",
        default="data/promis_external_gleason_failures.csv",
        type=str,
    )
    parser.add_argument("--gleason-col", default="gleason", type=str)
    parser.add_argument(
        "--grade-group-offset",
        default=1,
        type=int,
        help=(
            "Value added to PROMIS gleason to create grade_group. "
            "Default maps PROMIS 0..4 to grade_group 1..5."
        ),
    )
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument(
        "--resample",
        action="store_true",
        help=(
            "Resample DWI/ADC to T2 and all modalities to --target-spacing. "
            "By default PROMIS is assumed to already be resampled."
        ),
    )
    parser.add_argument(
        "--target-spacing",
        default=TARGET_SPACING,
        nargs=3,
        type=float,
        metavar=("SX", "SY", "SZ"),
        help=(
            "Expected voxel spacing, and output spacing when --resample is used, "
            "in SimpleITK x y z order. Default: 0.5 0.5 1.0."
        ),
    )
    parser.add_argument(
        "--expected-inplane-size",
        default=(256, 256),
        nargs=2,
        type=int,
        metavar=("X", "Y"),
        help="Expected PROMIS in-plane image size in SimpleITK x y order.",
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
        "--limit",
        default=None,
        type=int,
        help="Optional debug limit on input rows.",
    )
    return parser.parse_args()


def resolve_path(raw_image_root: Path, value) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return raw_image_root / path


def scan_id_from_row(row) -> str:
    return Path(str(row[MODALITY_COLUMNS["t2"]])).parent.name


def read_modality_images(row, raw_image_root: Path):
    sitk = require_simpleitk()
    paths = {
        name: resolve_path(raw_image_root, row[column])
        for name, column in MODALITY_COLUMNS.items()
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing image file(s): " + ", ".join(missing))
    images = {
        name: sitk.ReadImage(str(path), sitk.sitkFloat32)
        for name, path in paths.items()
    }
    return images, paths


def image_geometry(image):
    return {
        "size": tuple(int(v) for v in image.GetSize()),
        "spacing": tuple(float(v) for v in image.GetSpacing()),
        "origin": tuple(float(v) for v in image.GetOrigin()),
        "direction": tuple(float(v) for v in image.GetDirection()),
    }


def validate_preprocessed_geometry(
    images,
    expected_spacing: tuple[float, float, float],
    expected_inplane_size: tuple[int, int],
):
    reference = image_geometry(images["t2"])
    for name in ("dwi", "adc"):
        geometry = image_geometry(images[name])
        if geometry["size"] != reference["size"]:
            raise ValueError(
                f"{name} size {geometry['size']} does not match T2 size "
                f"{reference['size']}. Use --resample if this is expected."
            )
        for key in ("spacing", "origin", "direction"):
            if not np.allclose(geometry[key], reference[key], rtol=0, atol=1e-5):
                raise ValueError(
                    f"{name} {key} does not match T2. Use --resample if this is expected."
                )

    if not np.allclose(reference["spacing"], expected_spacing, rtol=0, atol=1e-5):
        raise ValueError(
            f"T2 spacing {reference['spacing']} does not match expected "
            f"{expected_spacing}. Use --target-spacing or --resample if needed."
        )

    if reference["size"][:2] != tuple(expected_inplane_size):
        raise ValueError(
            f"T2 in-plane size {reference['size'][:2]} does not match expected "
            f"{tuple(expected_inplane_size)}."
        )


def modality_arrays(images):
    sitk = require_simpleitk()
    arrays = [
        sitk.GetArrayFromImage(images[name]).astype(np.float32)
        for name in ("t2", "dwi", "adc")
    ]
    shapes = [array.shape for array in arrays]
    if len(set(shapes)) != 1:
        raise ValueError(f"modality array shapes do not match: {shapes}")
    return arrays


def resampled_modality_arrays(images, target_spacing: tuple[float, float, float]):
    dwi_on_t2 = resample_to_reference(images["dwi"], images["t2"])
    adc_on_t2 = resample_to_reference(images["adc"], images["t2"])
    resampled = [
        resample_to_spacing(images["t2"], target_spacing),
        resample_to_spacing(dwi_on_t2, target_spacing),
        resample_to_spacing(adc_on_t2, target_spacing),
    ]
    return modality_arrays({"t2": resampled[0], "dwi": resampled[1], "adc": resampled[2]})


def preprocess_promis_scan(
    row,
    raw_image_root: Path,
    cache_dir: Path,
    overwrite: bool,
    target_spacing: tuple[float, float, float],
    expected_inplane_size: tuple[int, int],
    split: str,
    resample: bool,
):
    scan_id = scan_id_from_row(row)
    images, paths = read_modality_images(row, raw_image_root)

    original_sizes = {
        "t2_original_size": image_size(images["t2"]),
        "dwi_original_size": image_size(images["dwi"]),
        "adc_original_size": image_size(images["adc"]),
        "t2_original_spacing": image_spacing(images["t2"]),
        "dwi_original_spacing": image_spacing(images["dwi"]),
        "adc_original_spacing": image_spacing(images["adc"]),
    }

    if resample:
        arrays = resampled_modality_arrays(images, target_spacing)
    else:
        validate_preprocessed_geometry(images, target_spacing, expected_inplane_size)
        arrays = modality_arrays(images)

    image = np.stack(arrays, axis=0).astype(np.float32)
    all_zero_scan, all_zero_modalities = zero_content_report(image)
    if all_zero_scan:
        raise ValueError("all-zero scan after preprocessing")
    if all_zero_modalities:
        raise ValueError(f"all-zero modality after preprocessing: {all_zero_modalities}")

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


def write_failures(path: Path, failures: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
    target_spacing = tuple(float(value) for value in args.target_spacing)
    expected_inplane_size = tuple(int(value) for value in args.expected_inplane_size)

    if any(value <= 0 for value in target_spacing):
        raise ValueError(f"Target spacing must be positive, got {target_spacing}")

    df = pd.read_csv(input_path)
    if args.limit is not None:
        df = df.head(int(args.limit)).copy()

    required = [*MODALITY_COLUMNS.values(), args.gleason_col]
    missing_columns = [column for column in required if column not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing required input column(s): {missing_columns}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    failure_output.parent.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / args.split).mkdir(parents=True, exist_ok=True)
    (qc_dir / args.split).mkdir(parents=True, exist_ok=True)

    missing_required = df[df[required].isna().any(axis=1)]
    failures = [
        {
            "new_id": scan_id_from_row(row),
            "person_id": scan_id_from_row(row),
            "reason": f"missing required column value among {required}",
        }
        for _, row in missing_required.iterrows()
    ]
    work = df.dropna(subset=required).copy()
    work[args.gleason_col] = work[args.gleason_col].astype(int)

    success_rows = []
    start_time = time.time()
    print(f"Input rows: {len(df):,}")
    print(f"Rows excluded before preprocessing: {len(missing_required):,}")
    print(f"Starting PROMIS preprocessing for {len(work):,} scans")
    print(f"PROMIS root: {raw_image_root}")
    print(f"PROMIS preprocessing mode: {'resample' if args.resample else 'stack-only'}")
    print(f"Expected/target spacing x/y/z: {format_tuple(target_spacing)}")
    print(f"Expected in-plane size x/y: {format_tuple(expected_inplane_size)}")

    for current_index, (_, row) in enumerate(work.iterrows(), start=1):
        scan_id = scan_id_from_row(row)
        try:
            image, info = preprocess_promis_scan(
                row=row,
                raw_image_root=raw_image_root,
                cache_dir=cache_dir,
                overwrite=args.overwrite,
                target_spacing=target_spacing,
                expected_inplane_size=expected_inplane_size,
                split=args.split,
                resample=args.resample,
            )
            grade_group = int(row[args.gleason_col]) + int(args.grade_group_offset)
            qc_path = qc_dir / args.split / f"{scan_id}.png"
            qc_ok = 0
            qc_error = ""
            if args.skip_qc:
                qc_error = "skip_qc"
            elif args.overwrite or not qc_path.exists():
                try:
                    write_qc_png(image, qc_path, scan_id, grade_group, args.split)
                    qc_ok = 1
                except Exception as exc:
                    qc_error = str(exc)
            else:
                qc_ok = 1

            row_data = row.to_dict()
            row_data.update(
                {
                    "new_id": scan_id,
                    "person_id": scan_id,
                    "pseudo_study_uid": scan_id,
                    "split": args.split,
                    "promis_gleason": int(row[args.gleason_col]),
                    "grade_group": grade_group,
                    "gleason_binary_cs": int(grade_group >= 2),
                    "csPCa": int(grade_group >= 2),
                    "qc_path": str(qc_path.resolve()),
                    "qc_ok": qc_ok,
                    "qc_error": qc_error,
                    "preprocess_ok": 1,
                }
            )
            row_data.update(info)
            success_rows.append(row_data)
        except Exception as exc:
            failures.append(
                {
                    "new_id": scan_id,
                    "person_id": scan_id,
                    "reason": str(exc),
                }
            )

        if args.log_every > 0 and (
            current_index == 1
            or current_index % args.log_every == 0
            or current_index == len(work)
        ):
            elapsed = time.time() - start_time
            rate = current_index / elapsed if elapsed > 0 else 0.0
            remaining = len(work) - current_index
            eta_seconds = remaining / rate if rate > 0 else 0.0
            print(
                f"[{current_index:,}/{len(work):,}] "
                f"scan={scan_id} ok={len(success_rows):,} "
                f"failed={len(failures):,} rate={rate:.2f} scans/s "
                f"eta={eta_seconds / 60:.1f} min",
                flush=True,
            )

    manifest = pd.DataFrame(success_rows)
    if len(manifest) == 0:
        write_failures(failure_output, failures)
        raise SystemExit(f"No scans preprocessed successfully. See {failure_output}")

    manifest.to_csv(output_path, index=False)
    write_failures(failure_output, failures)

    print(f"Preprocessed successfully: {len(manifest):,} scans")
    print(f"Failed or excluded scans reported: {len(failures):,}", file=sys.stderr)
    print(f"Wrote manifest: {output_path}")
    print(f"Wrote failures: {failure_output}")


if __name__ == "__main__":
    main()
