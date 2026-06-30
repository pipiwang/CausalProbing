#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import pandas as pd

from util.adversarial_candidates import get_adversarial_candidate


def is_missing(value) -> bool:
    if pd.isna(value):
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def is_observed_value(value) -> bool:
    if is_missing(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y"}
    return int(value) == 1


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create a Gleason complete-case CSV for one adversarial variable. "
            "Rows are retained only when the configured observed mask is true "
            "and the configured adversarial target column is non-missing."
        )
    )
    parser.add_argument("--input_csv", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--variable", required=True)
    parser.add_argument(
        "--target_column",
        default=None,
        help="Override the configured adversarial target column.",
    )
    parser.add_argument(
        "--observed_column",
        default=None,
        help="Override the configured observed-mask column.",
    )
    parser.add_argument("--patient_col", default="person_id")
    parser.add_argument("--split_col", default="split")
    parser.add_argument(
        "--summary_csv",
        type=Path,
        default=None,
        help="Optional append-only CSV with before/after row and patient counts.",
    )
    return parser.parse_args()


def count_patients(df, patient_col):
    if patient_col not in df.columns:
        return ""
    return int(df[patient_col].nunique(dropna=True))


def split_counts(df, split_col):
    if split_col not in df.columns:
        return {}
    return df[split_col].astype(str).value_counts(dropna=False).sort_index().to_dict()


def main():
    args = parse_args()
    candidate = get_adversarial_candidate(args.variable)
    if not candidate and (args.target_column is None or args.observed_column is None):
        raise ValueError(
            f"Unknown variable '{args.variable}'. Provide both --target_column and "
            "--observed_column to use an ad hoc variable."
        )

    target_column = args.target_column or candidate["target_column"]
    observed_column = args.observed_column or candidate["observed_column"]

    df = pd.read_csv(args.input_csv)
    missing_columns = [
        column
        for column in [target_column, observed_column]
        if column not in df.columns
    ]
    if missing_columns:
        raise KeyError(f"Missing required column(s): {missing_columns}")

    observed = df[observed_column].apply(is_observed_value)
    target_present = ~df[target_column].apply(is_missing)
    keep = observed & target_present
    complete = df.loc[keep].copy()
    if complete.empty:
        raise ValueError(
            f"No complete-case rows for {args.variable}: "
            f"{observed_column}=true and {target_column}=non-missing"
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    complete.to_csv(args.output_csv, index=False)

    summary = {
        "variable": args.variable,
        "target_column": target_column,
        "observed_column": observed_column,
        "input_csv": str(args.input_csv),
        "output_csv": str(args.output_csv),
        "rows_before": len(df),
        "rows_after": len(complete),
        "rows_excluded": len(df) - len(complete),
        "patients_before": count_patients(df, args.patient_col),
        "patients_after": count_patients(complete, args.patient_col),
        "patients_excluded": (
            count_patients(df, args.patient_col)
            - count_patients(complete, args.patient_col)
            if args.patient_col in df.columns
            else ""
        ),
        "split_counts_before": split_counts(df, args.split_col),
        "split_counts_after": split_counts(complete, args.split_col),
    }

    print("Complete-case CSV written")
    for key, value in summary.items():
        print(f"{key}: {value}")

    if args.summary_csv:
        args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
        fields = list(summary)
        write_header = not args.summary_csv.exists()
        with args.summary_csv.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            if write_header:
                writer.writeheader()
            writer.writerow(summary)


if __name__ == "__main__":
    main()
