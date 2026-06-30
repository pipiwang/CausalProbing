import argparse
import json
from pathlib import Path

import pandas as pd

from util.adversarial_candidates import ADVERSARIAL_CANDIDATES
from util.paths import GLEASON_CLASSIFICATION_CSV


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


def category_summary(series: pd.Series):
    values = series.dropna()
    if len(values) == 0:
        return [], {}, None, False

    values = values.astype(int)
    counts = values.value_counts().sort_index()
    categories = [int(value) for value in counts.index.tolist()]
    expected = list(range(max(categories) + 1)) if categories else []
    contiguous_zero_based = categories == expected
    num_classes = len(expected) if contiguous_zero_based and len(categories) >= 2 else None
    return categories, {int(k): int(v) for k, v in counts.items()}, num_classes, contiguous_zero_based


def summarize_outcome_distribution(df: pd.DataFrame, label_column: str):
    if label_column not in df.columns:
        return None
    values = df[label_column].dropna().astype(int)
    counts = values.value_counts().sort_index()
    return {int(k): int(v) for k, v in counts.items()}


def summarize_candidate(df: pd.DataFrame, variable: str, spec: dict):
    target_column = spec["target_column"]
    observed_column = spec["observed_column"]
    configured_num_classes = spec.get("adversarial_num_classes")
    allow_raw_class_ids = bool(spec.get("allow_raw_class_ids", False))
    row = {
        "adversarial_variable": variable,
        "adversarial_column": target_column,
        "adversarial_observed_column": observed_column,
        "configured_adversarial_num_classes": configured_num_classes or "",
        "description": spec.get("description", ""),
        "status": "ok",
        "rows": len(df),
        "observed_flag_1": 0,
        "usable_for_adversarial_loss": 0,
        "not_used_for_adversarial_loss": len(df),
        "categories": "",
        "category_counts": "",
        "adversarial_num_classes": "",
        "contiguous_zero_based": False,
        "slurm_variables": "",
        "checker_args": "",
    }

    missing_columns = [
        column
        for column in [target_column, observed_column]
        if column not in df.columns
    ]
    if missing_columns:
        row["status"] = "missing_column:" + ",".join(missing_columns)
        return row

    observed = df[observed_column].apply(is_observed_value)
    target_present = ~df[target_column].apply(is_missing)
    usable = observed & target_present
    categories, counts, num_classes, contiguous = category_summary(df.loc[usable, target_column])

    row["observed_flag_1"] = int(observed.sum())
    row["usable_for_adversarial_loss"] = int(usable.sum())
    row["not_used_for_adversarial_loss"] = int((~usable).sum())
    row["categories"] = json.dumps(categories)
    row["category_counts"] = json.dumps(counts)
    row["contiguous_zero_based"] = bool(contiguous)
    if configured_num_classes is not None:
        row["adversarial_num_classes"] = int(configured_num_classes)
        if len(categories) < 2:
            row["status"] = "fewer_than_two_observed_categories"
        elif categories and max(categories) >= int(configured_num_classes):
            row["status"] = "category_exceeds_configured_num_classes"
        elif not contiguous and not allow_raw_class_ids:
            row["status"] = "non_contiguous_categories"
        elif allow_raw_class_ids and not contiguous:
            row["status"] = "ok_raw_class_ids"
    elif num_classes is not None:
        row["adversarial_num_classes"] = int(num_classes)
    else:
        if len(categories) < 2:
            row["status"] = "fewer_than_two_observed_categories"
        else:
            row["status"] = "non_contiguous_categories"

    if row["status"] in {"ok", "ok_raw_class_ids"}:
        row["slurm_variables"] = (
            f"ADVERSARIAL_VARIABLE={variable};"
            f"ADVERSARIAL_COLUMN={target_column};"
            f"ADVERSARIAL_OBSERVED_COLUMN={observed_column};"
            f"ADVERSARIAL_NUM_CLASSES={row['adversarial_num_classes']}"
        )
        row["checker_args"] = (
            f"--adversarial_variable {variable} "
            f"--adversarial_column {target_column} "
            f"--adversarial_observed_column {observed_column} "
            f"--adversarial_num_classes {row['adversarial_num_classes']}"
        )
    return row


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize candidate adversarial variables and category counts."
    )
    parser.add_argument("--csv_path", default=str(GLEASON_CLASSIFICATION_CSV), type=str)
    parser.add_argument(
        "--output",
        default="adversarial_variable_candidates.csv",
        type=str,
        help="Path for candidate summary CSV.",
    )
    parser.add_argument(
        "--include_missing",
        action="store_true",
        help="Keep candidates with missing columns in the printed table.",
    )
    parser.add_argument(
        "--label_column",
        default="grade_group",
        type=str,
        help="Outcome label column to summarize separately, not as an adversarial candidate.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.csv_path)
    rows = [
        summarize_candidate(df, variable, spec)
        for variable, spec in ADVERSARIAL_CANDIDATES.items()
    ]
    out = pd.DataFrame(rows)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    printable = (
        out
        if args.include_missing
        else out[out["status"].isin(["ok", "ok_raw_class_ids"])]
    )
    columns = [
        "adversarial_variable",
        "adversarial_column",
        "adversarial_observed_column",
        "configured_adversarial_num_classes",
        "adversarial_num_classes",
        "usable_for_adversarial_loss",
        "not_used_for_adversarial_loss",
        "category_counts",
        "status",
    ]
    print(f"Read {len(df):,} rows from {args.csv_path}")
    print(f"Wrote candidate table: {output_path}")
    outcome_counts = summarize_outcome_distribution(df, args.label_column)
    if outcome_counts is not None:
        print(f"\nOutcome distribution, not adversarial: {args.label_column}")
        print(json.dumps(outcome_counts, sort_keys=True))
    else:
        print(f"\nOutcome distribution skipped; missing column: {args.label_column}")
    print("\nAdversarial candidate distribution:")
    print(printable[columns].to_string(index=False))


if __name__ == "__main__":
    main()
