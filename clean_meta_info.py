"""Build a model-ready clinical metadata table for MRI-only Gleason experiments.

The output keeps one row per MRI scan and creates:
- cleaned numeric covariates
- coarse categorical adversarial targets
- *_observed masks for masked losses
- derived grouped comorbidity indicators
- parsed Gleason/Grade Group labels
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

from util.paths import (
    META_INFO_CLEANED_CSV,
    META_INFO_CLEANED_DICTIONARY_CSV,
    META_INFO_PROCESSED_CSV,
)


ID_COLUMNS = ["new_id", "pseudo_study_uid", "person_id", "procedure_date"]
REQUIRED_MODALITY_COLUMNS = ["t2", "dwi", "adc"]
PSA_SOURCE_COLUMN = "scan_current_psa_value_at_the_time_of_mri"

BINARY_COLUMNS = [
    "hypertension",
    "cardiovascular_disease",
    "chronic_kidney_disease",
    "diabetes",
    "copd",
    "asthma",
    "dre_abnormal",
]

ENCODED_CATEGORY_COLUMNS = ["smoking_encoded", "alcohol_encoded"]


def true_mask(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(
        {"true", "t", "1", "1.0", "yes", "y"}
    )


def filter_complete_mri_modalities(raw: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED_MODALITY_COLUMNS if column not in raw.columns]
    if missing:
        raise KeyError(f"Missing required modality column(s): {missing}")

    keep = pd.Series(True, index=raw.index)
    for column in REQUIRED_MODALITY_COLUMNS:
        keep &= true_mask(raw[column])
    return raw.loc[keep].reset_index(drop=True)


def count_patients(df: pd.DataFrame) -> int:
    if "person_id" not in df.columns:
        raise KeyError("Missing required patient identifier column: person_id")
    return int(df["person_id"].nunique(dropna=True))


def parse_procedure_date(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip()
    iso_mask = text.str.match(r"^\d{4}-\d{2}-\d{2}", na=False)
    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    parsed.loc[iso_mask] = pd.to_datetime(
        text.loc[iso_mask],
        format="%Y-%m-%d",
        errors="coerce",
    )
    parsed.loc[~iso_mask] = pd.to_datetime(
        text.loc[~iso_mask],
        dayfirst=True,
        errors="coerce",
    )
    return parsed


def observed_mask(series: pd.Series) -> pd.Series:
    return series.notna().astype("int8")


def coerce_numeric(df: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")


def clean_binary(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    values = values.where(values.isin([0, 1]))
    return values.astype("Int8")


def clean_nullable_int(
    series: pd.Series,
    dtype: str = "Int16",
    min_value: int | None = None,
    max_value: int | None = None,
) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    keep = values.notna() & np.isclose(values, np.round(values))
    if min_value is not None:
        keep &= values >= min_value
    if max_value is not None:
        keep &= values <= max_value
    values = values.round().where(keep, pd.NA)
    return values.astype(dtype)


def make_bin(
    series: pd.Series,
    bins: list[float],
    labels: list[str],
) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return pd.cut(numeric, bins=bins, labels=labels, right=False, ordered=True)


def add_categorical_codes(df: pd.DataFrame, column: str) -> None:
    series = df[column]
    if isinstance(series.dtype, CategoricalDtype):
        codes = series.cat.codes.astype("Int16")
        df[f"{column}_code"] = codes.where(codes >= 0, pd.NA)
        return

    observed = series.notna()
    categories = sorted(series.loc[observed].dropna().unique())
    mapping = {category: idx for idx, category in enumerate(categories)}
    df[f"{column}_code"] = series.map(mapping).astype("Int16")


def parse_gleason_score(value: object) -> tuple[float, float, float]:
    if pd.isna(value):
        return np.nan, np.nan, np.nan

    text = str(value).strip().lower()
    if not text or text in {"nan", "none", "null"}:
        return np.nan, np.nan, np.nan

    if "+" in text:
        left, right = text.split("+", 1)
        try:
            primary = int(left.strip())
            secondary = int(right.strip())
        except ValueError:
            return np.nan, np.nan, np.nan
        return primary, secondary, primary + secondary

    try:
        total = int(float(text))
    except ValueError:
        return np.nan, np.nan, np.nan

    return np.nan, np.nan, total


def gleason_total_to_grade_group(
    total: float,
    primary: float = np.nan,
    secondary: float = np.nan,
) -> float:
    if pd.isna(total):
        return np.nan
    if total <= 6:
        return 1
    if total == 7:
        if primary == 3 and secondary == 4:
            return 2
        if primary == 4 and secondary == 3:
            return 3
        return np.nan
    if total == 8:
        return 4
    if total >= 9:
        return 5
    return np.nan


def any_observed_binary(df: pd.DataFrame, columns: list[str]) -> tuple[pd.Series, pd.Series]:
    present = [column for column in columns if column in df.columns]
    if not present:
        empty = pd.Series(pd.NA, index=df.index, dtype="Int8")
        return empty, pd.Series(0, index=df.index, dtype="int8")

    block = df[present].apply(pd.to_numeric, errors="coerce")
    observed = block.notna().any(axis=1)
    any_positive = block.eq(1).any(axis=1)
    result = any_positive.astype("Int8")
    result = result.where(observed, pd.NA)
    return result, observed.astype("int8")


def build_clean_metadata(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()

    numeric_columns = [
        "bmi",
        "psa_value",
        PSA_SOURCE_COLUMN,
        "scan_prostate_volume_ml",
        "max_pirads",
        "age",
        *BINARY_COLUMNS,
        *ENCODED_CATEGORY_COLUMNS,
        "csPCa",
    ]
    coerce_numeric(df, numeric_columns)
    if PSA_SOURCE_COLUMN in df.columns:
        df["psa_value"] = df[PSA_SOURCE_COLUMN]

    out = pd.DataFrame(index=df.index)

    for column in ID_COLUMNS:
        if column in df.columns:
            out[column] = df[column]

    if "procedure_date" in out.columns:
        out["procedure_date"] = parse_procedure_date(out["procedure_date"])

    for column in ["bmi", "psa_value", "scan_prostate_volume_ml", "max_pirads", "age"]:
        if column in df.columns:
            out[column] = df[column]
            out[f"{column}_observed"] = observed_mask(df[column])

    if "psa_value" in out.columns:
        out["log1p_psa_value"] = np.log1p(out["psa_value"].clip(lower=0))
        out["psa_group"] = make_bin(
            out["psa_value"],
            bins=[0, 4, 10, 20, np.inf],
            labels=["lt4", "4_to_10", "10_to_20", "gte20"],
        )
        add_categorical_codes(out, "psa_group")

    if "bmi" in out.columns:
        out["bmi_group"] = make_bin(
            out["bmi"],
            bins=[0, 25, 30, np.inf],
            labels=["lt25", "25_to_30", "gte30"],
        )
        add_categorical_codes(out, "bmi_group")

    if "scan_prostate_volume_ml" in out.columns:
        out["log1p_scan_prostate_volume_ml"] = np.log1p(
            out["scan_prostate_volume_ml"].clip(lower=0)
        )
        out["prostate_volume_group"] = make_bin(
            out["scan_prostate_volume_ml"],
            bins=[0, 30, 60, np.inf],
            labels=["lt30", "30_to_60", "gte60"],
        )
        add_categorical_codes(out, "prostate_volume_group")

    if "age" in out.columns:
        out["age_group"] = make_bin(
            out["age"],
            bins=[0, 60, 70, 80, np.inf],
            labels=["lt60", "60_to_70", "70_to_80", "gte80"],
        )
        add_categorical_codes(out, "age_group")

    if "max_pirads" in out.columns:
        out["max_pirads"] = clean_nullable_int(
            out["max_pirads"],
            dtype="Int8",
            min_value=1,
            max_value=5,
        )
        out["pirads_high"] = (out["max_pirads"] >= 4).astype("Int8")
        out["pirads_high"] = out["pirads_high"].where(out["max_pirads"].notna(), pd.NA)
        out["pirads_high_observed"] = observed_mask(out["pirads_high"])

    if "scan_gleason_score" in df.columns:
        parsed = df["scan_gleason_score"].apply(parse_gleason_score)
        gleason = pd.DataFrame(
            parsed.tolist(),
            columns=["gleason_primary", "gleason_secondary", "gleason_total"],
            index=df.index,
        )
        out = pd.concat([out, gleason], axis=1)
        out["grade_group"] = [
            gleason_total_to_grade_group(total, primary, secondary)
            for total, primary, secondary in zip(
                out["gleason_total"],
                out["gleason_primary"],
                out["gleason_secondary"],
            )
        ]
        out["grade_group"] = clean_nullable_int(
            out["grade_group"],
            dtype="Int8",
            min_value=1,
            max_value=5,
        )
        out["grade_group_observed"] = observed_mask(out["grade_group"])
        out["gleason_binary_cs"] = (out["gleason_total"] >= 7).astype("Int8")
        out["gleason_binary_cs"] = out["gleason_binary_cs"].where(
            out["gleason_total"].notna(),
            pd.NA,
        )

    if "csPCa" in df.columns:
        out["csPCa"] = clean_binary(df["csPCa"])
        out["csPCa_observed"] = observed_mask(out["csPCa"])

    for column in BINARY_COLUMNS:
        if column in df.columns:
            out[column] = clean_binary(df[column])
            out[f"{column}_observed"] = observed_mask(out[column])

    for column in ENCODED_CATEGORY_COLUMNS:
        if column in df.columns:
            out[column] = clean_nullable_int(df[column], dtype="Int16", min_value=0)
            out[f"{column}_observed"] = observed_mask(out[column])

    grouped_targets = {
        "cardio_any": ["hypertension", "cardiovascular_disease"],
        "respiratory_any": ["copd", "asthma"],
        "renal_metabolic_any": ["diabetes", "chronic_kidney_disease"],
    }
    for group_name, columns in grouped_targets.items():
        out[group_name], out[f"{group_name}_observed"] = any_observed_binary(out, columns)

    return out


def write_data_dictionary(output_path: Path, clean: pd.DataFrame) -> None:
    rows = []
    for column in clean.columns:
        if column.endswith("_observed"):
            role = "mask"
        elif column.endswith("_code"):
            role = "categorical_code"
        elif column in {"grade_group", "gleason_binary_cs", "csPCa"}:
            role = "outcome"
        elif column in ID_COLUMNS:
            role = "identifier"
        else:
            role = "feature"
        rows.append({"column": column, "role": role, "dtype": str(clean[column].dtype)})

    dictionary = pd.DataFrame(rows)
    dictionary.to_csv(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=str(META_INFO_PROCESSED_CSV),
        help="Raw or semi-processed metadata CSV.",
    )
    parser.add_argument(
        "--output",
        default=str(META_INFO_CLEANED_CSV),
        help="Path for cleaned metadata CSV.",
    )
    parser.add_argument(
        "--dictionary-output",
        default=str(META_INFO_CLEANED_DICTIONARY_CSV),
        help="Path for a compact data dictionary CSV.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    dictionary_path = Path(args.dictionary_output)

    raw = pd.read_csv(input_path)
    filtered = filter_complete_mri_modalities(raw)
    clean = build_clean_metadata(filtered)
    clean.to_csv(output_path, index=False)
    write_data_dictionary(dictionary_path, clean)

    print(f"Read {len(raw):,} rows from {input_path}")
    print(
        "Kept "
        f"{len(filtered):,} rows with t2, dwi, and adc all TRUE "
        f"({count_patients(filtered):,} unique patients by person_id)"
    )
    print(f"Wrote {len(clean):,} rows x {len(clean.columns):,} columns to {output_path}")
    print(f"Output contains {count_patients(clean):,} unique patients by person_id")
    print(f"Wrote data dictionary to {dictionary_path}")


if __name__ == "__main__":
    main()
