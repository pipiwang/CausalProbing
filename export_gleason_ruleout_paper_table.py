import argparse
import csv
import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "log_hpc"
    / "results"
    / "ruleout_stats_best_auc_seed_paired_auc_balanced_acc_sens80spec.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "log_hpc"
    / "results"
    / "ruleout_paper_table_best_auc_seed_bootstrap.csv"
)

TARGET_ORDER = (
    "ruleout_age",
    "ruleout_bmi",
    "ruleout_alcohol_encoded",
    "ruleout_smoking_encoded",
    "ruleout_cardio_any",
    "ruleout_respiratory_any",
    "ruleout_dre_abnormal",
    "ruleout_diabetes",
    "ruleout_renal_metabolic_any",
    "ruleout_max_pirads",
    "ruleout_pirads_high",
    "ruleout_psa_value",
    "ruleout_scan_prostate_volume_ml",
)
TARGET_LABELS = {
    "ruleout_none": "Baseline",
    "ruleout_age": "Age",
    "ruleout_bmi": "BMI",
    "ruleout_alcohol_encoded": "Alcohol use",
    "ruleout_smoking_encoded": "Smoking status",
    "ruleout_cardio_any": "Cardiovascular disease",
    "ruleout_respiratory_any": "Respiratory disease",
    "ruleout_dre_abnormal": "Abnormal DRE",
    "ruleout_diabetes": "Diabetes",
    "ruleout_renal_metabolic_any": "Renal/metabolic condition",
    "ruleout_max_pirads": "PI-RADS score",
    "ruleout_pirads_high": "PI-RADS binary cutoff",
    "ruleout_psa_value": "PSA value",
    "ruleout_scan_prostate_volume_ml": "Prostate volume",
    "ruleout_scan_prostate_volume": "Prostate volume",
}
PAPER_HEADERS = [
    "Variable",
    "AUC",
    "95CI",
    "Sens@80spec",
    "95CI",
    "deltaAUC",
    "pvalue",
    "significance indicator (n.s, *, **, ***)",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert seed-paired Gleason rule-out stats into a compact paper-table CSV."
        )
    )
    parser.add_argument("--input_csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output_csv", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--selection", default="best_auc")
    parser.add_argument("--baseline", default="ruleout_none")
    parser.add_argument("--analysis_method", default="seed_paired_bootstrap")
    parser.add_argument("--auc_metric", default="test_auc")
    parser.add_argument("--sens_metric", default="test_sens_at_80_spec")
    parser.add_argument("--metric_digits", type=int, default=2)
    parser.add_argument("--pvalue_digits", type=int, default=4)
    parser.add_argument(
        "--allow_missing",
        action="store_true",
        help="Skip targets missing either the AUC or Sens@80spec bootstrap row.",
    )
    return parser.parse_args()


def to_float(value):
    if value is None:
        return None
    value = str(value).strip()
    if value == "" or value.upper() == "NA":
        return None
    number = float(value)
    if math.isnan(number):
        return None
    return number


def format_number(value, digits):
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def format_pvalue(value, digits):
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def format_ci(base_value, ci_low, ci_high, digits):
    if base_value is None or ci_low is None or ci_high is None:
        return ""
    return f"({base_value + ci_low:.{digits}f}, {base_value + ci_high:.{digits}f})"


def clinical_label(name):
    name = str(name)
    if name in TARGET_LABELS:
        return TARGET_LABELS[name]
    cleaned = name.removeprefix("ruleout_").removesuffix("_encoded")
    cleaned = cleaned.replace("_ml", "").replace("_", " ")
    replacements = {
        "psa": "PSA",
        "pirads": "PI-RADS",
        "bmi": "BMI",
        "scan prostate volume": "Prostate volume",
    }
    return replacements.get(cleaned, cleaned.title())


def target_sort_key(target):
    target = str(target)
    if target in TARGET_ORDER:
        return (TARGET_ORDER.index(target), target)
    return (len(TARGET_ORDER), clinical_label(target))


def significance_indicator(pvalue):
    if pvalue is None:
        return ""
    if pvalue < 0.001:
        return "***"
    if pvalue < 0.01:
        return "**"
    if pvalue < 0.05:
        return "*"
    return "n.s"


def read_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not reader.fieldnames:
        raise ValueError(f"No header row found in {path}")
    return reader.fieldnames, rows


def require_columns(fieldnames, required, path):
    missing = sorted(set(required).difference(fieldnames))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def filtered_bootstrap_rows(rows, args):
    filtered = []
    for row in rows:
        if str(row.get("selection", "")) != args.selection:
            continue
        if str(row.get("baseline", "")) != args.baseline:
            continue
        if str(row.get("analysis_method", "")) != args.analysis_method:
            continue
        if row.get("metric") not in {args.auc_metric, args.sens_metric}:
            continue
        filtered.append(row)
    if not filtered:
        raise ValueError(
            "No rows matched "
            f"selection={args.selection!r}, baseline={args.baseline!r}, "
            f"analysis_method={args.analysis_method!r}"
        )
    return filtered


def formatted_metric_row(row, value_column, args):
    value = to_float(row[value_column])
    ci_base = to_float(row["baseline_mean"])
    return value, format_number(value, args.metric_digits), format_ci(
        ci_base,
        to_float(row["ci_low"]),
        to_float(row["ci_high"]),
        args.metric_digits,
    )


def build_baseline_row(rows, args):
    auc_row = next(row for row in rows if row["metric"] == args.auc_metric)
    sens_row = next(row for row in rows if row["metric"] == args.sens_metric)
    _, auc_text, _ = formatted_metric_row(auc_row, "baseline_mean", args)
    _, sens_text, _ = formatted_metric_row(sens_row, "baseline_mean", args)
    return [
        clinical_label(args.baseline),
        auc_text,
        "",
        sens_text,
        "",
        format_number(0.0, args.metric_digits),
        "",
        "",
    ]


def build_paper_rows(rows, args):
    by_target_metric = {
        (str(row["target"]), str(row["metric"])): row
        for row in rows
    }
    targets = sorted({str(row["target"]) for row in rows}, key=target_sort_key)
    paper_rows = [build_baseline_row(rows, args)]
    skipped = []

    for target in targets:
        auc_row = by_target_metric.get((target, args.auc_metric))
        sens_row = by_target_metric.get((target, args.sens_metric))
        if auc_row is None or sens_row is None:
            skipped.append(target)
            if args.allow_missing:
                continue
            missing = []
            if auc_row is None:
                missing.append(args.auc_metric)
            if sens_row is None:
                missing.append(args.sens_metric)
            raise ValueError(f"{target} is missing bootstrap rows for: {', '.join(missing)}")

        _, auc_text, auc_ci = formatted_metric_row(auc_row, "target_mean", args)
        _, sens_text, sens_ci = formatted_metric_row(sens_row, "target_mean", args)
        delta_auc = to_float(auc_row["mean_delta"])
        pvalue = to_float(auc_row["p_two_sided"])
        paper_rows.append(
            [
                clinical_label(target),
                auc_text,
                auc_ci,
                sens_text,
                sens_ci,
                format_number(delta_auc, args.metric_digits),
                format_pvalue(pvalue, args.pvalue_digits),
                significance_indicator(pvalue),
            ]
        )
    return paper_rows, skipped


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(PAPER_HEADERS)
        writer.writerows(rows)


def main():
    args = parse_args()
    source_fieldnames, source_rows = read_rows(args.input_csv)
    require_columns(
        source_fieldnames,
        {
            "selection",
            "baseline",
            "target",
            "metric",
            "baseline_mean",
            "target_mean",
            "mean_delta",
            "analysis_method",
            "ci_low",
            "ci_high",
            "p_two_sided",
        },
        args.input_csv,
    )
    bootstrap_rows = filtered_bootstrap_rows(source_rows, args)
    paper_rows, skipped = build_paper_rows(bootstrap_rows, args)
    write_csv(args.output_csv, paper_rows)

    print(f"Wrote {len(paper_rows)} paper-table rows to {args.output_csv}")
    if skipped:
        print(f"Skipped targets with missing metrics: {', '.join(skipped)}")


if __name__ == "__main__":
    main()
