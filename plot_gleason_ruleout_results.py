import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SELECTION_FILES = {
    "best_auc": ("test_metrics_best_auc.json", "test_metrics_best_auc_mri_only.json"),
    "best_balanced_acc": (
        "test_metrics_best_balanced_acc.json",
        "test_metrics_best_balanced_acc_mri_only.json",
    ),
    "best_loss": ("test_metrics_best_loss.json", "test_metrics_best_loss_mri_only.json"),
    "best_qwk": ("test_metrics_best_qwk.json", "test_metrics_best_qwk_mri_only.json"),
    "primary_best": ("test_metrics_best.json", "test_metrics_best_mri_only.json"),
    "last": ("test_metrics_last.json", "test_metrics_last_mri_only.json"),
}
SELECTION_CHECKPOINT_STEMS = {
    "best_auc": "best_auc",
    "best_balanced_acc": "best_balanced_acc",
    "best_loss": "best_loss",
    "best_qwk": "best_qwk",
    "primary_best": "best",
    "last": "last",
}

DEFAULT_METRICS = ("test_auc", "test_balanced_acc", "test_sens_at_80_spec")
DEFAULT_EXCLUDED_TARGETS = ("ruleout_renal_metabolic_any", )
TARGET_ORDER = (
    "ruleout_age",
    "ruleout_bmi",
    "ruleout_alcohol_encoded",
    "ruleout_smoking_encoded",
    "ruleout_cardio_any",
    "ruleout_respiratory_any",
    "ruleout_dre_abnormal",
    "ruleout_diabetes",
    # "ruleout_renal_metabolic_any",
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
    "ruleout_max_pirads": "PI-RADS score",
    "ruleout_pirads_high": "PI-RADS binary cutoff",
    "ruleout_psa_value": "PSA value",
    "ruleout_scan_prostate_volume_ml": "Prostate volume",
    "ruleout_scan_prostate_volume": "Prostate volume",
    "ruleout_cardio_any": "Cardiovascular disease",
    "ruleout_diabetes": "Diabetes",
    # "ruleout_renal_metabolic_any": "Renal/metabolic condition",
    "ruleout_respiratory_any": "Respiratory disease",
    "ruleout_dre_abnormal": "Abnormal DRE",
}
METRIC_LABELS = {
    "test_auc": "AUROC",
    "test_balanced_acc": "Balanced accuracy",
    "test_sens_at_80_spec": "Sensitivity at 80% specificity",
    "test_acc": "Accuracy",
    "test_sensitivity": "Sensitivity",
    "test_specificity": "Specificity",
    "test_f1": "F1 score",
}
PALETTE = {
    "baseline": "#4D4D4D",
    "improvement": "#0072B2",
    "drop": "#D55E00",
    "neutral": "#6A737B",
    "ci": "#4D4D4D",
    "grid": "#D8D8D8",
    "chance": "#8A8A8A",
    "label_negative": "#6A737B",
    "label_positive": "#D55E00",
}
KEY_DECREASE_TARGETS = {
    "ruleout_psa_value",
    "ruleout_scan_prostate_volume_ml",
    "ruleout_scan_prostate_volume",
}


def require_plotting():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError as exc:
        raise SystemExit(
            "Plotting requires matplotlib. Activate the project environment and install "
            "the plotting requirements, for example: pip install -r code/requirements.txt"
        ) from exc

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#222222",
            "axes.labelcolor": "#222222",
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "xtick.labelsize": 14,
            "ytick.labelsize": 12,
            "legend.fontsize": 14,
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return plt, Line2D


def clinical_label(name):
    if pd.isna(name):
        return ""
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
        "cardio any": "Cardiovascular disease",
    }
    if cleaned in replacements:
        return replacements[cleaned]
    return cleaned.title()


def metric_label(metric):
    return METRIC_LABELS.get(str(metric), str(metric).replace("_", " ").title())


def target_sort_key(target):
    target = str(target)
    if target in TARGET_ORDER:
        return (TARGET_ORDER.index(target), target)
    return (len(TARGET_ORDER), clinical_label(target))


def ordered_targets(targets):
    return sorted(dict.fromkeys(str(target) for target in targets), key=target_sort_key)


def normalize_target_exclusions(values):
    exclusions = set()
    label_to_target = {
        clinical_label(target).casefold(): target
        for target in TARGET_LABELS
    }
    for value in values or []:
        value = str(value).strip()
        if not value:
            continue
        exclusions.add(value)
        exclusions.add(label_to_target.get(value.casefold(), value))
    return exclusions


def significance_stars(value):
    if value is None or pd.isna(value):
        return ""
    value = float(value)
    if value < 0.001:
        return "***"
    if value < 0.01:
        return "**"
    if value < 0.05:
        return "*"
    return ""


def find_results_csv(results_dir, selection, explicit_path=None, include_ordinal=False):
    if explicit_path:
        path = Path(explicit_path)
        if not path.exists():
            raise FileNotFoundError(f"Results CSV not found: {path}")
        return path

    results_dir = Path(results_dir)
    preferred_names = [
        f"ruleout_stats_{selection}_seed_paired_auc_balanced_acc_sens80spec.csv",
        f"ruleout_stats_{selection}_seed_paired_auc_balanced_acc.csv",
    ]
    for preferred_name in preferred_names:
        preferred = results_dir / preferred_name
        if preferred.exists():
            return preferred

    matches = sorted(results_dir.glob(f"ruleout_stats*{selection}*seed_paired*.csv"))
    if not include_ordinal:
        matches = [path for path in matches if "ordinal" not in path.name]
    if not matches:
        raise FileNotFoundError(
            f"No seed-paired results CSV found for selection={selection!r} in {results_dir}"
        )
    return matches[0]


def load_paired_results(args):
    path = find_results_csv(
        args.results_dir,
        args.selection,
        explicit_path=args.results_csv,
        include_ordinal=args.include_ordinal,
    )
    df = pd.read_csv(path)
    required = {
        "selection",
        "baseline",
        "target",
        "metric",
        "baseline_mean",
        "target_mean",
        "mean_delta",
        "analysis_method",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    df = df[df["selection"].astype(str) == args.selection].copy()
    if args.metrics:
        df = df[df["metric"].isin(args.metrics)].copy()
    if args.baseline:
        df = df[df["baseline"].astype(str) == args.baseline].copy()
    if args.model and "model" in df.columns:
        df = df[df["model"].astype(str) == args.model].copy()
    if args.train_mode and "train_mode" in df.columns:
        df = df[df["train_mode"].astype(str) == args.train_mode].copy()
    excluded_targets = normalize_target_exclusions(getattr(args, "exclude_targets", []))
    if excluded_targets:
        df = df[~df["target"].astype(str).isin(excluded_targets)].copy()
    if df.empty:
        raise ValueError(f"No rows remain after filtering {path}")
    return path, df


def save_figure(fig, output_base, dpi=450):
    output_base = Path(output_base)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_base.with_suffix(".png")
    pdf_path = output_base.with_suffix(".pdf")
    fig.savefig(png_path, dpi=dpi)
    fig.savefig(pdf_path)
    return png_path, pdf_path


def task_figure_stem(args, name):
    cohort = getattr(args, "cohort", "internal")
    model = getattr(args, "model", None)
    train_mode = getattr(args, "train_mode", None)
    model_parts = [part for part in (model, train_mode) if part]
    model_tag = "_".join(model_parts)
    if model_tag:
        return f"gleason_{cohort}_{args.task_type}_{model_tag}_{name}"
    return f"gleason_{cohort}_{args.task_type}_{name}"


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # ax.spines["bottom"].set_linewidth(1.)  # x-axis
    # ax.spines["left"].set_linewidth(1.)    # y-axis
    ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.8, alpha=0.7)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="both", length=3, width=1.0)


def forest_xlim(metric_df, padding=1.5):
    ci_low = pd.to_numeric(metric_df["ci_low"], errors="coerce").dropna()
    ci_high = pd.to_numeric(metric_df["ci_high"], errors="coerce").dropna()

    min_ci = float(ci_low.min()) if not ci_low.empty else 0.0
    max_ci = float(ci_high.max()) if not ci_high.empty else 0.0
    left_limit = -max(1.0, math.ceil(abs(min(min_ci, 0.0)) + padding))
    right_limit = max(1.0, math.ceil(max(max_ci, 0.0) + padding))
    return left_limit, right_limit


def plot_forest(args):
    results_path, df = load_paired_results(args)
    df = df[df["analysis_method"].astype(str) == "seed_paired_bootstrap"].copy()
    required = {"ci_low", "ci_high", "p_two_sided"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{results_path} is missing required forest columns: {sorted(missing)}")
    if df.empty:
        raise ValueError(f"No seed_paired_bootstrap rows found in {results_path}")

    metrics = [metric for metric in args.metrics if metric in set(df["metric"])]
    targets = ordered_targets(df["target"])
    if args.dry_run:
        significant = df[(df["ci_low"] > 0.0) | (df["ci_high"] < 0.0)]
        print(f"Forest input: {results_path}")
        print(f"Rows: {len(df)} seed_paired_bootstrap rows")
        print(f"Metrics: {', '.join(metric_label(metric) for metric in metrics)}")
        print(f"Targets: {', '.join(clinical_label(target) for target in targets)}")
        print(f"CI-significant rows: {len(significant)}")
        return

    plt, _ = require_plotting()
    fig_height = max(5.8, 0.42 * len(targets) + 1.8)
    fig, axes = plt.subplots(1, len(metrics), figsize=(6.2 * len(metrics), fig_height), sharey=True)
    axes = np.atleast_1d(axes)

    y_positions = np.arange(len(targets))

    for ax, metric in zip(axes, metrics):
        metric_df = df[df["metric"] == metric].set_index("target")
        for y_pos, target in zip(y_positions, targets):
            if target not in metric_df.index:
                continue
            row = metric_df.loc[target]
            mean_delta = float(row["mean_delta"])
            ci_low = float(row["ci_low"])
            ci_high = float(row["ci_high"])
            color = PALETTE["improvement"] if mean_delta >= 0 else PALETTE["drop"]
            xerr = [[mean_delta - ci_low], [ci_high - mean_delta]]
            ax.errorbar(
                mean_delta,
                y_pos,
                xerr=xerr,
                fmt="o",
                color=color,
                ecolor=PALETTE["ci"],
                elinewidth=2.0,
                capsize=6,
                markersize=8,
                zorder=3,
            )
            significant = (
                ci_low > 0.0 or ci_high < 0.0
            ) and not pd.isna(row.get("p_two_sided"))
            if significant:
                text_x = ci_high + 0.12 if mean_delta >= 0 else ci_low - 0.12
                ha = "left" if mean_delta >= 0 else "right"
                ax.text(
                    text_x,
                    y_pos,
                    significance_stars(row["p_two_sided"]),
                    ha=ha,
                    va="center",
                    fontsize=15,fontweight="bold",
                    color="#000000",
                )

        ax.axvline(0, color="#333333", linewidth=1.0, linestyle="-")
        ax.set_title(metric_label(metric), pad=12,fontsize=16,fontweight="bold",color="#666666")
        
        ax.set_xlim(*forest_xlim(metric_df))
        # ax.set_xlim(-0.5, 0.5)
        # ax.tick_params(axis="both", width=2)
        ax.tick_params(axis="x", labelsize=14, colors="#666666",)
        # for label in ax.get_xticklabels():
        #     label.set_fontweight("bold")
        ax.set_xlabel("Performance change versus baseline")
        ax.set_yticks(y_positions)
        # ax.set_yticklabels([clinical_label(target) for target in targets], fontsize=14)
        ax.set_yticklabels([clinical_label(target) for target in targets],fontsize=18,color="#666666",fontweight="bold")
        ax.invert_yaxis()
        style_axis(ax)

    fig.suptitle(
        "Change in Gleason prediction performance after clinical-information rule-out",
        y=1.02,
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.975,
        "Seed-paired bootstrap mean difference versus MRI-only baseline",
        ha="center",
        va="top",
        fontsize=12,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output_base = Path(args.output_dir) / task_figure_stem(
        args, f"ruleout_forest_{args.selection}"
    )
    png_path, pdf_path = save_figure(fig, output_base, dpi=args.dpi)
    plt.close(fig)
    print(f"Wrote forest plot: {png_path}")
    print(f"Wrote forest plot: {pdf_path}")


def plot_dumbbell(args):
    results_path, df = load_paired_results(args)
    df = df[df["analysis_method"].astype(str) == args.dumbbell_method].copy()
    if df.empty:
        raise ValueError(f"No {args.dumbbell_method} rows found in {results_path}")

    metrics = [metric for metric in args.metrics if metric in set(df["metric"])]
    targets = ordered_targets(df["target"])
    if args.dry_run:
        large_decreases = df[df["mean_delta"] <= -args.large_drop_threshold]
        print(f"Dumbbell input: {results_path}")
        print(f"Rows: {len(df)} {args.dumbbell_method} rows")
        print(f"Metrics: {', '.join(metric_label(metric) for metric in metrics)}")
        print(f"Targets: {', '.join(clinical_label(target) for target in targets)}")
        print(
            "Large decreases: "
            + ", ".join(
                f"{clinical_label(row.target)} {metric_label(row.metric)} {row.mean_delta:+.2f}"
                for row in large_decreases.itertuples()
            )
        )
        return

    plt, Line2D = require_plotting()
    fig_height = max(5.8, 0.42 * len(targets) + 1.8)
    fig, axes = plt.subplots(1, len(metrics), figsize=(6.4 * len(metrics), fig_height), sharey=True)
    axes = np.atleast_1d(axes)
    y_positions = np.arange(len(targets))

    min_perf = float(np.nanmin(df[["baseline_mean", "target_mean"]].to_numpy()))
    max_perf = float(np.nanmax(df[["baseline_mean", "target_mean"]].to_numpy()))
    pad = max(1.0, (max_perf - min_perf) * 0.08)

    for ax, metric in zip(axes, metrics):
        metric_df = df[df["metric"] == metric].set_index("target")
        for y_pos, target in zip(y_positions, targets):
            if target not in metric_df.index:
                continue
            row = metric_df.loc[target]
            baseline = float(row["baseline_mean"])
            target_mean = float(row["target_mean"])
            delta = float(row["mean_delta"])
            is_large_drop = target in KEY_DECREASE_TARGETS and delta < 0
            color = PALETTE["drop"] if is_large_drop or delta <= -args.large_drop_threshold else PALETTE["improvement"]
            line_width = 2.4 if is_large_drop or delta <= -args.large_drop_threshold else 1.6
            ax.plot(
                [baseline, target_mean],
                [y_pos, y_pos],
                color=color,
                linewidth=line_width,
                alpha=0.78,
                zorder=2,
            )
            ax.scatter(
                baseline,
                y_pos,
                color=PALETTE["baseline"],
                edgecolor="white",
                linewidth=0.8,
                s=48,
                zorder=3,
            )
            ax.scatter(
                target_mean,
                y_pos,
                color=color,
                edgecolor="white",
                linewidth=0.8,
                s=56,
                zorder=4,
            )
            if args.annotate_delta:
                text_x = max(baseline, target_mean) + 0.18
                ax.text(
                    text_x,
                    y_pos,
                    f"{delta:+.2f}",
                    ha="left",
                    va="center",
                    fontsize=9.5,
                    color=color,
                )

        ax.set_title(metric_label(metric), pad=12)
        ax.set_xlabel("Performance")
        ax.set_xlim(min_perf - pad, max_perf + pad + (1.2 if args.annotate_delta else 0))
        ax.set_yticks(y_positions)
        ax.set_yticklabels([clinical_label(target) for target in targets])
        ax.invert_yaxis()
        style_axis(ax)

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PALETTE["baseline"], label="Baseline mean"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PALETTE["improvement"], label="Rule-out mean"),
        Line2D([0], [0], color=PALETTE["drop"], linewidth=2.4, label="Highlighted decrease"),
    ]
    axes[-1].legend(handles=legend_handles, loc="lower right", frameon=False)
    fig.suptitle(
        "Baseline versus adversarial rule-out model performance",
        y=1.02,
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.975,
        "Absolute test performance for MRI-only Gleason prediction",
        ha="center",
        va="top",
        fontsize=12,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output_base = Path(args.output_dir) / task_figure_stem(
        args, f"ruleout_dumbbell_{args.selection}"
    )
    png_path, pdf_path = save_figure(fig, output_base, dpi=args.dpi)
    plt.close(fig)
    print(f"Wrote dumbbell plot: {png_path}")
    print(f"Wrote dumbbell plot: {pdf_path}")


def prediction_filename(metric_filename):
    return metric_filename.replace(".json", "_predictions.csv")


def prediction_file_candidates(root, ruleout, model, train_mode, seed, selection):
    base_name, adversarial_name = SELECTION_FILES[selection]
    filenames = (
        [prediction_filename(base_name), prediction_filename(adversarial_name)]
        if ruleout == "ruleout_none"
        else [prediction_filename(adversarial_name), prediction_filename(base_name)]
    )
    run_dir = Path(root) / ruleout / model / train_mode / str(seed)
    return [run_dir / filename for filename in filenames]


def discover_prediction_files(root, model, train_mode, selection, baseline="ruleout_none"):
    root = Path(root)
    files = []
    for ruleout_dir in sorted(root.glob("ruleout_*")):
        run_dir = ruleout_dir / model / train_mode
        if not run_dir.exists():
            continue
        for seed_dir in sorted(run_dir.iterdir(), key=lambda path: int(path.name) if path.name.isdigit() else 10**9):
            if not seed_dir.is_dir() or not seed_dir.name.isdigit():
                continue
            candidates = prediction_file_candidates(
                root, ruleout_dir.name, model, train_mode, seed_dir.name, selection
            )
            for candidate in candidates:
                if candidate.exists():
                    files.append(candidate)
                    break
    if not files and baseline:
        files.extend(sorted(root.glob("*predictions.csv")))
    return files


def derive_condition_from_path(path):
    path = Path(path)
    for part in path.parts:
        if part.startswith("ruleout_"):
            return part
    stem = path.stem
    for selection in SELECTION_FILES:
        stem = stem.replace(f"test_metrics_{selection}_mri_only_predictions", "ruleout_none")
        stem = stem.replace(f"test_metrics_{selection}_predictions", "prediction_file")
    return stem


def derive_seed_from_path(path):
    for part in reversed(Path(path).parts):
        if str(part).isdigit():
            return str(part)
    return "unknown"


def choose_column(columns, preferred):
    for column in preferred:
        if column in columns:
            return column
    return None


def choose_score_column(columns):
    preferred = [
        "binary_score",
        "score",
        "probability",
        "predicted_probability",
        "positive_probability",
        "risk_score",
        "score_1",
        "prob_1",
    ]
    score_col = choose_column(columns, preferred)
    if score_col:
        return score_col, False

    hard_pred = choose_column(
        columns,
        [
            "pred_threshold_default_0_5",
            "pred_default_0_5",
            "prediction",
            "pred",
            "label_pred",
        ],
    )
    if hard_pred:
        return hard_pred, True
    return None, False


def load_prediction_groups(args):
    files = [Path(path) for path in args.prediction_files or []]
    if args.prediction_root:
        files.extend(
            discover_prediction_files(
                args.prediction_root,
                args.model,
                args.train_mode,
                args.selection,
                baseline=args.baseline,
            )
        )
    files = list(dict.fromkeys(files))
    if getattr(args, "seeds", None):
        wanted_seeds = {str(seed) for seed in args.seeds}
        files = [path for path in files if derive_seed_from_path(path) in wanted_seeds]
    if not files:
        raise FileNotFoundError(
            "No prediction CSV files were provided or discovered. Use --prediction_files "
            "or --prediction_root pointing to the Gleason output tree."
        )

    groups = []
    for path in files:
        if not path.exists():
            raise FileNotFoundError(f"Prediction CSV not found: {path}")
        df = pd.read_csv(path)
        condition_col = args.condition_col or choose_column(
            df.columns,
            ["condition", "ruleout", "target", "model_name", "experiment", "adversarial_definition"],
        )
        if condition_col:
            for condition, sub_df in df.groupby(condition_col, dropna=False):
                groups.append((str(condition), derive_seed_from_path(path), path, sub_df.copy()))
        else:
            groups.append((derive_condition_from_path(path), derive_seed_from_path(path), path, df))
    return groups


def prepare_roc_table(groups, args):
    prepared = []
    level_notes = []
    for condition, seed, path, df in groups:
        label_col = args.label_col or choose_column(
            df.columns, ["binary_label", "gleason_binary_cs", "csPCa", "label"]
        )
        score_col = args.score_col
        hard_score = False
        if score_col is None:
            score_col, hard_score = choose_score_column(df.columns)
        if not label_col or not score_col:
            print(
                f"Skipping {path}: could not identify label and score columns "
                f"(label={label_col}, score={score_col})",
                file=sys.stderr,
            )
            continue

        use_cols = [label_col, score_col]
        patient_col = args.patient_col if args.patient_col in df.columns else None
        if args.roc_level == "patient" and patient_col:
            use_cols.append(patient_col)
        work = df[use_cols].copy()
        work[label_col] = pd.to_numeric(work[label_col], errors="coerce")
        work[score_col] = pd.to_numeric(work[score_col], errors="coerce")
        work = work.dropna(subset=[label_col, score_col])
        if work.empty:
            continue
        work[label_col] = (work[label_col].astype(float) > 0).astype(int)

        aggregation_level = "scan"
        if args.roc_level == "patient" and patient_col:
            before = len(work)
            work = (
                work.groupby(patient_col, as_index=False)
                .agg({label_col: "max", score_col: "mean"})
                .reset_index(drop=True)
            )
            aggregation_level = "patient"
            level_notes.append(
                f"{clinical_label(condition)}: aggregated {before} scans to {len(work)} patients by {patient_col}"
            )
        elif args.roc_level == "patient":
            level_notes.append(
                f"{clinical_label(condition)}: no {args.patient_col} column found; ROC is scan-level"
            )

        prepared.append(
            {
                "condition": condition,
                "seed": seed,
                "path": path,
                "label_col": label_col,
                "score_col": score_col,
                "hard_score": hard_score,
                "aggregation_level": aggregation_level,
                "labels": work[label_col].to_numpy(dtype=int),
                "scores": work[score_col].to_numpy(dtype=float),
            }
        )
    for note in level_notes:
        print(note)
    return prepared


def roc_curve_manual(labels, scores):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    if positives == 0 or negatives == 0:
        return None

    order = np.argsort(-scores, kind="mergesort")
    labels = labels[order]
    scores = scores[order]
    distinct = np.where(np.diff(scores))[0]
    threshold_indices = np.r_[distinct, labels.size - 1]
    tps = np.cumsum(labels)[threshold_indices]
    fps = 1 + threshold_indices - tps
    tps = np.r_[0, tps]
    fps = np.r_[0, fps]
    tpr = tps / positives
    fpr = fps / negatives
    if hasattr(np, "trapezoid"):
        auc = float(np.trapezoid(tpr, fpr))
    else:
        auc = float(np.trapz(tpr, fpr))
    return fpr, tpr, auc, positives, negatives


def target_color_map(conditions, plt):
    ordered = ["ruleout_none"] + [target for target in ordered_targets(conditions) if target != "ruleout_none"]
    cmap = plt.get_cmap("tab20")
    colors = {}
    for index, condition in enumerate(ordered):
        colors[condition] = PALETTE["baseline"] if condition == "ruleout_none" else cmap(index % 20)
    return colors


def seed_sort_key(seed):
    return int(seed) if str(seed).isdigit() else 10**9


def aggregate_roc_curves(curves, grid_size=201):
    fpr_grid = np.linspace(0.0, 1.0, int(grid_size))
    aggregated = []
    for condition in ordered_targets([curve[0]["condition"] for curve in curves]):
        condition_curves = [curve for curve in curves if curve[0]["condition"] == condition]
        tpr_values = []
        auc_values = []
        seeds = []
        template = condition_curves[0][0].copy()
        for item, fpr, tpr, auc in condition_curves:
            tpr_interp = np.interp(fpr_grid, np.asarray(fpr, dtype=float), np.asarray(tpr, dtype=float))
            tpr_interp[0] = 0.0
            tpr_interp[-1] = 1.0
            tpr_values.append(tpr_interp)
            auc_values.append(float(auc))
            seeds.append(item["seed"])

        tpr_array = np.vstack(tpr_values)
        template["seed"] = "mean"
        template["seeds"] = sorted(dict.fromkeys(seeds), key=seed_sort_key)
        template["n_seeds"] = len(template["seeds"])
        template["auroc_mean"] = float(np.mean(auc_values))
        template["auroc_sd"] = float(np.std(auc_values, ddof=1)) if len(auc_values) > 1 else 0.0
        template["tpr_sd"] = (
            np.std(tpr_array, axis=0, ddof=1) if len(tpr_values) > 1 else np.zeros_like(fpr_grid)
        )
        aggregated.append((template, fpr_grid, np.mean(tpr_array, axis=0), template["auroc_mean"]))
    return aggregated


def roc_summary_table(roc_df):
    rows = []
    for condition, sub_df in roc_df.groupby("condition", sort=False):
        aucs = pd.to_numeric(sub_df["auroc"], errors="coerce").dropna()
        rows.append(
            {
                "condition": condition,
                "clinical_label": clinical_label(condition),
                "n_seeds": int(sub_df["seed"].nunique()),
                "mean_auroc": float(aucs.mean()) if not aucs.empty else np.nan,
                "sd_auroc": float(aucs.std(ddof=1)) if len(aucs) > 1 else 0.0,
                "aggregation_level": sub_df["aggregation_level"].iloc[0],
                "label_col": sub_df["label_col"].iloc[0],
                "score_col": sub_df["score_col"].iloc[0],
            }
        )
    return pd.DataFrame(rows)


def plot_roc_curves(args):
    groups = load_prediction_groups(args)
    prepared = prepare_roc_table(groups, args)
    if args.seeds:
        wanted_seeds = {str(seed) for seed in args.seeds}
        prepared = [item for item in prepared if str(item["seed"]) in wanted_seeds]
    if not prepared:
        raise ValueError("No usable prediction groups found for ROC plotting")

    roc_rows = []
    curves = []
    for item in prepared:
        curve = roc_curve_manual(item["labels"], item["scores"])
        if curve is None:
            print(f"Skipping {item['condition']}: ROC undefined because one class is absent", file=sys.stderr)
            continue
        fpr, tpr, auc, positives, negatives = curve
        curves.append((item, fpr, tpr, auc))
        roc_rows.append(
            {
                "condition": item["condition"],
                "clinical_label": clinical_label(item["condition"]),
                "seed": item["seed"],
                "auroc": auc,
                "n": int(len(item["labels"])),
                "n_positive": positives,
                "n_negative": negatives,
                "label_col": item["label_col"],
                "score_col": item["score_col"],
                "used_hard_prediction": item["hard_score"],
                "aggregation_level": item["aggregation_level"],
                "source_file": str(item["path"]),
            }
        )
        if item["hard_score"]:
            print(
                f"Warning: {clinical_label(item['condition'])} used hard predictions from "
                f"{item['score_col']} because no continuous score column was found.",
                file=sys.stderr,
            )

    if not curves:
        raise ValueError("No ROC curves could be computed")

    roc_df = pd.DataFrame(roc_rows)
    summary_df = roc_summary_table(roc_df)
    if args.dry_run:
        print(f"ROC seed curves: {len(roc_df)}")
        for row in summary_df.sort_values("condition", key=lambda col: col.map(target_sort_key)).itertuples():
            print(
                f"{row.clinical_label}: mean AUROC={row.mean_auroc:.3f} "
                f"+/- {row.sd_auroc:.3f}, seeds={row.n_seeds}, "
                f"level={row.aggregation_level}, label={row.label_col}, score={row.score_col}"
            )
        return

    plt, _ = require_plotting()
    color_map = target_color_map([row["condition"] for row in roc_rows], plt)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    roc_csv = output_dir / f"roc_auc_values_{args.cohort}_{args.task_type}_per_seed.csv"
    roc_df.sort_values(["condition", "seed"], key=lambda col: col.map(target_sort_key) if col.name == "condition" else col).to_csv(
        roc_csv, index=False
    )
    summary_csv = output_dir / f"roc_auc_values_{args.cohort}_{args.task_type}_seed_mean.csv"
    summary_df.sort_values("condition", key=lambda col: col.map(target_sort_key)).to_csv(
        summary_csv, index=False
    )
    print(f"Wrote per-seed ROC AUROC table: {roc_csv}")
    print(f"Wrote seed-mean ROC AUROC table: {summary_csv}")

    focus_targets = [args.baseline] + list(args.focused_targets or [])
    if args.roc_seed_mode in {"mean", "both"}:
        mean_curves = aggregate_roc_curves(curves, grid_size=args.roc_grid_size)
        draw_roc_figure(
            mean_curves,
            color_map,
            output_dir / f"roc_curves_{args.cohort}_{args.task_type}_seed_mean_all_models",
            args,
            title="Seed-mean ROC curves for MRI-only Gleason prediction",
            subtitle="Baseline versus adversarial clinical-information rule-out models",
            show_seed_band=True,
        )

        focused_curves = [curve for curve in mean_curves if curve[0]["condition"] in set(focus_targets)]
        if len(focused_curves) >= 2 and not args.no_focused_roc:
            draw_roc_figure(
                focused_curves,
                color_map,
                output_dir / f"roc_curves_{args.cohort}_{args.task_type}_seed_mean_focused_key_targets",
                args,
                title="Focused seed-mean ROC curves for MRI-only Gleason prediction",
                subtitle="Baseline compared with PSA, age, and prostate-volume rule-out models",
                show_seed_band=True,
            )

    if args.roc_seed_mode in {"per_seed", "both"}:
        for seed in sorted({curve[0]["seed"] for curve in curves}, key=seed_sort_key):
            seed_curves = [curve for curve in curves if curve[0]["seed"] == seed]
            draw_roc_figure(
                seed_curves,
                color_map,
                output_dir / f"roc_curves_{args.cohort}_{args.task_type}_seed_{seed}_all_models",
                args,
                title=f"Seed {seed} ROC curves for MRI-only Gleason prediction",
                subtitle="Baseline versus adversarial clinical-information rule-out models",
            )
            focused_curves = [curve for curve in seed_curves if curve[0]["condition"] in set(focus_targets)]
            if len(focused_curves) >= 2 and not args.no_focused_roc:
                draw_roc_figure(
                    focused_curves,
                    color_map,
                    output_dir / f"roc_curves_{args.cohort}_{args.task_type}_seed_{seed}_focused_key_targets",
                    args,
                    title=f"Seed {seed} focused ROC curves for MRI-only Gleason prediction",
                    subtitle="Baseline compared with PSA, age, and prostate-volume rule-out models",
                )


def draw_roc_figure(curves, color_map, output_base, args, title, subtitle, show_seed_band=False):
    plt, _ = require_plotting()
    fig, ax = plt.subplots(figsize=(8.2, 8.2))
    ax.plot([0, 1], [0, 1], color=PALETTE["chance"], linestyle="--", linewidth=1.2, label="Chance")
    ordered_curves = sorted(curves, key=lambda curve: target_sort_key(curve[0]["condition"]))
    for item, fpr, tpr, auc in ordered_curves:
        condition = item["condition"]
        color = color_map.get(condition, PALETTE["neutral"])
        if show_seed_band and "tpr_sd" in item:
            lower = np.clip(tpr - item["tpr_sd"], 0.0, 1.0)
            upper = np.clip(tpr + item["tpr_sd"], 0.0, 1.0)
            ax.fill_between(fpr, lower, upper, color=color, alpha=0.13, linewidth=0)
            if item.get("n_seeds", 0) > 1:
                label = (
                    f"{clinical_label(condition)}, AUROC = "
                    f"{item['auroc_mean']:.2f} +/- {item['auroc_sd']:.2f}"
                )
            else:
                label = f"{clinical_label(condition)}, AUROC = {auc:.2f}"
        else:
            seed_suffix = "" if item.get("seed") in {None, "unknown", "mean"} else f", seed {item['seed']}"
            label = f"{clinical_label(condition)}{seed_suffix}, AUROC = {auc:.2f}"
        linewidth = 2.4 if condition == args.baseline else 1.9
        ax.plot(fpr, tpr, color=color, linewidth=linewidth, label=label)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title, fontsize=15, fontweight="bold", pad=30)
    ax.text(
        0.5,
        1.02,
        subtitle,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=11.5,
        color="#444444",
    )
    ax.grid(color=PALETTE["grid"], linewidth=0.8, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=2 if len(curves) <= 8 else 3,
        frameon=False,
    )
    fig.tight_layout()
    png_path, pdf_path = save_figure(fig, output_base, dpi=args.dpi)
    plt.close(fig)
    print(f"Wrote ROC plot: {png_path}")
    print(f"Wrote ROC plot: {pdf_path}")


def discover_ruleout_targets(root, model, train_mode, baseline):
    root = Path(root)
    targets = []
    for ruleout_dir in sorted(root.glob("ruleout_*")):
        if ruleout_dir.name == baseline:
            continue
        if (ruleout_dir / model / train_mode).exists():
            targets.append(ruleout_dir.name)
    return ordered_targets(targets)


def discover_matched_seeds(root, baseline, target, model, train_mode):
    root = Path(root)
    seed_sets = []
    for condition in [baseline, target]:
        run_dir = root / condition / model / train_mode
        if not run_dir.exists():
            return []
        seed_sets.append({path.name for path in run_dir.iterdir() if path.is_dir() and path.name.isdigit()})
    return sorted(set.intersection(*seed_sets), key=seed_sort_key)


def first_existing_prediction(root, condition, model, train_mode, seed, selection):
    for candidate in prediction_file_candidates(root, condition, model, train_mode, seed, selection):
        if candidate.exists():
            return candidate
    return None


def threshold_column_name(name):
    return f"threshold_{name}"


def prediction_column_name(name):
    return f"pred_threshold_{name}"


def prepare_waterfall_predictions(path, args):
    df = pd.read_csv(path)
    label_col = args.label_col or choose_column(
        df.columns, ["binary_label", "gleason_binary_cs", "csPCa", "label"]
    )
    score_col = args.score_col
    hard_score = False
    if score_col is None:
        score_col, hard_score = choose_score_column(df.columns)
    if not label_col or not score_col:
        raise ValueError(
            f"{path} is missing usable label/score columns "
            f"(label={label_col}, score={score_col})"
        )

    key_col = args.patient_col if args.waterfall_level == "patient" else args.scan_id_col
    if key_col not in df.columns:
        if args.waterfall_level == "patient":
            raise ValueError(f"{path} has no patient column {args.patient_col!r}")
        key_col = "row_index" if "row_index" in df.columns else None
    if not key_col:
        raise ValueError(f"{path} has no usable scan identifier column")

    threshold_col = threshold_column_name(args.threshold_name)
    pred_col = prediction_column_name(args.threshold_name)
    use_cols = [key_col, label_col, score_col]
    if threshold_col in df.columns:
        use_cols.append(threshold_col)
    if pred_col in df.columns:
        use_cols.append(pred_col)

    work = df[use_cols].copy()
    work[label_col] = pd.to_numeric(work[label_col], errors="coerce")
    work[score_col] = pd.to_numeric(work[score_col], errors="coerce")
    if threshold_col in work.columns:
        work[threshold_col] = pd.to_numeric(work[threshold_col], errors="coerce")
    if pred_col in work.columns:
        work[pred_col] = pd.to_numeric(work[pred_col], errors="coerce")
    work = work.dropna(subset=[key_col, label_col, score_col])
    if work.empty:
        raise ValueError(f"{path} has no usable prediction rows")
    work[label_col] = (work[label_col].astype(float) > 0).astype(int)

    threshold_value = args.threshold_value
    if threshold_value is None and threshold_col in work.columns:
        thresholds = work[threshold_col].dropna()
        if not thresholds.empty:
            threshold_value = float(thresholds.iloc[0])
    if threshold_value is None:
        threshold_value = 0.5

    if args.waterfall_level == "patient":
        work = (
            work.groupby(key_col, as_index=False)
            .agg({label_col: "max", score_col: "mean"})
            .reset_index(drop=True)
        )
        work[pred_col] = (work[score_col] >= threshold_value).astype(int)
    elif pred_col not in work.columns:
        work[pred_col] = (work[score_col] >= threshold_value).astype(int)
    else:
        work[pred_col] = work[pred_col].fillna(work[score_col] >= threshold_value).astype(int)

    return work.rename(
        columns={
            key_col: "case_id",
            label_col: "label",
            score_col: "score",
            pred_col: "prediction",
        }
    )[["case_id", "label", "score", "prediction"]]


def crossing_type(label, baseline_pred, ruleout_pred):
    label = int(label)
    baseline_pred = int(baseline_pred)
    ruleout_pred = int(ruleout_pred)
    if baseline_pred == 1 and ruleout_pred == 0:
        return "corrected_fp" if label == 0 else "lost_tp"
    if baseline_pred == 0 and ruleout_pred == 1:
        return "corrected_fn" if label == 1 else "new_fp"
    return "no_crossing"


def matched_waterfall_seed_table(root, target, seed, args):
    baseline_path = first_existing_prediction(
        root, args.baseline, args.model, args.train_mode, seed, args.selection
    )
    target_path = first_existing_prediction(
        root, target, args.model, args.train_mode, seed, args.selection
    )
    if baseline_path is None or target_path is None:
        missing = []
        if baseline_path is None:
            missing.append(args.baseline)
        if target_path is None:
            missing.append(target)
        print(f"Skipping seed {seed} for {target}: missing {', '.join(missing)} predictions", file=sys.stderr)
        return None

    baseline = prepare_waterfall_predictions(baseline_path, args)
    ruleout = prepare_waterfall_predictions(target_path, args)
    merged = baseline.merge(ruleout, on="case_id", how="inner", suffixes=("_baseline", "_ruleout"))
    if merged.empty:
        print(f"Skipping seed {seed} for {target}: no matched cases", file=sys.stderr)
        return None
    label_mismatch = merged["label_baseline"] != merged["label_ruleout"]
    if label_mismatch.any():
        print(
            f"Warning: {target} seed {seed} has {int(label_mismatch.sum())} label mismatches; "
            "using the baseline label.",
            file=sys.stderr,
        )
    merged = merged.rename(
        columns={
            "label_baseline": "label",
            "score_baseline": "baseline_probability",
            "score_ruleout": "ruleout_probability",
            "prediction_baseline": "baseline_prediction",
            "prediction_ruleout": "ruleout_prediction",
        }
    )
    merged["seed"] = seed
    merged["target"] = target
    merged["shift"] = merged["baseline_probability"] - merged["ruleout_probability"]
    merged["crossing_type"] = [
        crossing_type(row.label, row.baseline_prediction, row.ruleout_prediction)
        for row in merged.itertuples()
    ]
    return merged[
        [
            "target",
            "seed",
            "case_id",
            "label",
            "baseline_probability",
            "ruleout_probability",
            "shift",
            "baseline_prediction",
            "ruleout_prediction",
            "crossing_type",
        ]
    ]


def aggregate_waterfall_table(seed_df):
    crossing_levels = ["corrected_fp", "lost_tp", "corrected_fn", "new_fp"]
    rows = []
    for case_id, sub_df in seed_df.groupby("case_id", sort=False):
        crossing_counts = sub_df["crossing_type"].value_counts().to_dict()
        n_seeds = len(sub_df)
        majority_crossing = "no_crossing"
        for level in crossing_levels:
            if crossing_counts.get(level, 0) > n_seeds / 2.0:
                majority_crossing = level
                break
        row = {
            "case_id": case_id,
            "label": int(sub_df["label"].max()),
            "baseline_probability": float(sub_df["baseline_probability"].mean()),
            "ruleout_probability": float(sub_df["ruleout_probability"].mean()),
            "shift": float(sub_df["shift"].mean()),
            "shift_sd": float(sub_df["shift"].std(ddof=1)) if n_seeds > 1 else 0.0,
            "n_seeds": int(n_seeds),
            "crossing_type": majority_crossing,
            "any_crossing_seed_count": int((sub_df["crossing_type"] != "no_crossing").sum()),
        }
        for level in crossing_levels:
            row[f"{level}_seed_count"] = int(crossing_counts.get(level, 0))
        rows.append(row)
    return pd.DataFrame(rows)


def sort_waterfall_table(df, args):
    ascending = args.waterfall_sort == "ascending"
    return df.sort_values("shift", ascending=ascending).reset_index(drop=True)


def draw_waterfall_figure(df, output_base, target, args, seed=None):
    plt, Line2D = require_plotting()
    df = sort_waterfall_table(df, args)
    if args.max_cases:
        df = df.head(args.max_cases).copy()
    n_cases = len(df)
    if n_cases == 0:
        raise ValueError(f"No waterfall rows to plot for {target}")

    width = min(24.0, max(9.0, 7.0 + 0.035 * n_cases))
    fig, ax = plt.subplots(figsize=(width, 6.4))
    x = np.arange(n_cases)
    colors = np.where(df["label"].astype(int).to_numpy() > 0, PALETTE["label_positive"], PALETTE["label_negative"])
    yerr = df["shift_sd"].to_numpy(dtype=float) if args.show_seed_sd and "shift_sd" in df.columns else None
    ax.bar(x, df["shift"].to_numpy(dtype=float), color=colors, edgecolor="white", linewidth=0.2, width=0.86, yerr=yerr)

    marker_specs = {
        "corrected_fp": ("v", "#000000", "Corrected false positive"),
        "lost_tp": ("x", "#000000", "Lost true positive"),
        "corrected_fn": ("^", "#000000", "Corrected false negative"),
        "new_fp": ("o", "#000000", "New false positive"),
    }
    y_values = df["shift"].to_numpy(dtype=float)
    y_range = max(0.05, float(np.nanmax(y_values) - np.nanmin(y_values)))
    offset = y_range * 0.035
    for crossing, (marker, color, _) in marker_specs.items():
        mask = df["crossing_type"].astype(str).to_numpy() == crossing
        if not mask.any():
            continue
        marker_y = y_values[mask] + np.where(y_values[mask] >= 0.0, offset, -offset)
        ax.scatter(x[mask], marker_y, marker=marker, color=color, s=34, linewidths=1.2, zorder=4)

    ax.axhline(0, color="#222222", linewidth=1.0)
    ax.set_xlim(-0.7, n_cases - 0.3)
    ax.set_xlabel("Test patients sorted by probability shift" if args.waterfall_level == "patient" else "Test scans sorted by probability shift")
    ax.set_ylabel("Baseline probability - rule-out probability")
    seed_text = "seed-mean" if seed is None else f"seed {seed}"
    ax.set_title(
        f"{clinical_label(target)} rule-out probability-shift waterfall ({seed_text})",
        fontsize=15,
        fontweight="bold",
        pad=16,
    )
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.8, alpha=0.7)
    ax.grid(axis="x", visible=False)
    ax.set_xticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_handles = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor=PALETTE["label_negative"], label="True label 0"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=PALETTE["label_positive"], label="True label 1"),
    ]
    for crossing, (marker, color, label) in marker_specs.items():
        if (df["crossing_type"].astype(str) == crossing).any():
            legend_handles.append(Line2D([0], [0], marker=marker, color=color, linestyle="None", label=label))
    ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=3, frameon=False)
    fig.tight_layout()
    png_path, pdf_path = save_figure(fig, output_base, dpi=args.dpi)
    plt.close(fig)
    print(f"Wrote waterfall plot: {png_path}")
    print(f"Wrote waterfall plot: {pdf_path}")


def plot_waterfall(args):
    if not args.prediction_root:
        raise ValueError("waterfall requires --prediction_root pointing to the Gleason output tree")
    targets = ordered_targets(args.targets) if args.targets else discover_ruleout_targets(
        args.prediction_root, args.model, args.train_mode, args.baseline
    )
    if not targets:
        raise ValueError("No rule-out targets found for waterfall plotting")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for target in targets:
        seeds = args.seeds or discover_matched_seeds(
            args.prediction_root, args.baseline, target, args.model, args.train_mode
        )
        seeds = [str(seed) for seed in seeds]
        if not seeds:
            print(f"Skipping {target}: no matched seeds found", file=sys.stderr)
            continue
        seed_tables = [
            table
            for table in (
                matched_waterfall_seed_table(args.prediction_root, target, seed, args)
                for seed in seeds
            )
            if table is not None
        ]
        if not seed_tables:
            print(f"Skipping {target}: no usable matched prediction tables", file=sys.stderr)
            continue
        seed_df = pd.concat(seed_tables, ignore_index=True)
        aggregate_df = aggregate_waterfall_table(seed_df)
        aggregate_df = sort_waterfall_table(aggregate_df, args)

        if args.dry_run:
            crossing_counts = aggregate_df["crossing_type"].value_counts().to_dict()
            print(
                f"{clinical_label(target)}: cases={len(aggregate_df)}, seeds={len(seeds)}, "
                f"crossings={crossing_counts}"
            )
            continue

        seed_csv = output_dir / f"waterfall_values_{args.cohort}_{args.task_type}_{target}_{args.selection}_per_seed.csv"
        aggregate_csv = output_dir / f"waterfall_values_{args.cohort}_{args.task_type}_{target}_{args.selection}_seed_mean.csv"
        seed_df.to_csv(seed_csv, index=False)
        aggregate_df.to_csv(aggregate_csv, index=False)
        print(f"Wrote per-seed waterfall table: {seed_csv}")
        print(f"Wrote seed-mean waterfall table: {aggregate_csv}")

        if args.waterfall_seed_mode in {"mean", "both"}:
            draw_waterfall_figure(
                aggregate_df,
                output_dir / task_figure_stem(args, f"{target}_waterfall_{args.selection}_seed_mean"),
                target,
                args,
            )
        if args.waterfall_seed_mode in {"per_seed", "both"}:
            for seed, sub_df in seed_df.groupby("seed", sort=False):
                draw_waterfall_figure(
                    sub_df,
                    output_dir / task_figure_stem(args, f"{target}_waterfall_{args.selection}_seed_{seed}"),
                    target,
                    args,
                    seed=seed,
                )


def require_umap():
    try:
        import umap
    except ImportError as exc:
        raise SystemExit(
            "UMAP plotting requires umap-learn. Install it in the active environment, "
            "for example: pip install umap-learn"
        ) from exc
    return umap


def checkpoint_path_for_selection(root, condition, model, train_mode, seed, selection):
    stem = SELECTION_CHECKPOINT_STEMS[selection]
    return Path(root) / condition / model / train_mode / str(seed) / f"{stem}.pth.tar"


def state_dict_without_adversarial_heads(state_dict):
    return {
        key: value
        for key, value in state_dict.items()
        if not key.startswith("adversarial_heads.")
    }


def umap_metadata_columns(dataset, extra_columns=None):
    preferred = [
        "person_id",
        "new_id",
        "pseudo_study_uid",
        "procedure_date",
        "image_npy_path",
        "grade_group",
        "gleason_binary_cs",
        "csPCa",
        "psa_value",
        "psa_group",
        "psa_group_code",
        "age",
        "age_group",
        "age_group_code",
        "bmi",
        "bmi_group",
        "bmi_group_code",
        "cardio_any",
        "diabetes",
        "diabetes_observed",
        # "renal_metabolic_any",
        "respiratory_any",
        "dre_abnormal",
        "max_pirads",
        "pirads_high",
        "scan_prostate_volume_ml",
    ]
    for column in extra_columns or []:
        if column not in preferred:
            preferred.append(column)
    df = getattr(dataset, "df", None)
    if df is None:
        return []
    return [column for column in preferred if column in df.columns]


def clean_metadata_value(value):
    try:
        if value != value:
            return ""
    except TypeError:
        pass
    return value


def make_umap_model_args(args):
    model_args = argparse.Namespace(
        batch_size=args.batch_size,
        epochs=0,
        device=args.device,
        seed=0,
        num_workers=args.num_workers,
        pin_mem=not args.no_pin_mem,
        drop_last=False,
        data_root=args.data_root,
        csv_path=args.csv_path,
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        test_csv=args.test_csv,
        split_col=args.split_col,
        image_path_col=args.image_path_col,
        in_channels=args.in_channels,
        crop_spatial_size=args.crop_spatial_size,
        task_type=args.task_type,
        label_col=args.label_col,
        binary_label_col=args.binary_label_col,
        binary_positive_min=args.binary_positive_min,
        ordinal_levels=args.ordinal_levels,
        label_offset=args.label_offset,
        weighted_sampling=False,
        adversarial_specs="",
        adversarial_variable=None,
        adversarial_column=None,
        adversarial_observed_column=None,
        adversarial_num_classes=None,
        adversarial_loss_weight=1.0,
        grl_lambda=1.0,
        grl_schedule="constant",
        grl_gamma=10.0,
        model=args.model,
        train=args.train_mode,
        pretrain=args.pretrain,
        resume="",
        bottleneck_dim=args.bottleneck_dim,
        weight_decay=args.weight_decay,
        lr=args.lr,
        min_lr=0.0,
        warmup_epochs=0,
        layer_decay=args.layer_decay,
        layer_decay_type=args.layer_decay_type,
        output_dir="",
        log_dir="",
        start_epoch=0,
        val_interval=1,
        save_ckpt_interval=10,
        primary_metric=None,
        world_size=1,
        local_rank=-1,
        dist_on_itp=False,
        dist_url="env://",
    )
    return model_args


def build_umap_test_loader(args):
    from dataset.gleason_cls import build_gleason_classification_loaders

    model_args = make_umap_model_args(args)
    _, _, test_loader = build_gleason_classification_loaders(model_args)
    return model_args, test_loader


def extract_condition_embeddings(args, model_args, data_loader, condition, seed):
    import torch

    from engine.gleason_classification import (
        binary_scores_for_grade_threshold,
        decode_predictions,
        unpack_batch,
    )
    from models.build_gleason_classification import build_gleason_model
    from util.misc import load_trusted_checkpoint

    checkpoint_path = checkpoint_path_for_selection(
        args.prediction_root,
        condition,
        args.model,
        args.train_mode,
        seed,
        args.selection,
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device(args.device)
    torch.manual_seed(int(seed) if str(seed).isdigit() else 0)
    model, _ = build_gleason_model(args=model_args, device=device)
    checkpoint = load_trusted_checkpoint(checkpoint_path, map_location=device)
    state_dict = state_dict_without_adversarial_heads(checkpoint["model"])
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    rows = []
    features_all = []
    dataset = data_loader.dataset
    metadata_columns = umap_metadata_columns(dataset, args.color_cols)

    with torch.no_grad():
        for batch in data_loader:
            img, labels, _, _, dataidx = unpack_batch(batch)
            img = img.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            features = model.forward_features(img)
            logits = model.main_head(features)
            batch_features = features.detach().cpu().numpy()
            batch_labels = labels.detach().cpu().numpy().astype(int)
            batch_indices = dataidx.detach().cpu().numpy().astype(int)
            preds, scores = decode_predictions(logits.detach().cpu(), model_args)

            binary_labels, binary_scores = binary_scores_for_grade_threshold(
                batch_labels,
                scores,
                model_args,
            )
            if binary_scores is None:
                binary_scores = np.full(len(batch_labels), np.nan)

            for row_position, sample_index in enumerate(batch_indices):
                source_row = dataset.df.iloc[int(sample_index)]
                binary_label = int(binary_labels[row_position])
                binary_score = float(binary_scores[row_position])
                pred_binary = int(binary_score >= 0.5) if np.isfinite(binary_score) else int(preds[row_position])
                if pred_binary == 1 and binary_label == 1:
                    error_type = "true_positive"
                elif pred_binary == 0 and binary_label == 0:
                    error_type = "true_negative"
                elif pred_binary == 1 and binary_label == 0:
                    error_type = "false_positive"
                else:
                    error_type = "false_negative"
                row = {
                    "condition": condition,
                    "condition_label": clinical_label(condition),
                    "seed": seed,
                    "row_index": int(sample_index),
                    "label": int(batch_labels[row_position]),
                    "binary_label": binary_label,
                    "binary_score": binary_score,
                    "pred_default_0_5": pred_binary,
                    "error_type": error_type,
                    "checkpoint": str(checkpoint_path),
                }
                for column in metadata_columns:
                    row[column] = clean_metadata_value(source_row[column])
                rows.append(row)
                features_all.append(batch_features[row_position])

    if not rows:
        raise ValueError(f"No embeddings extracted for {condition} seed {seed}")
    return pd.DataFrame(rows), np.vstack(features_all)


def scale_embeddings(features):
    from sklearn.preprocessing import StandardScaler

    return StandardScaler().fit_transform(np.asarray(features, dtype=float))


def fit_umap_coordinates(features, args, random_state):
    umap = require_umap()
    reducer = umap.UMAP(
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        metric=args.umap_metric,
        random_state=random_state,
    )
    return reducer.fit_transform(scale_embeddings(features))


def color_is_categorical(series, max_categories=12):
    non_missing = series.dropna()
    if non_missing.empty:
        return True
    if not pd.api.types.is_numeric_dtype(non_missing):
        return True
    return int(non_missing.nunique(dropna=True)) <= max_categories


def draw_umap_figure(embedding_df, target, color_col, output_base, args, seed_label):
    plt, _ = require_plotting()
    conditions = [args.baseline, target]
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.8), sharex=True, sharey=True)
    if color_col not in embedding_df.columns:
        print(f"Skipping UMAP color column {color_col!r}: column not found", file=sys.stderr)
        plt.close(fig)
        return

    color_values = embedding_df[color_col]
    categorical = color_is_categorical(color_values)
    if categorical:
        categories = sorted(color_values.fillna("missing").astype(str).unique())
        cmap = plt.get_cmap("tab10" if len(categories) <= 10 else "tab20")
        color_map = {
            category: cmap(index % cmap.N)
            for index, category in enumerate(categories)
        }
    else:
        numeric_values = pd.to_numeric(color_values, errors="coerce")
        vmin = float(numeric_values.min())
        vmax = float(numeric_values.max())

    for ax, condition in zip(axes, conditions):
        sub_df = embedding_df[embedding_df["condition"] == condition]
        if categorical:
            colors = [
                color_map[str(value) if not pd.isna(value) else "missing"]
                for value in sub_df[color_col]
            ]
            ax.scatter(
                sub_df["umap_x"],
                sub_df["umap_y"],
                c=colors,
                s=args.umap_point_size,
                alpha=args.umap_alpha,
                linewidths=0,
            )
        else:
            scatter = ax.scatter(
                sub_df["umap_x"],
                sub_df["umap_y"],
                c=pd.to_numeric(sub_df[color_col], errors="coerce"),
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                s=args.umap_point_size,
                alpha=args.umap_alpha,
                linewidths=0,
            )
        ax.set_title(clinical_label(condition), fontsize=13, fontweight="bold")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(color=PALETTE["grid"], linewidth=0.7, alpha=0.45)

    if categorical:
        from matplotlib.lines import Line2D

        handles = [
            Line2D([0], [0], marker="o", color="none", markerfacecolor=color, label=category)
            for category, color in color_map.items()
        ]
        fig.legend(handles=handles, loc="lower center", ncol=min(5, len(handles)), frameon=False)
        bottom = 0.18
    else:
        cbar = fig.colorbar(scatter, ax=axes.ravel().tolist(), shrink=0.78)
        cbar.set_label(color_col.replace("_", " "))
        bottom = 0.08

    fig.suptitle(
        f"Joint UMAP of baseline and {clinical_label(target)} embeddings ({seed_label})",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.925,
        f"Colored by {color_col.replace('_', ' ')}",
        ha="center",
        va="center",
        fontsize=11,
        color="#444444",
    )
    fig.tight_layout(rect=(0, bottom, 1, 0.90))
    png_path, pdf_path = save_figure(fig, output_base, dpi=args.dpi)
    plt.close(fig)
    print(f"Wrote UMAP plot: {png_path}")
    print(f"Wrote UMAP plot: {pdf_path}")


def run_umap_for_seed(args, target, seed, data_loader=None, model_args=None):
    if data_loader is None or model_args is None:
        model_args, data_loader = build_umap_test_loader(args)
    baseline_df, baseline_features = extract_condition_embeddings(
        args, model_args, data_loader, args.baseline, seed
    )
    target_df, target_features = extract_condition_embeddings(
        args, model_args, data_loader, target, seed
    )
    embedding_df = pd.concat([baseline_df, target_df], ignore_index=True)
    features = np.vstack([baseline_features, target_features])
    coordinates = fit_umap_coordinates(features, args, random_state=int(seed) if str(seed).isdigit() else args.umap_random_state)
    embedding_df["umap_x"] = coordinates[:, 0]
    embedding_df["umap_y"] = coordinates[:, 1]
    return embedding_df


def plot_umap(args):
    targets = ordered_targets(args.targets) if args.targets else discover_ruleout_targets(
        args.prediction_root, args.model, args.train_mode, args.baseline
    )
    if not targets:
        raise ValueError("No rule-out targets found for UMAP plotting")

    output_dir = Path(args.figure_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        for target in targets:
            seeds = args.seeds or discover_matched_seeds(
                args.prediction_root, args.baseline, target, args.model, args.train_mode
            )
            missing = []
            for seed in seeds:
                for condition in [args.baseline, target]:
                    checkpoint_path = checkpoint_path_for_selection(
                        args.prediction_root,
                        condition,
                        args.model,
                        args.train_mode,
                        seed,
                        args.selection,
                    )
                    if not checkpoint_path.exists():
                        missing.append(str(checkpoint_path))
            print(
                f"{clinical_label(target)}: seeds={', '.join(map(str, seeds))}, "
                f"missing_checkpoints={len(missing)}"
            )
            for path in missing[:5]:
                print(f"  missing: {path}")
        return

    require_umap()
    model_args, data_loader = build_umap_test_loader(args)

    for target in targets:
        seeds = args.seeds or discover_matched_seeds(
            args.prediction_root, args.baseline, target, args.model, args.train_mode
        )
        seeds = [str(seed) for seed in seeds]
        if not seeds:
            print(f"Skipping {target}: no matched seeds found", file=sys.stderr)
            continue

        if args.umap_seed_mode == "per_seed":
            for seed in seeds:
                embedding_df = run_umap_for_seed(
                    args, target, seed, data_loader=data_loader, model_args=model_args
                )
                csv_path = output_dir / f"umap_embeddings_{args.cohort}_{args.task_type}_{target}_{args.selection}_seed_{seed}.csv"
                embedding_df.to_csv(csv_path, index=False)
                print(f"Wrote UMAP embedding table: {csv_path}")
                if args.dry_run:
                    continue
                for color_col in args.color_cols:
                    output_base = output_dir / task_figure_stem(
                        args,
                        f"{target}_umap_{args.selection}_seed_{seed}_color_{color_col}",
                    )
                    draw_umap_figure(embedding_df, target, color_col, output_base, args, f"seed {seed}")
            continue

        frames = []
        feature_blocks = []
        for seed in seeds:
            baseline_df, baseline_features = extract_condition_embeddings(
                args, model_args, data_loader, args.baseline, seed
            )
            target_df, target_features = extract_condition_embeddings(
                args, model_args, data_loader, target, seed
            )
            frames.extend([baseline_df, target_df])
            feature_blocks.extend([baseline_features, target_features])
        embedding_df = pd.concat(frames, ignore_index=True)
        coordinates = fit_umap_coordinates(
            np.vstack(feature_blocks),
            args,
            random_state=args.umap_random_state,
        )
        embedding_df["umap_x"] = coordinates[:, 0]
        embedding_df["umap_y"] = coordinates[:, 1]
        csv_path = output_dir / f"umap_embeddings_{args.cohort}_{args.task_type}_{target}_{args.selection}_pooled_seeds.csv"
        embedding_df.to_csv(csv_path, index=False)
        print(f"Wrote pooled UMAP embedding table: {csv_path}")
        for color_col in args.color_cols:
            output_base = output_dir / task_figure_stem(
                args,
                f"{target}_umap_{args.selection}_pooled_seeds_color_{color_col}",
            )
            draw_umap_figure(embedding_df, target, color_col, output_base, args, "pooled seeds")


def add_shared_result_args(parser):
    parser.add_argument("--results_dir", type=Path, default=Path("log_hpc/results"))
    parser.add_argument("--results_csv", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=Path("log_hpc/figures"))
    parser.add_argument("--selection", choices=sorted(SELECTION_FILES), default="best_auc")
    parser.add_argument(
        "--task_type",
        choices=["binary", "ordinal"],
        default="binary",
        help="Task label used in saved figure filenames.",
    )
    parser.add_argument(
        "--cohort",
        default="internal",
        help="Cohort/dataset label used in saved figure filenames, e.g. internal or promis.",
    )
    parser.add_argument("--baseline", default="ruleout_none")
    parser.add_argument("--model", default="profound_conv")
    parser.add_argument("--train_mode", default="fintune")
    parser.add_argument("--metrics", nargs="+", default=list(DEFAULT_METRICS))
    parser.add_argument(
        "--exclude_targets",
        nargs="*",
        default=list(DEFAULT_EXCLUDED_TARGETS),
        help=(
            "Rule-out targets to exclude from result plots. Accepts internal "
            "names such as ruleout_renal_metabolic_any or ruleout_diabetes, "
            "or labels such as 'Renal/metabolic condition'. Pass "
            "--exclude_targets with no values "
            "to include all targets."
        ),
    )
    parser.add_argument("--include_ordinal", action="store_true")
    parser.add_argument("--dpi", type=int, default=450)
    parser.add_argument("--dry_run", action="store_true", help="Validate inputs without rendering figures.")


def add_roc_args(parser):
    parser.add_argument("--prediction_files", nargs="+", default=None)
    parser.add_argument(
        "--prediction_root",
        type=Path,
        default=None,
        help="Optional output tree containing ruleout_* prediction CSVs.",
    )
    parser.add_argument("--output_dir", type=Path, default=Path("log_hpc/figures"))
    parser.add_argument("--selection", choices=sorted(SELECTION_FILES), default="best_auc")
    parser.add_argument(
        "--task_type",
        choices=["binary", "ordinal"],
        default="binary",
        help="Task label used in saved figure filenames.",
    )
    parser.add_argument(
        "--cohort",
        default="internal",
        help="Cohort/dataset label used in saved figure filenames, e.g. internal or promis.",
    )
    parser.add_argument("--baseline", default="ruleout_none")
    parser.add_argument("--seeds", nargs="+", default=None)
    parser.add_argument("--model", default="profound_conv")
    parser.add_argument("--train_mode", default="fintune")
    parser.add_argument("--label_col", default=None)
    parser.add_argument("--score_col", default=None)
    parser.add_argument("--condition_col", default=None)
    parser.add_argument("--patient_col", default="person_id")
    parser.add_argument("--roc_level", choices=["patient", "scan"], default="patient")
    parser.add_argument(
        "--roc_seed_mode",
        choices=["mean", "per_seed", "both"],
        default="mean",
        help="Plot seed-mean ROC curves by default, or one clean ROC figure per seed.",
    )
    parser.add_argument("--roc_grid_size", type=int, default=201)
    parser.add_argument(
        "--focused_targets",
        nargs="+",
        default=["ruleout_psa_value", "ruleout_age", "ruleout_scan_prostate_volume_ml"],
    )
    parser.add_argument("--no_focused_roc", action="store_true")
    parser.add_argument("--dpi", type=int, default=450)
    parser.add_argument("--dry_run", action="store_true", help="Validate inputs without rendering figures.")


def add_waterfall_args(parser):
    parser.add_argument(
        "--prediction_root",
        type=Path,
        required=True,
        help="Output tree containing ruleout_* prediction CSVs.",
    )
    parser.add_argument("--output_dir", type=Path, default=Path("log_hpc/figures"))
    parser.add_argument("--selection", choices=sorted(SELECTION_FILES), default="best_auc")
    parser.add_argument(
        "--task_type",
        choices=["binary", "ordinal"],
        default="binary",
        help="Task label used in saved figure filenames.",
    )
    parser.add_argument(
        "--cohort",
        default="internal",
        help="Cohort/dataset label used in saved figure filenames, e.g. internal or promis.",
    )
    parser.add_argument("--baseline", default="ruleout_none")
    parser.add_argument("--targets", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", default=None)
    parser.add_argument("--model", default="profound_conv")
    parser.add_argument("--train_mode", default="fintune")
    parser.add_argument("--label_col", default=None)
    parser.add_argument("--score_col", default=None)
    parser.add_argument("--patient_col", default="person_id")
    parser.add_argument("--scan_id_col", default="row_index")
    parser.add_argument("--waterfall_level", choices=["patient", "scan"], default="patient")
    parser.add_argument(
        "--waterfall_seed_mode",
        choices=["mean", "per_seed", "both"],
        default="mean",
        help="Plot seed-mean waterfall by default, or one clean waterfall per seed.",
    )
    parser.add_argument("--threshold_name", default="default_0_5")
    parser.add_argument("--threshold_value", type=float, default=None)
    parser.add_argument("--waterfall_sort", choices=["descending", "ascending"], default="descending")
    parser.add_argument("--max_cases", type=int, default=None)
    parser.add_argument("--show_seed_sd", action="store_true")
    parser.add_argument("--dpi", type=int, default=450)
    parser.add_argument("--dry_run", action="store_true", help="Validate inputs without rendering figures.")


def add_umap_args(parser):
    parser.add_argument(
        "--prediction_root",
        type=Path,
        required=True,
        help="Output tree containing ruleout_* checkpoint folders.",
    )
    parser.add_argument("--figure_output_dir", type=Path, default=Path("log_hpc/figures"))
    parser.add_argument("--selection", choices=sorted(SELECTION_FILES), default="best_auc")
    parser.add_argument(
        "--task_type",
        choices=["binary", "ordinal"],
        required=True,
        help="Gleason task used by the checkpoint.",
    )
    parser.add_argument("--cohort", default="internal")
    parser.add_argument("--baseline", default="ruleout_none")
    parser.add_argument("--targets", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", default=None)

    parser.add_argument("--data_root", default="data", type=str)
    parser.add_argument("--csv_path", default="data/gleason_classification.csv", type=str)
    parser.add_argument("--train_csv", default=None, type=str)
    parser.add_argument("--val_csv", default=None, type=str)
    parser.add_argument("--test_csv", default=None, type=str)
    parser.add_argument("--split_col", default="split", type=str)
    parser.add_argument("--image_path_col", default="image_npy_path", type=str)
    parser.add_argument("--in_channels", default=3, type=int)
    parser.add_argument("--crop_spatial_size", default="64,256,256")
    parser.add_argument("--label_col", default="grade_group", type=str)
    parser.add_argument("--binary_label_col", default=None, type=str)
    parser.add_argument("--binary_positive_min", default=2, type=int)
    parser.add_argument("--ordinal_levels", default=5, type=int)
    parser.add_argument("--label_offset", default=1, type=int)

    parser.add_argument("--model", choices=["resnet18", "profound_conv", "profound_vit"], required=True)
    parser.add_argument("--train_mode", choices=["fintune", "freeze", "scratch"], default="fintune")
    parser.add_argument("--pretrain", default=None, type=str)
    parser.add_argument("--bottleneck_dim", default=256, type=int)
    parser.add_argument("--weight_decay", default=1e-5, type=float)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--layer_decay", default=0.6, type=float)
    parser.add_argument("--layer_decay_type", choices=["single", "group"], default="group")
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--no_pin_mem", action="store_true")

    parser.add_argument(
        "--umap_seed_mode",
        choices=["per_seed", "pooled"],
        default="per_seed",
        help=(
            "Default is per_seed because embeddings from separate model seeds are "
            "not linearly aligned. pooled fits one qualitative UMAP over all seeds."
        ),
    )
    parser.add_argument("--color_cols", nargs="+", default=["binary_label", "psa_group"])
    parser.add_argument("--umap_neighbors", type=int, default=30)
    parser.add_argument("--umap_min_dist", type=float, default=0.1)
    parser.add_argument("--umap_metric", default="euclidean")
    parser.add_argument("--umap_random_state", type=int, default=0)
    parser.add_argument("--umap_point_size", type=float, default=18.0)
    parser.add_argument("--umap_alpha", type=float, default=0.78)
    parser.add_argument("--dpi", type=int, default=450)
    parser.add_argument("--dry_run", action="store_true", help="Validate targets/seeds/checkpoints without extraction.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Publication-quality plots for Gleason clinical-information rule-out experiments."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    forest = subparsers.add_parser("forest", help="Plot seed-paired bootstrap forest plots.")
    add_shared_result_args(forest)
    forest.set_defaults(func=plot_forest)

    dumbbell = subparsers.add_parser("dumbbell", help="Plot baseline versus rule-out dumbbell plots.")
    add_shared_result_args(dumbbell)
    dumbbell.add_argument("--dumbbell_method", default="seed_paired_bootstrap")
    dumbbell.add_argument("--large_drop_threshold", type=float, default=1.5)
    dumbbell.add_argument("--annotate_delta", action=argparse.BooleanOptionalAction, default=True)
    dumbbell.set_defaults(func=plot_dumbbell)

    roc = subparsers.add_parser("roc", help="Plot ROC curves from per-patient/per-scan prediction CSVs.")
    add_roc_args(roc)
    roc.set_defaults(func=plot_roc_curves)

    waterfall = subparsers.add_parser(
        "waterfall",
        help="Plot baseline minus rule-out probability shifts from paired prediction CSVs.",
    )
    add_waterfall_args(waterfall)
    waterfall.set_defaults(func=plot_waterfall)

    umap = subparsers.add_parser(
        "umap",
        help="Extract checkpoint embeddings and plot joint baseline-vs-ruleout UMAP panels.",
    )
    add_umap_args(umap)
    umap.set_defaults(func=plot_umap)
    return parser.parse_args()


def main():
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
