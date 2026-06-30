import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


SELECTION_STEMS = {
    "best_auc": "best_auc",
    "best_balanced_acc": "best_balanced_acc",
    "best_loss": "best_loss",
    "best_qwk": "best_qwk",
    "primary_best": "best",
    "last": "last",
}

DEFAULT_METRICS = (
    "test_auc",
    "test_balanced_acc",
    "test_acc",
    "test_sensitivity",
    "test_specificity",
    "test_f1",
    "test_loss",
)


def weight_tag(value):
    return str(value).replace(".", "p")


def weight_from_dir(path):
    value = path.name.removeprefix("loss_weight_")
    try:
        return float(value.replace("p", "."))
    except ValueError:
        return value


def sortable_weight(value):
    try:
        return (0, float(value))
    except (TypeError, ValueError):
        return (1, str(value))


def metric_filename(selection, variant):
    stem = SELECTION_STEMS[selection]
    if variant == "mri_only":
        return f"test_metrics_{stem}_mri_only.json"
    return f"test_metrics_{stem}.json"


def discover_weight_dirs(root):
    dirs = []
    for path in root.glob("loss_weight_*"):
        if path.is_dir():
            dirs.append((weight_from_dir(path), path))
    return sorted(dirs, key=lambda item: sortable_weight(item[0]))


def discover_ruleouts(weight_dirs, task_type, label_definition):
    ruleouts = set()
    for _, weight_dir in weight_dirs:
        base = weight_dir / "gleason" / task_type / label_definition
        for path in base.glob("ruleout_*"):
            if path.is_dir():
                ruleouts.add(path.name)
    return sorted(ruleouts)


def discover_seeds(weight_dirs, task_type, label_definition, ruleouts, model, train_mode):
    seeds = set()
    for _, weight_dir in weight_dirs:
        for ruleout in ruleouts:
            run_root = (
                weight_dir
                / "gleason"
                / task_type
                / label_definition
                / ruleout
                / model
                / train_mode
            )
            for path in run_root.iterdir() if run_root.exists() else []:
                if path.is_dir() and path.name.isdigit():
                    seeds.add(path.name)
    return sorted(seeds, key=int)


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def as_float(value):
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result):
        return None
    return result


def run_dir(weight_dir, task_type, label_definition, ruleout, model, train_mode, seed):
    return (
        weight_dir
        / "gleason"
        / task_type
        / label_definition
        / ruleout
        / model
        / train_mode
        / str(seed)
    )


def collect_rows(args, weight_dirs, ruleouts, seeds):
    rows = []
    missing_rows = []
    filename = metric_filename(args.selection, args.variant)

    for loss_weight, weight_dir in weight_dirs:
        for ruleout in ruleouts:
            for seed in seeds:
                path = (
                    run_dir(
                        weight_dir,
                        args.task_type,
                        args.label_definition,
                        ruleout,
                        args.model,
                        args.train_mode,
                        seed,
                    )
                    / filename
                )
                if not path.exists():
                    missing_rows.append(
                        {
                            "loss_weight": loss_weight,
                            "ruleout": ruleout,
                            "seed": seed,
                            "selection": args.selection,
                            "variant": args.variant,
                            "metric_file": str(path),
                        }
                    )
                    continue

                metrics = read_json(path)
                for metric in args.metrics:
                    rows.append(
                        {
                            "loss_weight": loss_weight,
                            "ruleout": ruleout,
                            "seed": seed,
                            "selection": args.selection,
                            "variant": args.variant,
                            "metric": metric,
                            "value": as_float(metrics.get(metric)),
                            "metric_file": str(path),
                            "checkpoint": metrics.get("checkpoint", ""),
                            "label_definition": metrics.get("label_definition", ""),
                            "checkpoint_adversarial_definition": metrics.get(
                                "checkpoint_adversarial_definition", ""
                            ),
                            "evaluation_adversarial_definition": metrics.get(
                                "evaluation_adversarial_definition", ""
                            ),
                            "drop_adversarial_head": metrics.get(
                                "drop_adversarial_head", ""
                            ),
                        }
                    )

    return rows, missing_rows


def summarize_rows(rows, all_seeds):
    grouped = defaultdict(list)
    for row in rows:
        value = row["value"]
        if value is None:
            continue
        key = (
            row["loss_weight"],
            row["ruleout"],
            row["selection"],
            row["variant"],
            row["metric"],
        )
        grouped[key].append((str(row["seed"]), value))

    summary = []
    for key, seed_values in sorted(grouped.items(), key=lambda item: (
        sortable_weight(item[0][0]),
        item[0][1],
        item[0][4],
    )):
        loss_weight, ruleout, selection, variant, metric = key
        seed_values = sorted(seed_values, key=lambda item: int(item[0]))
        values = [value for _, value in seed_values]
        observed_seeds = [seed for seed, _ in seed_values]
        missing_seeds = [seed for seed in all_seeds if seed not in observed_seeds]
        n = len(values)
        sd = stdev(values) if n > 1 else None
        sem = sd / math.sqrt(n) if sd is not None else None
        summary.append(
            {
                "loss_weight": loss_weight,
                "ruleout": ruleout,
                "selection": selection,
                "variant": variant,
                "metric": metric,
                "n": n,
                "mean": mean(values),
                "sd": sd,
                "sem": sem,
                "min": min(values),
                "max": max(values),
                "seeds": " ".join(observed_seeds),
                "missing_seeds": " ".join(missing_seeds),
                "values_by_seed": ";".join(
                    f"{seed}:{value:.6g}" for seed, value in seed_values
                ),
            }
        )
    return summary


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_best(summary, primary_metric):
    rows = [row for row in summary if row["metric"] == primary_metric]
    if not rows:
        return
    rows = sorted(rows, key=lambda row: (row["mean"], -float(row["loss_weight"])), reverse=True)
    print(f"\nBest by {primary_metric}:")
    for row in rows[:5]:
        sd_text = "" if row["sd"] is None else f" +/- {row['sd']:.4g}"
        print(
            f"  loss_weight={row['loss_weight']} "
            f"{row['ruleout']} mean={row['mean']:.4g}{sd_text} n={row['n']}"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize Gleason adversarial-strength sweep test metrics."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("output_cls/adversarial_strength_sweep"),
        help="Sweep root containing loss_weight_* directories.",
    )
    parser.add_argument(
        "--weights",
        nargs="*",
        default=None,
        help="Loss weights to read, e.g. 0.1 0.5 2.0. Defaults to discovery.",
    )
    parser.add_argument("--seeds", nargs="*", default=None)
    parser.add_argument("--ruleouts", nargs="*", default=None)
    parser.add_argument("--task_type", default="binary")
    parser.add_argument("--label_definition", default="grade_group_ge_2")
    parser.add_argument("--model", default="profound_conv")
    parser.add_argument("--train_mode", default="fintune")
    parser.add_argument(
        "--selection",
        default="best_auc",
        choices=sorted(SELECTION_STEMS),
        help="Checkpoint selection metric file to summarize.",
    )
    parser.add_argument(
        "--variant",
        default="mri_only",
        choices=("mri_only", "with_adversarial_head"),
        help="Use *_mri_only metrics or metrics with the adversarial head present.",
    )
    parser.add_argument("--metrics", nargs="+", default=list(DEFAULT_METRICS))
    parser.add_argument("--primary_metric", default="test_auc")
    parser.add_argument(
        "--long_csv",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--summary_csv",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--missing_csv",
        type=Path,
        default=None,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_stem = f"adversarial_strength_sweep_{args.selection}_{args.variant}"
    args.long_csv = args.long_csv or Path("results") / f"{output_stem}_long.csv"
    args.summary_csv = args.summary_csv or Path("results") / f"{output_stem}_summary.csv"
    args.missing_csv = args.missing_csv or Path("results") / f"{output_stem}_missing.csv"

    if args.weights:
        weight_dirs = [
            (weight, args.root / f"loss_weight_{weight_tag(weight)}")
            for weight in args.weights
        ]
    else:
        weight_dirs = discover_weight_dirs(args.root)
    if not weight_dirs:
        raise FileNotFoundError(f"No loss_weight_* directories found under {args.root}")

    ruleouts = args.ruleouts or discover_ruleouts(
        weight_dirs, args.task_type, args.label_definition
    )
    if not ruleouts:
        raise FileNotFoundError("No ruleout_* directories found in the sweep output")

    seeds = args.seeds or discover_seeds(
        weight_dirs,
        args.task_type,
        args.label_definition,
        ruleouts,
        args.model,
        args.train_mode,
    )
    if not seeds:
        raise FileNotFoundError("No seed directories found in the sweep output")

    rows, missing_rows = collect_rows(args, weight_dirs, ruleouts, seeds)
    summary = summarize_rows(rows, [str(seed) for seed in seeds])

    long_fields = [
        "loss_weight",
        "ruleout",
        "seed",
        "selection",
        "variant",
        "metric",
        "value",
        "metric_file",
        "checkpoint",
        "label_definition",
        "checkpoint_adversarial_definition",
        "evaluation_adversarial_definition",
        "drop_adversarial_head",
    ]
    summary_fields = [
        "loss_weight",
        "ruleout",
        "selection",
        "variant",
        "metric",
        "n",
        "mean",
        "sd",
        "sem",
        "min",
        "max",
        "seeds",
        "missing_seeds",
        "values_by_seed",
    ]
    missing_fields = [
        "loss_weight",
        "ruleout",
        "seed",
        "selection",
        "variant",
        "metric_file",
    ]

    write_csv(args.long_csv, rows, long_fields)
    write_csv(args.summary_csv, summary, summary_fields)
    write_csv(args.missing_csv, missing_rows, missing_fields)

    print(f"Read weights: {' '.join(str(weight) for weight, _ in weight_dirs)}")
    print(f"Read ruleouts: {' '.join(ruleouts)}")
    print(f"Read seeds: {' '.join(str(seed) for seed in seeds)}")
    print(f"Wrote long table: {args.long_csv}")
    print(f"Wrote summary table: {args.summary_csv}")
    print(f"Wrote missing-file table: {args.missing_csv}")
    print_best(summary, args.primary_metric)


if __name__ == "__main__":
    main()
