import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path


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

SELECTION_STEMS = {
    "best_auc": "best_auc",
    "best_balanced_acc": "best_balanced_acc",
    "best_loss": "best_loss",
    "best_qwk": "best_qwk",
    "primary_best": "best",
    "last": "last",
}

DEFAULT_METRICS = ("test_auc", "test_balanced_acc", "test_sens_at_80_spec")
PER_SEED_PRIMARY_METRICS = ("test_auc", "test_balanced_acc", "test_sens_at_80_spec")


def regularized_beta(x, a, b):
    """Regularized incomplete beta I_x(a, b), using a continued fraction."""
    if x < 0.0 or x > 1.0:
        raise ValueError("x must be in [0, 1]")
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0

    max_iter = 200
    eps = 3.0e-14
    fpmin = 1.0e-300

    def beta_fraction(x_value, a_value, b_value):
        qab = a_value + b_value
        qap = a_value + 1.0
        qam = a_value - 1.0
        c = 1.0
        d = 1.0 - qab * x_value / qap
        if abs(d) < fpmin:
            d = fpmin
        d = 1.0 / d
        h = d

        for m in range(1, max_iter + 1):
            m2 = 2 * m
            aa = m * (b_value - m) * x_value / (
                (qam + m2) * (a_value + m2)
            )
            d = 1.0 + aa * d
            if abs(d) < fpmin:
                d = fpmin
            c = 1.0 + aa / c
            if abs(c) < fpmin:
                c = fpmin
            d = 1.0 / d
            h *= d * c

            aa = -(
                (a_value + m)
                * (qab + m)
                * x_value
                / ((a_value + m2) * (qap + m2))
            )
            d = 1.0 + aa * d
            if abs(d) < fpmin:
                d = fpmin
            c = 1.0 + aa / c
            if abs(c) < fpmin:
                c = fpmin
            d = 1.0 / d
            delta = d * c
            h *= delta
            if abs(delta - 1.0) <= eps:
                break
        return h

    log_beta_term = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    beta_term = math.exp(log_beta_term)

    if x < (a + 1.0) / (a + b + 2.0):
        return beta_term * beta_fraction(x, a, b) / a
    return 1.0 - beta_term * beta_fraction(1.0 - x, b, a) / b


def student_t_cdf(t_value, df):
    if df <= 0:
        return float("nan")
    if t_value == 0:
        return 0.5
    x = df / (df + t_value * t_value)
    ib = regularized_beta(x, df / 2.0, 0.5)
    if t_value > 0:
        return 1.0 - 0.5 * ib
    return 0.5 * ib


def student_t_ppf(probability, df):
    if not 0.0 < probability < 1.0:
        raise ValueError("probability must be in (0, 1)")
    if probability == 0.5:
        return 0.0
    sign = 1.0
    p = probability
    if probability < 0.5:
        sign = -1.0
        p = 1.0 - probability

    lo, hi = 0.0, 1.0
    while student_t_cdf(hi, df) < p:
        hi *= 2.0
        if hi > 1.0e6:
            break
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if student_t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return sign * (lo + hi) / 2.0


def paired_ttest(differences, confidence=0.95):
    n = len(differences)
    if n < 2:
        return {
            "analysis_method": "not_run_single_seed",
            "statistic_type": None,
            "statistic": None,
            "n": n,
            "mean_delta": differences[0] if n == 1 else None,
            "sd_delta": None,
            "sem_delta": None,
            "t_stat": None,
            "df": None,
            "p_two_sided": None,
            "ci_low": None,
            "ci_high": None,
            "note": "paired t-test requires at least two matched seeds",
        }

    mean_delta = sum(differences) / n
    variance = sum((value - mean_delta) ** 2 for value in differences) / (n - 1)
    sd_delta = math.sqrt(variance)
    sem_delta = sd_delta / math.sqrt(n)
    df = n - 1

    if sem_delta == 0.0:
        t_stat = math.inf if mean_delta > 0 else -math.inf if mean_delta < 0 else 0.0
        p_two_sided = 0.0 if mean_delta != 0 else 1.0
        ci_low = mean_delta
        ci_high = mean_delta
    else:
        t_stat = mean_delta / sem_delta
        tail = 1.0 - student_t_cdf(abs(t_stat), df)
        p_two_sided = min(1.0, max(0.0, 2.0 * tail))
        alpha = 1.0 - confidence
        critical = student_t_ppf(1.0 - alpha / 2.0, df)
        ci_low = mean_delta - critical * sem_delta
        ci_high = mean_delta + critical * sem_delta

    return {
        "analysis_method": "paired_seed_ttest",
        "statistic_type": "t",
        "statistic": t_stat,
        "n": n,
        "mean_delta": mean_delta,
        "sd_delta": sd_delta,
        "sem_delta": sem_delta,
        "t_stat": t_stat,
        "df": df,
        "p_two_sided": p_two_sided,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "note": "",
    }


def paired_seed_bootstrap(
    differences,
    confidence=0.95,
    n_bootstraps=10000,
    random_seed=0,
):
    n = len(differences)
    if n < 2:
        return {
            "analysis_method": "seed_paired_bootstrap_not_run",
            "statistic_type": None,
            "statistic": None,
            "n": n,
            "mean_delta": differences[0] if n == 1 else None,
            "sd_delta": None,
            "sem_delta": None,
            "diff_se": None,
            "baseline_se": None,
            "target_se": None,
            "t_stat": None,
            "df": None,
            "p_two_sided": None,
            "ci_low": None,
            "ci_high": None,
            "note": "seed-paired bootstrap requires at least two matched seeds",
        }

    observed_delta = sum(differences) / n
    rng = random.Random(random_seed)
    deltas = []
    for _ in range(n_bootstraps):
        sample = [rng.choice(differences) for _ in range(n)]
        deltas.append(sum(sample) / n)

    deltas_sorted = sorted(deltas)
    alpha = 1.0 - confidence
    sd_delta = math.sqrt(
        sum((delta - sum(deltas) / len(deltas)) ** 2 for delta in deltas)
        / (len(deltas) - 1)
    ) if len(deltas) > 1 else 0.0
    p_lower = (1 + sum(1 for delta in deltas if delta <= 0.0)) / (len(deltas) + 1)
    p_upper = (1 + sum(1 for delta in deltas if delta >= 0.0)) / (len(deltas) + 1)
    p_two_sided = min(1.0, 2.0 * min(p_lower, p_upper))

    return {
        "analysis_method": "seed_paired_bootstrap",
        "statistic_type": "bootstrap_mean_delta",
        "statistic": observed_delta,
        "n": n,
        "mean_delta": observed_delta,
        "sd_delta": sd_delta,
        "sem_delta": None,
        "diff_se": sd_delta,
        "baseline_se": None,
        "target_se": None,
        "t_stat": None,
        "df": None,
        "p_two_sided": p_two_sided,
        "ci_low": percentile(deltas_sorted, alpha / 2.0),
        "ci_high": percentile(deltas_sorted, 1.0 - alpha / 2.0),
        "note": f"seed bootstrap iterations={len(deltas)}",
    }


def baseline_seed_bootstrap(
    values,
    confidence=0.95,
    n_bootstraps=10000,
    random_seed=0,
):
    n = len(values)
    observed_mean = sum(values) / n if values else None
    if n < 2:
        return {
            "analysis_method": "baseline_seed_bootstrap_not_run",
            "statistic_type": None,
            "statistic": observed_mean,
            "n": n,
            "mean_delta": 0.0 if n == 1 else None,
            "sd_delta": None,
            "sem_delta": None,
            "diff_se": None,
            "baseline_se": None,
            "target_se": None,
            "t_stat": None,
            "df": None,
            "p_two_sided": None,
            "ci_low": None,
            "ci_high": None,
            "note": "baseline CI requires at least two baseline seeds",
        }

    rng = random.Random(random_seed)
    means = []
    for _ in range(n_bootstraps):
        sample = [rng.choice(values) for _ in range(n)]
        means.append(sum(sample) / n)

    means_sorted = sorted(means)
    alpha = 1.0 - confidence
    value_variance = sum((value - observed_mean) ** 2 for value in values) / (n - 1)
    sd_value = math.sqrt(value_variance)
    sd_mean = math.sqrt(
        sum((mean - sum(means) / len(means)) ** 2 for mean in means)
        / (len(means) - 1)
    ) if len(means) > 1 else 0.0

    return {
        "analysis_method": "baseline_seed_bootstrap",
        "statistic_type": "bootstrap_mean",
        "statistic": observed_mean,
        "n": n,
        "mean_delta": 0.0,
        "sd_delta": sd_value,
        "sem_delta": sd_value / math.sqrt(n),
        "diff_se": None,
        "baseline_se": sd_mean,
        "target_se": None,
        "t_stat": None,
        "df": None,
        "p_two_sided": None,
        "ci_low": percentile(means_sorted, alpha / 2.0),
        "ci_high": percentile(means_sorted, 1.0 - alpha / 2.0),
        "note": f"baseline mean CI; seed bootstrap iterations={len(means)}",
    }


def read_metric_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def seed_dirs(run_dir):
    dirs = []
    for path in run_dir.iterdir() if run_dir.exists() else []:
        if path.is_dir() and path.name.isdigit():
            dirs.append(path)
    return sorted(dirs, key=lambda item: int(item.name))


def selection_files(selection, prefix):
    if prefix == "test_metrics":
        return SELECTION_FILES[selection]
    stem = SELECTION_STEMS[selection]
    return f"{prefix}_{stem}.json", f"{prefix}_{stem}_mri_only.json"


def metric_file_for_seed(
    root, ruleout, model, train_mode, seed, selection, metric_prefix="test_metrics"
):
    for path in metric_file_candidates(
        root, ruleout, model, train_mode, seed, selection, metric_prefix
    ):
        if path.exists():
            return path
    return metric_file_candidates(
        root, ruleout, model, train_mode, seed, selection, metric_prefix
    )[0]


def metric_file_candidates(
    root, ruleout, model, train_mode, seed, selection, metric_prefix="test_metrics"
):
    base_name, adversarial_name = selection_files(selection, metric_prefix)
    if ruleout == "ruleout_none":
        filenames = [base_name, adversarial_name]
    else:
        filenames = [adversarial_name, base_name]
    run_dir = root / ruleout / model / train_mode / str(seed)
    return [run_dir / filename for filename in filenames]


def prediction_filename(metric_filename):
    return metric_filename.replace(".json", "_predictions.csv")


def prediction_files_for_selection(selection, metric_prefix, prediction_prefix):
    if prediction_prefix:
        stem = SELECTION_STEMS[selection]
        return (
            f"{prediction_prefix}_{stem}.csv",
            f"{prediction_prefix}_{stem}_mri_only.csv",
        )
    base_name, adversarial_name = selection_files(selection, metric_prefix)
    return prediction_filename(base_name), prediction_filename(adversarial_name)


def prediction_file_candidates(
    root,
    ruleout,
    model,
    train_mode,
    seed,
    selection,
    metric_prefix="test_metrics",
    prediction_prefix=None,
):
    base_name, adversarial_name = prediction_files_for_selection(
        selection, metric_prefix, prediction_prefix
    )
    if ruleout == "ruleout_none":
        filenames = [
            prediction_filename(base_name),
            prediction_filename(adversarial_name),
        ]
    else:
        filenames = [
            prediction_filename(adversarial_name),
            prediction_filename(base_name),
        ]
    run_dir = root / ruleout / model / train_mode / str(seed)
    return [run_dir / filename for filename in filenames]


def prediction_file_for_seed(
    root,
    ruleout,
    model,
    train_mode,
    seed,
    selection,
    metric_prefix="test_metrics",
    prediction_prefix=None,
):
    candidates = prediction_file_candidates(
        root,
        ruleout,
        model,
        train_mode,
        seed,
        selection,
        metric_prefix,
        prediction_prefix,
    )
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def discover_ruleouts(root, include_baseline=False):
    names = []
    for path in sorted(root.glob("ruleout_*")):
        if not path.is_dir():
            continue
        if path.name == "ruleout_none" and not include_baseline:
            continue
        names.append(path.name)
    return names


def discover_matched_seeds(
    root, baseline, target, model, train_mode, selection, metric_prefix="test_metrics"
):
    baseline_dir = root / baseline / model / train_mode
    target_dir = root / target / model / train_mode
    baseline_seeds = {path.name for path in seed_dirs(baseline_dir)}
    target_seeds = {path.name for path in seed_dirs(target_dir)}
    seeds = sorted(baseline_seeds & target_seeds, key=int)

    matched = []
    for seed in seeds:
        baseline_file = metric_file_for_seed(
            root, baseline, model, train_mode, seed, selection, metric_prefix
        )
        target_file = metric_file_for_seed(
            root, target, model, train_mode, seed, selection, metric_prefix
        )
        if baseline_file.exists() and target_file.exists():
            matched.append(seed)
    return matched


def discover_metric_seeds(
    root, ruleout, model, train_mode, selection, metric_prefix="test_metrics"
):
    run_dir = root / ruleout / model / train_mode
    matched = []
    for seed_dir in seed_dirs(run_dir):
        seed = seed_dir.name
        metric_file = metric_file_for_seed(
            root, ruleout, model, train_mode, seed, selection, metric_prefix
        )
        if metric_file.exists():
            matched.append(seed)
    return matched


def numeric_test_metrics(data):
    return {
        key: value
        for key, value in data.items()
        if key.startswith("test_")
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
    }


def ordered_per_seed_metrics(rows):
    metrics = sorted(
        {
            key
            for row in rows
            for key in row
            if key.startswith("test_")
            and not key.startswith("baseline_")
            and not key.startswith("delta_")
        }
    )
    primary = [metric for metric in PER_SEED_PRIMARY_METRICS if metric in metrics]
    return primary + [metric for metric in metrics if metric not in primary]


def per_seed_columns(rows, table_mode):
    metrics = ordered_per_seed_metrics(rows)
    if table_mode != "full":
        return (
            ["target", "seed"]
            + metrics
            + [f"delta_{metric}" for metric in metrics]
            + ["note"]
        )
    return (
        [
            "selection",
            "baseline",
            "target",
            "model",
            "train_mode",
            "seed",
            "metric_file",
            "baseline_metric_file",
        ]
        + metrics
        + [f"baseline_{metric}" for metric in metrics]
        + [f"delta_{metric}" for metric in metrics]
        + ["note"]
    )


def per_seed_targets(args):
    if args.targets:
        targets = [args.baseline] + list(args.targets)
    else:
        targets = [args.baseline] + discover_ruleouts(args.root)

    seen = set()
    ordered_targets = []
    for target in targets:
        if target in seen:
            continue
        seen.add(target)
        ordered_targets.append(target)
    return ordered_targets


def requested_per_seed_metrics(args, target_metrics, baseline_metrics):
    available_metrics = set(target_metrics) | set(baseline_metrics)
    if args.metrics == ["all"]:
        return sorted(available_metrics)
    return [metric for metric in args.metrics if metric in available_metrics]


def no_matched_seed_note(root, baseline, target, model, train_mode, selection, metric_prefix):
    baseline_dir = root / baseline / model / train_mode
    target_dir = root / target / model / train_mode
    baseline_seeds = {path.name for path in seed_dirs(baseline_dir)}
    target_seeds = {path.name for path in seed_dirs(target_dir)}
    shared_seeds = sorted(baseline_seeds & target_seeds, key=int)

    notes = []
    if not baseline_dir.exists():
        notes.append(f"baseline seed directory missing: {baseline_dir}")
    if not target_dir.exists():
        notes.append(f"target seed directory missing: {target_dir}")
    if baseline_dir.exists() and target_dir.exists() and not shared_seeds:
        notes.append(
            f"no shared seed directories; baseline_seeds={','.join(sorted(baseline_seeds, key=int)) or 'none'}; "
            f"target_seeds={','.join(sorted(target_seeds, key=int)) or 'none'}"
        )
    if shared_seeds:
        seed = shared_seeds[0]
        baseline_candidates = metric_file_candidates(
            root, baseline, model, train_mode, seed, selection, metric_prefix
        )
        target_candidates = metric_file_candidates(
            root, target, model, train_mode, seed, selection, metric_prefix
        )
        if not any(path.exists() for path in baseline_candidates):
            notes.append(
                "baseline metric missing for shared seed "
                f"{seed}: expected one of "
                + " or ".join(str(path) for path in baseline_candidates)
            )
        if not any(path.exists() for path in target_candidates):
            notes.append(
                "target metric missing for shared seed "
                f"{seed}: expected one of "
                + " or ".join(str(path) for path in target_candidates)
            )
    if not notes:
        notes.append("no matched metric files found")
    return "; ".join(notes)


def discover_matched_prediction_seeds(
    root,
    baseline,
    target,
    model,
    train_mode,
    selection,
    metric_prefix="test_metrics",
    prediction_prefix=None,
):
    baseline_dir = root / baseline / model / train_mode
    target_dir = root / target / model / train_mode
    baseline_seeds = {path.name for path in seed_dirs(baseline_dir)}
    target_seeds = {path.name for path in seed_dirs(target_dir)}
    seeds = sorted(baseline_seeds & target_seeds, key=int)

    matched = []
    for seed in seeds:
        baseline_file = prediction_file_for_seed(
            root,
            baseline,
            model,
            train_mode,
            seed,
            selection,
            metric_prefix,
            prediction_prefix,
        )
        target_file = prediction_file_for_seed(
            root,
            target,
            model,
            train_mode,
            seed,
            selection,
            metric_prefix,
            prediction_prefix,
        )
        if baseline_file.exists() and target_file.exists():
            matched.append(seed)
    return matched


def discover_prediction_seeds(
    root,
    ruleout,
    model,
    train_mode,
    selection,
    metric_prefix="test_metrics",
    prediction_prefix=None,
):
    run_dir = root / ruleout / model / train_mode
    matched = []
    for seed_dir in seed_dirs(run_dir):
        seed = seed_dir.name
        prediction_file = prediction_file_for_seed(
            root,
            ruleout,
            model,
            train_mode,
            seed,
            selection,
            metric_prefix,
            prediction_prefix,
        )
        if prediction_file.exists():
            matched.append(seed)
    return matched


def read_prediction_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def first_available(row, columns):
    for column in columns:
        value = row.get(column)
        if value not in (None, ""):
            return value
    return None


def as_int(value):
    return int(float(value))


def as_float(value):
    return float(value)


def auc_from_scores(labels, scores):
    positives = sum(1 for label in labels if label == 1)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None

    ranked = sorted(zip(scores, labels), key=lambda item: item[0])
    rank_sum_positive = 0.0
    rank = 1
    index = 0
    while index < len(ranked):
        end = index + 1
        while end < len(ranked) and ranked[end][0] == ranked[index][0]:
            end += 1
        average_rank = (rank + rank + (end - index) - 1) / 2.0
        rank_sum_positive += average_rank * sum(
            1 for _, label in ranked[index:end] if label == 1
        )
        rank += end - index
        index = end

    auc = (
        rank_sum_positive - positives * (positives + 1) / 2.0
    ) / (positives * negatives)
    return auc * 100.0


def sensitivity_at_min_specificity(labels, scores, min_specificity=0.80):
    positives = sum(1 for label in labels if label == 1)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None

    best_sensitivity = None
    thresholds = sorted(set(scores), reverse=True)
    for threshold in [math.inf, *thresholds, -math.inf]:
        tp = fp = fn = tn = 0
        for label, score in zip(labels, scores):
            pred = 1 if score >= threshold else 0
            if label == 1 and pred == 1:
                tp += 1
            elif label == 1 and pred == 0:
                fn += 1
            elif label == 0 and pred == 1:
                fp += 1
            elif label == 0 and pred == 0:
                tn += 1
        specificity = tn / negatives
        if specificity >= min_specificity:
            sensitivity = tp / positives
            if best_sensitivity is None or sensitivity > best_sensitivity:
                best_sensitivity = sensitivity
    return 100.0 * best_sensitivity if best_sensitivity is not None else None


def binary_metrics(labels, scores, preds, metric):
    if not labels:
        return None
    if metric == "test_auc":
        return auc_from_scores(labels, scores)
    if metric == "test_sens_at_80_spec":
        return sensitivity_at_min_specificity(labels, scores, min_specificity=0.80)

    tn = fp = fn = tp = 0
    for label, pred in zip(labels, preds):
        if label == 1 and pred == 1:
            tp += 1
        elif label == 1 and pred == 0:
            fn += 1
        elif label == 0 and pred == 1:
            fp += 1
        elif label == 0 and pred == 0:
            tn += 1

    total = tn + fp + fn + tp
    sensitivity = tp / (tp + fn) if tp + fn > 0 else None
    specificity = tn / (tn + fp) if tn + fp > 0 else None
    precision = tp / (tp + fp) if tp + fp > 0 else None
    recall = sensitivity

    if metric == "test_acc":
        return 100.0 * (tp + tn) / total if total > 0 else None
    if metric == "test_sensitivity":
        return 100.0 * sensitivity if sensitivity is not None else None
    if metric == "test_specificity":
        return 100.0 * specificity if specificity is not None else None
    if metric == "test_balanced_acc":
        if sensitivity is None or specificity is None:
            return None
        return 100.0 * (sensitivity + specificity) / 2.0
    if metric == "test_f1":
        if precision is None or recall is None or precision + recall == 0.0:
            return None
        return 100.0 * (2.0 * precision * recall / (precision + recall))
    return None


def prediction_records_from_rows(
    baseline_rows,
    target_rows,
    pair_key,
    cluster_key,
    prediction_column,
    seed,
):
    target_by_key = {}
    for row in target_rows:
        key = first_available(row, [pair_key, "image_npy_path", "pseudo_study_uid", "new_id", "row_index"])
        if key is not None:
            target_by_key[key] = row

    records = []
    mismatched_labels = 0
    for baseline_row in baseline_rows:
        key = first_available(
            baseline_row,
            [pair_key, "image_npy_path", "pseudo_study_uid", "new_id", "row_index"],
        )
        if key is None or key not in target_by_key:
            continue
        target_row = target_by_key[key]

        baseline_label_value = first_available(baseline_row, ["binary_label", "label"])
        target_label_value = first_available(target_row, ["binary_label", "label"])
        if baseline_label_value is None or target_label_value is None:
            continue
        baseline_label = as_int(baseline_label_value)
        target_label = as_int(target_label_value)
        if baseline_label != target_label:
            mismatched_labels += 1
            continue

        baseline_score_value = first_available(baseline_row, ["binary_score", "score"])
        target_score_value = first_available(target_row, ["binary_score", "score"])
        baseline_pred_value = first_available(
            baseline_row,
            [prediction_column, "pred_threshold_default_0_5", "pred_default_0_5"],
        )
        target_pred_value = first_available(
            target_row,
            [prediction_column, "pred_threshold_default_0_5", "pred_default_0_5"],
        )
        if (
            baseline_score_value is None
            or target_score_value is None
            or baseline_pred_value is None
            or target_pred_value is None
        ):
            continue

        cluster = first_available(
            baseline_row,
            [cluster_key, "person_id", pair_key, "image_npy_path", "row_index"],
        )
        records.append(
            {
                "seed": str(seed),
                "pair_key": key,
                "cluster": f"{seed}:{cluster}",
                "label": baseline_label,
                "baseline_score": as_float(baseline_score_value),
                "target_score": as_float(target_score_value),
                "baseline_pred": as_int(baseline_pred_value),
                "target_pred": as_int(target_pred_value),
            }
        )
    return records, mismatched_labels


def baseline_prediction_records_from_rows(
    rows,
    pair_key,
    cluster_key,
    prediction_column,
    seed,
):
    records = []
    for row in rows:
        key = first_available(
            row,
            [pair_key, "image_npy_path", "pseudo_study_uid", "new_id", "row_index"],
        )
        label_value = first_available(row, ["binary_label", "label"])
        score_value = first_available(row, ["binary_score", "score"])
        pred_value = first_available(
            row,
            [prediction_column, "pred_threshold_default_0_5", "pred_default_0_5"],
        )
        if key is None or label_value is None or score_value is None or pred_value is None:
            continue

        cluster = first_available(
            row,
            [cluster_key, "person_id", pair_key, "image_npy_path", "row_index"],
        )
        records.append(
            {
                "seed": str(seed),
                "pair_key": key,
                "cluster": f"{seed}:{cluster}",
                "label": as_int(label_value),
                "baseline_score": as_float(score_value),
                "baseline_pred": as_int(pred_value),
            }
        )
    return records


def metric_from_records(records, metric, side):
    labels = [record["label"] for record in records]
    scores = [record[f"{side}_score"] for record in records]
    preds = [record[f"{side}_pred"] for record in records]
    return binary_metrics(labels, scores, preds, metric)


def percentile(sorted_values, probability):
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = probability * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def paired_bootstrap_test(
    records,
    metric,
    confidence=0.95,
    n_bootstraps=10000,
    random_seed=0,
    cluster_key="person_id",
):
    if not records:
        return {
            "analysis_method": "paired_prediction_bootstrap_not_run",
            "statistic_type": None,
            "statistic": None,
            "n": 0,
            "mean_delta": None,
            "sd_delta": None,
            "sem_delta": None,
            "diff_se": None,
            "baseline_se": None,
            "target_se": None,
            "t_stat": None,
            "df": None,
            "p_two_sided": None,
            "ci_low": None,
            "ci_high": None,
            "note": "no paired prediction rows",
        }

    baseline_value = metric_from_records(records, metric, "baseline")
    target_value = metric_from_records(records, metric, "target")
    if baseline_value is None or target_value is None:
        return {
            "analysis_method": "paired_prediction_bootstrap_not_run",
            "statistic_type": None,
            "statistic": None,
            "n": len(records),
            "mean_delta": None,
            "sd_delta": None,
            "sem_delta": None,
            "diff_se": None,
            "baseline_se": None,
            "target_se": None,
            "t_stat": None,
            "df": None,
            "p_two_sided": None,
            "ci_low": None,
            "ci_high": None,
            "note": f"metric is undefined for {metric}",
        }

    observed_delta = target_value - baseline_value
    clusters = defaultdict(list)
    for record in records:
        clusters[record["cluster"]].append(record)
    cluster_items = list(clusters.items())
    rng = random.Random(random_seed)
    deltas = []
    for _ in range(n_bootstraps):
        sampled_records = []
        for _ in cluster_items:
            _, cluster_records = rng.choice(cluster_items)
            sampled_records.extend(cluster_records)
        baseline_boot = metric_from_records(sampled_records, metric, "baseline")
        target_boot = metric_from_records(sampled_records, metric, "target")
        if baseline_boot is None or target_boot is None:
            continue
        deltas.append(target_boot - baseline_boot)

    if not deltas:
        return {
            "analysis_method": "paired_prediction_bootstrap_not_run",
            "statistic_type": None,
            "statistic": None,
            "n": len(records),
            "mean_delta": observed_delta,
            "sd_delta": None,
            "sem_delta": None,
            "diff_se": None,
            "baseline_se": None,
            "target_se": None,
            "t_stat": None,
            "df": None,
            "p_two_sided": None,
            "ci_low": None,
            "ci_high": None,
            "note": "all bootstrap samples had undefined metrics",
        }

    deltas_sorted = sorted(deltas)
    alpha = 1.0 - confidence
    sd_delta = math.sqrt(
        sum((delta - sum(deltas) / len(deltas)) ** 2 for delta in deltas)
        / (len(deltas) - 1)
    ) if len(deltas) > 1 else 0.0
    p_lower = (1 + sum(1 for delta in deltas if delta <= 0.0)) / (len(deltas) + 1)
    p_upper = (1 + sum(1 for delta in deltas if delta >= 0.0)) / (len(deltas) + 1)
    p_two_sided = min(1.0, 2.0 * min(p_lower, p_upper))
    if cluster_key == "person_id":
        analysis_method = "patient_clustered_paired_bootstrap"
    else:
        analysis_method = "clustered_paired_bootstrap"

    return {
        "analysis_method": analysis_method,
        "statistic_type": "bootstrap_delta",
        "statistic": observed_delta,
        "n": len(records),
        "mean_delta": observed_delta,
        "sd_delta": sd_delta,
        "sem_delta": None,
        "diff_se": sd_delta,
        "baseline_se": None,
        "target_se": None,
        "t_stat": None,
        "df": None,
        "p_two_sided": p_two_sided,
        "ci_low": percentile(deltas_sorted, alpha / 2.0),
        "ci_high": percentile(deltas_sorted, 1.0 - alpha / 2.0),
        "note": f"clusters={len(cluster_items)}; bootstraps={len(deltas)}",
    }


def baseline_prediction_bootstrap_test(
    records,
    metric,
    confidence=0.95,
    n_bootstraps=10000,
    random_seed=0,
    cluster_key="person_id",
):
    if not records:
        return {
            "analysis_method": "baseline_prediction_bootstrap_not_run",
            "statistic_type": None,
            "statistic": None,
            "n": 0,
            "mean_delta": None,
            "sd_delta": None,
            "sem_delta": None,
            "diff_se": None,
            "baseline_se": None,
            "target_se": None,
            "t_stat": None,
            "df": None,
            "p_two_sided": None,
            "ci_low": None,
            "ci_high": None,
            "note": "no baseline prediction rows",
        }

    baseline_value = metric_from_records(records, metric, "baseline")
    if baseline_value is None:
        return {
            "analysis_method": "baseline_prediction_bootstrap_not_run",
            "statistic_type": None,
            "statistic": None,
            "n": len(records),
            "mean_delta": None,
            "sd_delta": None,
            "sem_delta": None,
            "diff_se": None,
            "baseline_se": None,
            "target_se": None,
            "t_stat": None,
            "df": None,
            "p_two_sided": None,
            "ci_low": None,
            "ci_high": None,
            "note": f"metric is undefined for {metric}",
        }

    clusters = defaultdict(list)
    for record in records:
        clusters[record["cluster"]].append(record)
    cluster_items = list(clusters.items())
    rng = random.Random(random_seed)
    values = []
    for _ in range(n_bootstraps):
        sampled_records = []
        for _ in cluster_items:
            _, cluster_records = rng.choice(cluster_items)
            sampled_records.extend(cluster_records)
        value = metric_from_records(sampled_records, metric, "baseline")
        if value is not None:
            values.append(value)

    if not values:
        return {
            "analysis_method": "baseline_prediction_bootstrap_not_run",
            "statistic_type": None,
            "statistic": baseline_value,
            "n": len(records),
            "mean_delta": 0.0,
            "sd_delta": None,
            "sem_delta": None,
            "diff_se": None,
            "baseline_se": None,
            "target_se": None,
            "t_stat": None,
            "df": None,
            "p_two_sided": None,
            "ci_low": None,
            "ci_high": None,
            "note": "all bootstrap samples had undefined metrics",
        }

    values_sorted = sorted(values)
    alpha = 1.0 - confidence
    sd_value = math.sqrt(
        sum((value - sum(values) / len(values)) ** 2 for value in values)
        / (len(values) - 1)
    ) if len(values) > 1 else 0.0
    if cluster_key == "person_id":
        analysis_method = "patient_clustered_baseline_bootstrap"
    else:
        analysis_method = "clustered_baseline_bootstrap"

    return {
        "analysis_method": analysis_method,
        "statistic_type": "bootstrap_mean",
        "statistic": baseline_value,
        "n": len(records),
        "mean_delta": 0.0,
        "sd_delta": None,
        "sem_delta": None,
        "diff_se": None,
        "baseline_se": sd_value,
        "target_se": None,
        "t_stat": None,
        "df": None,
        "p_two_sided": None,
        "ci_low": percentile(values_sorted, alpha / 2.0),
        "ci_high": percentile(values_sorted, 1.0 - alpha / 2.0),
        "note": f"baseline mean CI; clusters={len(cluster_items)}; bootstraps={len(values)}",
    }


def format_value(value, precision=4, missing_value="NA"):
    if value is None:
        return missing_value
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return f"{value:.{precision}f}"
    return str(value)


def result_columns(rows, table_mode, analysis=None):
    if analysis == "per_seed":
        return per_seed_columns(rows, table_mode)

    compact_columns = [
        "selection",
        "target",
        "metric",
        "seeds",
        "baseline_mean",
        "target_mean",
        "mean_delta",
        "note",
    ]
    full_columns = [
        "selection",
        "baseline",
        "target",
        "model",
        "train_mode",
        "metric",
        "seeds",
        "baseline_mean",
        "target_mean",
        "mean_delta",
        "n",
        "analysis_method",
        "baseline_se",
        "target_se",
        "diff_se",
        "sd_delta",
        "sem_delta",
        "ci_low",
        "ci_high",
        "statistic_type",
        "statistic",
        "t_stat",
        "df",
        "p_two_sided",
        "note",
    ]
    holistic_columns = [
        "selection",
        "baseline",
        "target",
        "model",
        "train_mode",
        "metric",
        "seeds",
        "baseline_mean",
        "target_mean",
        "mean_delta",
        "n",
        "analysis_method",
        "baseline_se",
        "target_se",
        "diff_se",
        "sd_delta",
        "sem_delta",
        "ci_low",
        "ci_high",
        "statistic_type",
        "statistic",
        "t_stat",
        "df",
        "p_two_sided",
        "note",
    ]
    if table_mode == "holistic":
        return holistic_columns
    compact_output = table_mode == "compact"
    if table_mode == "auto":
        compact_output = not any((row.get("n") or 0) >= 2 for row in rows)

    if compact_output:
        for row in rows:
            if (row.get("n") or 0) < 2 and row.get("note") == (
                "paired t-test requires at least two matched seeds"
            ):
                row["note"] = ""
        return compact_columns
    if table_mode == "full":
        return full_columns
    return full_columns


def build_aggregate_baseline_rows(args):
    rows = []
    baseline_seeds = discover_metric_seeds(
        args.root,
        args.baseline,
        args.model,
        args.train_mode,
        args.selection,
        args.metric_prefix,
    )
    for metric in args.metrics:
        values = []
        seed_list = []
        missing = []
        for seed in baseline_seeds:
            metric_file = metric_file_for_seed(
                args.root,
                args.baseline,
                args.model,
                args.train_mode,
                seed,
                args.selection,
                args.metric_prefix,
            )
            data = read_metric_json(metric_file)
            if metric not in data:
                missing.append(seed)
                continue
            values.append(float(data[metric]))
            seed_list.append(seed)

        baseline_mean = sum(values) / len(values) if values else None
        stats = baseline_seed_bootstrap(
            values,
            confidence=args.confidence,
            n_bootstraps=args.seed_bootstrap_iterations,
            random_seed=args.seed_bootstrap_seed,
        )
        row = {
            "selection": args.selection,
            "baseline": args.baseline,
            "target": args.baseline,
            "model": args.model,
            "train_mode": args.train_mode,
            "metric": metric,
            "seeds": ",".join(seed_list),
            "baseline_mean": baseline_mean,
            "target_mean": baseline_mean,
            **stats,
        }
        notes = []
        if row.get("note"):
            notes.append(row["note"])
        if missing:
            notes.append("metric missing for baseline seeds: " + ",".join(missing))
        if not baseline_seeds:
            notes.append(
                "no baseline metric files found under "
                f"{args.root / args.baseline / args.model / args.train_mode}"
            )
        row["note"] = "; ".join(notes)
        rows.append(row)
    return rows


def build_prediction_baseline_rows(args):
    rows = []
    baseline_seeds = discover_prediction_seeds(
        args.root,
        args.baseline,
        args.model,
        args.train_mode,
        args.selection,
        args.metric_prefix,
        args.prediction_prefix,
    )
    records = []
    for seed in baseline_seeds:
        prediction_file = prediction_file_for_seed(
            args.root,
            args.baseline,
            args.model,
            args.train_mode,
            seed,
            args.selection,
            args.metric_prefix,
            args.prediction_prefix,
        )
        records.extend(
            baseline_prediction_records_from_rows(
                read_prediction_csv(prediction_file),
                args.pair_key,
                args.cluster_key,
                args.prediction_column,
                seed,
            )
        )

    for metric in args.metrics:
        baseline_mean = metric_from_records(records, metric, "baseline") if records else None
        stats = baseline_prediction_bootstrap_test(
            records,
            metric,
            confidence=args.confidence,
            n_bootstraps=args.bootstrap_iterations,
            random_seed=args.bootstrap_seed,
            cluster_key=args.cluster_key,
        )
        row = {
            "selection": args.selection,
            "baseline": args.baseline,
            "target": args.baseline,
            "model": args.model,
            "train_mode": args.train_mode,
            "metric": metric,
            "seeds": ",".join(baseline_seeds),
            "baseline_mean": baseline_mean,
            "target_mean": baseline_mean,
            **stats,
        }
        notes = []
        if row.get("note"):
            notes.append(row["note"])
        notes.append(
            f"pair_key={args.pair_key}; cluster_key={args.cluster_key}; "
            f"prediction_column={args.prediction_column}"
        )
        if not baseline_seeds:
            notes.append(
                "no baseline prediction files found under "
                f"{args.root / args.baseline / args.model / args.train_mode}"
            )
        row["note"] = "; ".join(notes)
        rows.append(row)
    return rows


def build_per_seed_rows(args):
    rows = []
    targets = per_seed_targets(args)
    baseline_cache = {}

    for target in targets:
        seeds = discover_metric_seeds(
            args.root,
            target,
            args.model,
            args.train_mode,
            args.selection,
            args.metric_prefix,
        )
        for seed in seeds:
            metric_file = metric_file_for_seed(
                args.root,
                target,
                args.model,
                args.train_mode,
                seed,
                args.selection,
                args.metric_prefix,
            )
            target_data = read_metric_json(metric_file)
            target_metrics = numeric_test_metrics(target_data)

            baseline_file = metric_file_for_seed(
                args.root,
                args.baseline,
                args.model,
                args.train_mode,
                seed,
                args.selection,
                args.metric_prefix,
            )
            baseline_metrics = {}
            notes = []
            if baseline_file.exists():
                if seed not in baseline_cache:
                    baseline_cache[seed] = numeric_test_metrics(read_metric_json(baseline_file))
                baseline_metrics = baseline_cache[seed]
            else:
                notes.append(f"same-seed baseline metric file missing: {baseline_file}")

            all_metrics = requested_per_seed_metrics(
                args, target_metrics, baseline_metrics
            )
            row = {
                "selection": args.selection,
                "baseline": args.baseline,
                "target": target,
                "model": args.model,
                "train_mode": args.train_mode,
                "seed": seed,
                "metric_file": metric_file,
                "baseline_metric_file": baseline_file if baseline_file.exists() else None,
                "note": "",
            }
            for metric in all_metrics:
                target_value = target_metrics.get(metric)
                baseline_value = baseline_metrics.get(metric)
                row[metric] = target_value
                row[f"baseline_{metric}"] = baseline_value
                if target_value is not None and baseline_value is not None:
                    row[f"delta_{metric}"] = target_value - baseline_value
                if target_value is None:
                    notes.append(f"{metric} missing for target")
                if baseline_file.exists() and baseline_value is None:
                    notes.append(f"{metric} missing for same-seed baseline")

            row["note"] = "; ".join(notes)
            rows.append(row)
    return rows


def build_rows(args):
    if args.analysis == "per_seed":
        return build_per_seed_rows(args)

    rows = []
    if args.include_baseline_row:
        if args.analysis == "paired_bootstrap":
            rows.extend(build_prediction_baseline_rows(args))
        else:
            rows.extend(build_aggregate_baseline_rows(args))
    targets = args.targets or discover_ruleouts(args.root)

    for target in targets:
        if args.analysis == "paired_bootstrap":
            matched_seeds = discover_matched_prediction_seeds(
                args.root,
                args.baseline,
                target,
                args.model,
                args.train_mode,
                args.selection,
                args.metric_prefix,
                args.prediction_prefix,
            )
            all_records = []
            mismatched_labels = 0
            for seed in matched_seeds:
                baseline_file = prediction_file_for_seed(
                    args.root,
                    args.baseline,
                    args.model,
                    args.train_mode,
                    seed,
                    args.selection,
                    args.metric_prefix,
                    args.prediction_prefix,
                )
                target_file = prediction_file_for_seed(
                    args.root,
                    target,
                    args.model,
                    args.train_mode,
                    seed,
                    args.selection,
                    args.metric_prefix,
                    args.prediction_prefix,
                )
                records, seed_mismatched_labels = prediction_records_from_rows(
                    read_prediction_csv(baseline_file),
                    read_prediction_csv(target_file),
                    args.pair_key,
                    args.cluster_key,
                    args.prediction_column,
                    seed,
                )
                all_records.extend(records)
                mismatched_labels += seed_mismatched_labels

            for metric in args.metrics:
                baseline_mean = (
                    metric_from_records(all_records, metric, "baseline")
                    if all_records
                    else None
                )
                target_mean = (
                    metric_from_records(all_records, metric, "target")
                    if all_records
                    else None
                )
                stats = paired_bootstrap_test(
                    all_records,
                    metric,
                    confidence=args.confidence,
                    n_bootstraps=args.bootstrap_iterations,
                    random_seed=args.bootstrap_seed,
                    cluster_key=args.cluster_key,
                )
                row = {
                    "selection": args.selection,
                    "baseline": args.baseline,
                    "target": target,
                    "model": args.model,
                    "train_mode": args.train_mode,
                    "metric": metric,
                    "seeds": ",".join(matched_seeds),
                    "baseline_mean": baseline_mean,
                    "target_mean": target_mean,
                    **stats,
                }
                notes = []
                if row.get("note"):
                    notes.append(row["note"])
                notes.append(
                    f"pair_key={args.pair_key}; cluster_key={args.cluster_key}; "
                    f"prediction_column={args.prediction_column}"
                )
                if mismatched_labels:
                    notes.append(f"mismatched labels skipped={mismatched_labels}")
                row["note"] = "; ".join(notes)
                rows.append(row)
            continue

        matched_seeds = discover_matched_seeds(
            args.root,
            args.baseline,
            target,
            args.model,
            args.train_mode,
            args.selection,
            args.metric_prefix,
        )
        no_match_note = ""
        if not matched_seeds:
            no_match_note = no_matched_seed_note(
                args.root,
                args.baseline,
                target,
                args.model,
                args.train_mode,
                args.selection,
                args.metric_prefix,
            )
        for metric in args.metrics:
            baseline_values = []
            target_values = []
            differences = []
            seed_list = []
            missing = []

            for seed in matched_seeds:
                baseline_file = metric_file_for_seed(
                    args.root,
                    args.baseline,
                    args.model,
                    args.train_mode,
                    seed,
                    args.selection,
                    args.metric_prefix,
                )
                target_file = metric_file_for_seed(
                    args.root,
                    target,
                    args.model,
                    args.train_mode,
                    seed,
                    args.selection,
                    args.metric_prefix,
                )
                baseline_data = read_metric_json(baseline_file)
                target_data = read_metric_json(target_file)
                if metric not in baseline_data or metric not in target_data:
                    missing.append(seed)
                    continue
                baseline_value = float(baseline_data[metric])
                target_value = float(target_data[metric])
                baseline_values.append(baseline_value)
                target_values.append(target_value)
                differences.append(target_value - baseline_value)
                seed_list.append(seed)

            stats_list = []
            if args.aggregate_test in {"paired_ttest", "both"}:
                stats_list.append(paired_ttest(differences, confidence=args.confidence))
            if args.aggregate_test in {"seed_paired_bootstrap", "both"}:
                stats_list.append(
                    paired_seed_bootstrap(
                        differences,
                        confidence=args.confidence,
                        n_bootstraps=args.seed_bootstrap_iterations,
                        random_seed=args.seed_bootstrap_seed,
                    )
                )
            baseline_mean = (
                sum(baseline_values) / len(baseline_values) if baseline_values else None
            )
            target_mean = sum(target_values) / len(target_values) if target_values else None
            for stats in stats_list:
                row = {
                    "selection": args.selection,
                    "baseline": args.baseline,
                    "target": target,
                    "model": args.model,
                    "train_mode": args.train_mode,
                    "metric": metric,
                    "seeds": ",".join(seed_list),
                    "baseline_mean": baseline_mean,
                    "target_mean": target_mean,
                    **stats,
                }
                if no_match_note:
                    row["note"] = (row["note"] + "; " if row["note"] else "") + no_match_note
                if missing:
                    row["note"] = (row["note"] + "; " if row["note"] else "") + (
                        "metric missing for seeds: " + ",".join(missing)
                    )
                rows.append(row)
    return rows


def print_table(rows, table_mode, analysis=None):
    columns = result_columns(rows, table_mode, analysis)
    missing_value = "" if analysis == "per_seed" else "NA"
    print("\t".join(columns))
    for row in rows:
        print(
            "\t".join(
                format_value(row.get(column), missing_value=missing_value)
                for column in columns
            )
        )


def write_csv(rows, path, table_mode, analysis=None):
    columns = result_columns(rows, table_mode, analysis)
    missing_value = "" if analysis == "per_seed" else "NA"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    column: format_value(row.get(column), missing_value=missing_value)
                    for column in columns
                }
            )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare Gleason baseline vs adversarial ruleout metric JSONs with "
            "paired t-tests across matched seed directories."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("output_cls/gleason/binary/grade_group_ge_2"),
        help="Directory containing ruleout_* experiment folders.",
    )
    parser.add_argument("--baseline", default="ruleout_none")
    parser.add_argument("--model", default="profound_conv")
    parser.add_argument("--train_mode", default="fintune")
    parser.add_argument(
        "--selection",
        choices=sorted(SELECTION_FILES),
        default="best_auc",
        help="Checkpoint-selection policy to compare consistently.",
    )
    parser.add_argument(
        "--metric_prefix",
        default="test_metrics",
        help=(
            "Metric JSON filename prefix. Default reads test_metrics_*.json; "
            "PROMIS external runs use promis_external_metrics."
        ),
    )
    parser.add_argument(
        "--prediction_prefix",
        default=None,
        help=(
            "Optional prediction CSV filename prefix. If unset, prediction CSVs "
            "are inferred from metric filenames as <metric_stem>_predictions.csv. "
            "PROMIS external runs use promis_external_predictions."
        ),
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=list(DEFAULT_METRICS),
        help=(
            "Metrics to compare or report. Aggregate mode reads these JSON fields. "
            "Per-seed mode reports these fields by default, or use 'all' to "
            "include all numeric test_* metrics. "
            "Paired-bootstrap mode supports test_auc, test_acc, "
            "test_balanced_acc, test_sens_at_80_spec, test_sensitivity, "
            "test_specificity, and test_f1."
        ),
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=None,
        help="Specific ruleout folders to compare. Defaults to all except baseline.",
    )
    parser.add_argument(
        "--no_baseline_row",
        dest="include_baseline_row",
        action="store_false",
        help=(
            "Do not emit the baseline-only row. By default the output includes "
            "one baseline row per metric with ci_low/ci_high giving the "
            "baseline evaluation confidence interval."
        ),
    )
    parser.set_defaults(include_baseline_row=True)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument(
        "--csv_output",
        type=Path,
        default=None,
        help="Optional CSV path for the summary table.",
    )
    parser.add_argument(
        "--table_mode",
        choices=["auto", "compact", "full", "holistic"],
        default="holistic",
        help=(
            "holistic writes a stable schema with comparison and paired-test "
            "columns. For per_seed, full includes paths and baseline_* columns; "
            "other modes write a compact metric/delta table."
        ),
    )
    parser.add_argument(
        "--analysis",
        choices=["aggregate", "paired_bootstrap", "per_seed"],
        default="aggregate",
        help=(
            "aggregate compares metric JSONs as before. paired_bootstrap joins "
            "saved per-scan prediction CSVs and uses clustered paired bootstrap "
            "for metric differences. per_seed writes one row per seed metric "
            "JSON with same-seed baseline deltas."
        ),
    )
    parser.add_argument(
        "--aggregate_test",
        choices=["paired_ttest", "seed_paired_bootstrap", "both"],
        default="paired_ttest",
        help=(
            "Statistical test for aggregate metric JSONs across matched seeds. "
            "Use both to emit paired t-test and seed-paired bootstrap rows."
        ),
    )
    parser.add_argument(
        "--pair_key",
        default="image_npy_path",
        help=(
            "Prediction CSV column used to pair the same scan across models. "
            "Fallbacks are image_npy_path, pseudo_study_uid, new_id, row_index."
        ),
    )
    parser.add_argument(
        "--cluster_key",
        default="person_id",
        help=(
            "Prediction CSV column used as the bootstrap cluster. Use person_id "
            "to account for repeated scans per patient, or image_npy_path for "
            "scan-level paired bootstrap."
        ),
    )
    parser.add_argument(
        "--prediction_column",
        default="pred_threshold_default_0_5",
        help=(
            "Prediction CSV binary-prediction column used for thresholded "
            "metrics such as balanced accuracy and F1."
        ),
    )
    parser.add_argument(
        "--bootstrap_iterations",
        type=int,
        default=10000,
        help="Number of paired-bootstrap resamples.",
    )
    parser.add_argument(
        "--bootstrap_seed",
        type=int,
        default=0,
        help="Random seed for paired-bootstrap resampling.",
    )
    parser.add_argument(
        "--seed_bootstrap_iterations",
        type=int,
        default=10000,
        help="Number of seed-level bootstrap resamples for aggregate mode.",
    )
    parser.add_argument(
        "--seed_bootstrap_seed",
        type=int,
        default=0,
        help="Random seed for seed-level aggregate bootstrap resampling.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rows = build_rows(args)
    print_table(rows, args.table_mode, args.analysis)
    if args.csv_output:
        write_csv(rows, args.csv_output, args.table_mode, args.analysis)
        print(f"Wrote CSV: {args.csv_output}")


if __name__ == "__main__":
    main()
