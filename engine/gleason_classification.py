import math
import os
import sys

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

import util.lr_sched as lr_sched
import util.misc as misc


def ordinal_targets(labels: torch.Tensor, num_levels: int) -> torch.Tensor:
    thresholds = torch.arange(num_levels - 1, device=labels.device)
    return (labels.view(-1, 1) > thresholds.view(1, -1)).float()


def prepare_targets(labels: torch.Tensor, args) -> torch.Tensor:
    if args.task_type == "binary":
        return labels.float().view(-1, 1)
    if args.task_type == "ordinal":
        return ordinal_targets(labels.long(), args.ordinal_levels)
    raise NotImplementedError(f"unknown task_type: {args.task_type}")


def criterion_for_task(args):
    if args.task_type in {"binary", "ordinal"}:
        return torch.nn.BCEWithLogitsLoss()
    raise NotImplementedError(f"unknown task_type: {args.task_type}")


def decode_predictions(logits: torch.Tensor, args):
    if args.task_type == "binary":
        scores = torch.sigmoid(logits).view(-1).cpu().numpy()
        preds = (scores >= 0.5).astype(int)
        return preds, scores

    scores = torch.sigmoid(logits).cpu().numpy()
    preds = (scores >= 0.5).sum(axis=1).astype(int)
    return preds, scores


def binary_scores_for_grade_threshold(labels, scores, args):
    if args.task_type == "binary":
        return labels.astype(int), np.asarray(scores, dtype=float)

    positive_class_index = int(args.binary_positive_min) - int(args.label_offset)
    binary_labels = (labels >= positive_class_index).astype(int)
    threshold_index = positive_class_index - 1
    if threshold_index < 0 or threshold_index >= scores.shape[1]:
        return binary_labels, None
    return binary_labels, np.asarray(scores[:, threshold_index], dtype=float)


def rate_at_operating_point(labels, scores, min_specificity=None, min_sensitivity=None):
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    if len(np.unique(labels)) < 2:
        return float("nan"), float("nan")

    fpr, tpr, thresholds = roc_curve(labels, scores)
    specificity = 1.0 - fpr

    if min_specificity is not None:
        keep = specificity >= min_specificity
        if not np.any(keep):
            return float("nan"), float("nan")
        kept = np.where(keep)[0]
        best = kept[np.argmax(tpr[kept])]
        return float(tpr[best] * 100), float(thresholds[best])

    if min_sensitivity is not None:
        keep = tpr >= min_sensitivity
        if not np.any(keep):
            return float("nan"), float("nan")
        kept = np.where(keep)[0]
        best = kept[np.argmax(specificity[kept])]
        return float(specificity[best] * 100), float(thresholds[best])

    raise ValueError("Provide min_specificity or min_sensitivity")


def binary_threshold_stats(labels, scores, threshold):
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    preds = (scores >= threshold).astype(int)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "threshold": float(threshold),
        "acc": float(accuracy_score(labels, preds) * 100),
        "balanced_acc": float(balanced_accuracy_score(labels, preds) * 100),
        "sensitivity": (
            float(tp / (tp + fn) * 100) if (tp + fn) > 0 else float("nan")
        ),
        "specificity": (
            float(tn / (tn + fp) * 100) if (tn + fp) > 0 else float("nan")
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def select_binary_thresholds(labels, scores):
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    thresholds = {"default_0_5": 0.5}
    if len(np.unique(labels)) < 2:
        return thresholds

    fpr, tpr, roc_thresholds = roc_curve(labels, scores)
    specificity = 1.0 - fpr
    finite = np.isfinite(roc_thresholds)
    candidate_thresholds = roc_thresholds[finite]
    if len(candidate_thresholds) == 0:
        return thresholds

    balanced_acc = (tpr[finite] + specificity[finite]) / 2.0
    youden = tpr[finite] + specificity[finite] - 1.0
    thresholds["max_balanced_acc"] = float(
        candidate_thresholds[np.argmax(balanced_acc)]
    )
    thresholds["youden"] = float(candidate_thresholds[np.argmax(youden)])

    _, threshold = rate_at_operating_point(
        labels, scores, min_specificity=0.80
    )
    if not np.isnan(threshold):
        thresholds["sens_at_80_spec"] = float(threshold)
    _, threshold = rate_at_operating_point(
        labels, scores, min_sensitivity=0.80
    )
    if not np.isnan(threshold):
        thresholds["spec_at_80_sens"] = float(threshold)
    return thresholds


def threshold_metrics_for_scores(labels, scores, thresholds):
    metrics = {}
    for name, threshold in thresholds.items():
        stats = binary_threshold_stats(labels, scores, threshold)
        for key, value in stats.items():
            metrics[f"{name}_{key}"] = value
    return metrics


def ordinal_class_probabilities(threshold_scores):
    threshold_scores = np.asarray(threshold_scores, dtype=float)
    threshold_scores = np.minimum.accumulate(threshold_scores, axis=1)
    first = 1.0 - threshold_scores[:, :1]
    middle = threshold_scores[:, :-1] - threshold_scores[:, 1:]
    last = threshold_scores[:, -1:]
    probs = np.concatenate([first, middle, last], axis=1)
    probs = np.clip(probs, 0.0, 1.0)
    normalizer = probs.sum(axis=1, keepdims=True)
    return np.divide(
        probs,
        normalizer,
        out=np.full_like(probs, 1.0 / probs.shape[1]),
        where=normalizer > 0,
    )


def add_binary_operating_metrics(stats, labels, preds, scores, prefix=""):
    stats[f"{prefix}balanced_acc"] = float(
        balanced_accuracy_score(labels, preds) * 100
    )
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    stats[f"{prefix}sensitivity"] = (
        float(tp / (tp + fn) * 100) if (tp + fn) > 0 else float("nan")
    )
    stats[f"{prefix}specificity"] = (
        float(tn / (tn + fp) * 100) if (tn + fp) > 0 else float("nan")
    )
    try:
        stats[f"{prefix}auc"] = float(roc_auc_score(labels, scores) * 100)
    except ValueError:
        stats[f"{prefix}auc"] = float("nan")
    sens, sens_threshold = rate_at_operating_point(
        labels, scores, min_specificity=0.80
    )
    spec, spec_threshold = rate_at_operating_point(
        labels, scores, min_sensitivity=0.80
    )
    stats[f"{prefix}sens_at_80_spec"] = sens
    stats[f"{prefix}sens_at_80_spec_threshold"] = sens_threshold
    stats[f"{prefix}spec_at_80_sens"] = spec
    stats[f"{prefix}spec_at_80_sens_threshold"] = spec_threshold


def unpack_batch(batch):
    if len(batch) == 3:
        img, labels, dataidx = batch
        return img, labels, {}, {}, dataidx
    if len(batch) == 4:
        img, labels, adversarial_targets, dataidx = batch
        return img, labels, adversarial_targets, {}, dataidx
    if len(batch) == 5:
        img, labels, adversarial_targets, adversarial_masks, dataidx = batch
        return img, labels, adversarial_targets, adversarial_masks, dataidx
    raise ValueError(f"Unexpected batch length: {len(batch)}")


def main_logits(model_output):
    if isinstance(model_output, dict):
        return model_output["main"]
    return model_output


def adversarial_logits(model_output):
    if isinstance(model_output, dict):
        return model_output.get("adversarial", {})
    return {}


def adversarial_loss(model_output, adversarial_targets, adversarial_masks, args):
    logits_by_name = adversarial_logits(model_output)
    if not logits_by_name:
        return None, {}

    loss_fn = torch.nn.CrossEntropyLoss()
    losses = {}
    total = 0.0
    for name, logits in logits_by_name.items():
        if name not in adversarial_targets:
            raise KeyError(f"Missing adversarial target for '{name}'")
        target = adversarial_targets[name].to(logits.device, non_blocking=True)
        mask = adversarial_masks.get(name)
        if mask is not None:
            mask = mask.to(logits.device, non_blocking=True).bool()
            if not mask.any():
                continue
            logits = logits[mask]
            target = target[mask]
        loss = loss_fn(logits, target.long())
        losses[name] = loss
        total = total + loss
    if not losses:
        return None, {}
    return total * args.adversarial_loss_weight, losses


def grl_lambda_for_step(epoch, data_iter_step, steps_per_epoch, args):
    target_lambda = float(getattr(args, "grl_lambda", 1.0))
    if getattr(args, "grl_schedule", "constant") == "constant":
        return target_lambda

    if getattr(args, "grl_schedule", "constant") == "dann":
        total_steps = max(int(getattr(args, "epochs", 1)) * int(steps_per_epoch), 1)
        current_step = min(epoch * int(steps_per_epoch) + data_iter_step, total_steps)
        progress = current_step / total_steps
        gamma = float(getattr(args, "grl_gamma", 10.0))
        return target_lambda * (2.0 / (1.0 + math.exp(-gamma * progress)) - 1.0)

    raise ValueError(f"unknown grl_schedule: {args.grl_schedule}")


def set_model_grl_lambda(model, value):
    if hasattr(model, "set_grl_lambda"):
        model.set_grl_lambda(value)
    else:
        model.grl_lambda = float(value)


def train_one_epoch(
    model,
    data_loader,
    optimizer,
    device,
    epoch: int,
    log_writer=None,
    args=None,
):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"
    print_freq = 20
    loss_fn = criterion_for_task(args)

    optimizer.zero_grad(set_to_none=True)
    for data_iter_step, batch in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        img, labels, adversarial_targets, adversarial_masks, dataidx = unpack_batch(batch)
        img = img.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        targets = prepare_targets(labels, args)

        lr_sched.adjust_learning_rate(
            optimizer, data_iter_step / len(data_loader) + epoch, args
        )
        current_grl_lambda = grl_lambda_for_step(
            epoch, data_iter_step, len(data_loader), args
        )
        set_model_grl_lambda(model, current_grl_lambda)
        model_output = model(img)
        logits = main_logits(model_output)
        grade_loss = loss_fn(logits, targets)
        loss = grade_loss
        adv_loss, adv_losses = adversarial_loss(
            model_output, adversarial_targets, adversarial_masks, args
        )
        if adv_loss is not None:
            loss = loss + adv_loss
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print("nan", torch.isnan(logits).any(), torch.isnan(img).any(), dataidx)
            print("inf", torch.isinf(logits).any(), torch.isinf(img).any(), dataidx)
            sys.exit(1)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        metric_logger.update(loss=loss_value, grade_loss=grade_loss.item())
        if adv_loss is not None:
            metric_logger.update(adversarial_loss=adv_loss.item())
        for name, value in adv_losses.items():
            metric_logger.update(**{f"adv_{name}_loss": value.item()})
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr, grl_lambda=current_grl_lambda)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar("train_loss", loss_value_reduce, epoch_1000x)
            log_writer.add_scalar(
                "train_grade_loss",
                misc.all_reduce_mean(grade_loss.item()),
                epoch_1000x,
            )
            if adv_loss is not None:
                log_writer.add_scalar(
                    "train_adversarial_loss",
                    misc.all_reduce_mean(adv_loss.item()),
                    epoch_1000x,
                )
            log_writer.add_scalar("grl_lambda", current_grl_lambda, epoch_1000x)
            log_writer.add_scalar("lr", lr, epoch_1000x)

    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def validation(model, data_loader_val, device, epoch, args):
    stats = evaluate(model, data_loader_val, device, args, phase="val")
    print(f"epoch: {epoch}/{args.epochs}, val loss: {stats['loss']:.6f}")
    return stats["loss"], stats


def evaluate(model, data_loader, device, args, phase):
    stats, _, _ = evaluate_with_scores(model, data_loader, device, args, phase)
    return stats


def evaluate_with_scores(model, data_loader, device, args, phase):
    model.eval()
    loss_fn = criterion_for_task(args)
    losses = []
    grade_losses = []
    adversarial_losses = []
    logits_all = []
    labels_all = []
    adv_logits_all = {}
    adv_targets_all = {}

    with torch.no_grad():
        for batch in data_loader:
            img, labels, adversarial_targets, adversarial_masks, _ = unpack_batch(batch)
            img = img.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            targets = prepare_targets(labels, args)
            model_output = model(img)
            logits = main_logits(model_output)
            grade_loss = loss_fn(logits, targets)
            loss = grade_loss
            adv_loss, _ = adversarial_loss(
                model_output, adversarial_targets, adversarial_masks, args
            )
            if adv_loss is not None:
                loss = loss + adv_loss
                adversarial_losses.append(adv_loss.item())
            losses.append(loss.item())
            grade_losses.append(grade_loss.item())
            logits_all.append(logits.detach().cpu())
            labels_all.append(labels.detach().cpu())
            for name, adv_logits in adversarial_logits(model_output).items():
                target = adversarial_targets[name]
                mask = adversarial_masks.get(name)
                if mask is not None:
                    mask = mask.bool()
                    if not mask.any():
                        continue
                    adv_logits = adv_logits[mask]
                    target = target[mask]
                adv_logits_all.setdefault(name, []).append(adv_logits.detach().cpu())
                adv_targets_all.setdefault(name, []).append(target.detach().cpu())

    logits = torch.cat(logits_all, 0)
    labels = torch.cat(labels_all, 0).numpy().astype(int)
    preds, scores = decode_predictions(logits, args)

    stats = {
        "loss": float(np.mean(losses)),
        "grade_loss": float(np.mean(grade_losses)),
        "acc": float(accuracy_score(labels, preds) * 100),
        "balanced_acc": float(balanced_accuracy_score(labels, preds) * 100),
    }
    if adversarial_losses:
        stats["adversarial_loss"] = float(np.mean(adversarial_losses))

    binary_labels, binary_scores = binary_scores_for_grade_threshold(labels, scores, args)
    if binary_scores is not None and args.task_type == "binary":
        binary_preds = (binary_scores >= 0.5).astype(int)
        add_binary_operating_metrics(stats, binary_labels, binary_preds, binary_scores)

    if args.task_type == "ordinal":
        if binary_scores is not None:
            binary_preds = (binary_scores >= 0.5).astype(int)
            add_binary_operating_metrics(
                stats,
                binary_labels,
                binary_preds,
                binary_scores,
                prefix=f"{args.label_col}_ge_{args.binary_positive_min}_",
            )
            binary_prefix = f"{args.label_col}_ge_{args.binary_positive_min}_"
            for metric in [
                "auc",
                "sens_at_80_spec",
                "sens_at_80_spec_threshold",
                "spec_at_80_sens",
                "spec_at_80_sens_threshold",
            ]:
                stats[metric] = stats[f"{binary_prefix}{metric}"]
        stats["mae"] = float(np.mean(np.abs(preds - labels)))
        stats["qwk"] = float(cohen_kappa_score(labels, preds, weights="quadratic"))
        class_probs = ordinal_class_probabilities(scores)
        try:
            stats["ordinal_macro_auc"] = float(
                roc_auc_score(
                    labels,
                    class_probs,
                    labels=np.arange(args.ordinal_levels),
                    multi_class="ovr",
                    average="macro",
                )
                * 100
            )
        except ValueError:
            stats["ordinal_macro_auc"] = float("nan")

    for name, chunks in adv_logits_all.items():
        if not chunks:
            continue
        adv_logits = torch.cat(chunks, 0)
        adv_targets = torch.cat(adv_targets_all[name], 0).numpy().astype(int)
        adv_prob = torch.softmax(adv_logits, dim=1).numpy()
        adv_preds = adv_prob.argmax(axis=1)
        stats[f"adv_{name}_acc"] = float(accuracy_score(adv_targets, adv_preds) * 100)
        stats[f"adv_{name}_balanced_acc"] = float(
            balanced_accuracy_score(adv_targets, adv_preds) * 100
        )
        if adv_prob.shape[1] == 2:
            try:
                stats[f"adv_{name}_auc"] = float(
                    roc_auc_score(adv_targets, adv_prob[:, 1]) * 100
                )
            except ValueError:
                stats[f"adv_{name}_auc"] = float("nan")

    print(f"{phase} stats: {stats}")
    return stats, labels, scores


def test(model, test_loader, args):
    filepath_best = os.path.join(args.output_dir, "best.pth.tar")
    checkpoint = misc.load_trusted_checkpoint(filepath_best, map_location=args.device)
    model.load_state_dict(checkpoint["model"])
    return evaluate(model, test_loader, args.device, args, phase="test")
