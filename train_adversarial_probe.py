import argparse
import datetime
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    class SummaryWriter:
        def __init__(self, *args, **kwargs):
            pass

        def add_scalar(self, *args, **kwargs):
            pass

        def flush(self):
            pass

import util.lr_sched as lr_sched
import util.misc as misc
from util.gleason_config import ADVERSARIAL_VARIABLE_CHOICES, resolve_adversarial_config
from util.paths import CLASSIFICATION_LOG_DIR, CLASSIFICATION_OUTPUT_DIR, DATA_ROOT


def parse_int_tuple(value):
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    value = str(value).replace("(", "").replace(")", "")
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


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


def adversarial_logits(model_output):
    if isinstance(model_output, dict):
        return model_output.get("adversarial", {})
    return {}


def accuracy_percent(targets, preds):
    targets = np.asarray(targets).astype(int)
    preds = np.asarray(preds).astype(int)
    if targets.size == 0:
        return float("nan")
    return float(np.mean(targets == preds) * 100)


def balanced_accuracy_percent(targets, preds):
    targets = np.asarray(targets).astype(int)
    preds = np.asarray(preds).astype(int)
    recalls = []
    for class_id in np.unique(targets):
        mask = targets == class_id
        if mask.any():
            recalls.append(np.mean(preds[mask] == class_id))
    if not recalls:
        return float("nan")
    return float(np.mean(recalls) * 100)


def binary_auc_percent(targets, scores):
    targets = np.asarray(targets).astype(int)
    scores = np.asarray(scores, dtype=float)
    positive = targets == 1
    negative = targets == 0
    n_pos = int(positive.sum())
    n_neg = int(negative.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    sorted_scores = scores[order]
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    auc = (ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc * 100)


def get_args_parser():
    parser = argparse.ArgumentParser(
        "Adversarial-only probe training",
        description=(
            "Train only adversarial head(s) on a frozen Gleason checkpoint. "
            "No Gleason loss is used."
        ),
    )
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--pin_mem", action="store_true")
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)
    parser.add_argument("--drop_last", action="store_true")
    parser.set_defaults(drop_last=False)

    parser.add_argument("--data_root", default=str(DATA_ROOT), type=str)
    parser.add_argument("--csv_path", required=True, type=str)
    parser.add_argument("--train_csv", default=None, type=str)
    parser.add_argument("--val_csv", default=None, type=str)
    parser.add_argument("--test_csv", default=None, type=str)
    parser.add_argument("--split_col", default="split", type=str)
    parser.add_argument("--image_path_col", default="image_npy_path", type=str)
    parser.add_argument("--in_channels", default=3, type=int)
    parser.add_argument("--crop_spatial_size", default=(64, 256, 256), type=parse_int_tuple)

    parser.add_argument("--task_type", choices=["binary", "ordinal"], required=True)
    parser.add_argument("--label_col", default="grade_group", type=str)
    parser.add_argument("--binary_label_col", default=None, type=str)
    parser.add_argument("--binary_positive_min", default=2, type=int)
    parser.add_argument("--ordinal_levels", default=5, type=int)
    parser.add_argument("--label_offset", default=1, type=int)
    parser.add_argument("--weighted_sampling", action="store_true")
    parser.set_defaults(weighted_sampling=False)

    parser.add_argument(
        "--adversarial_specs",
        default="",
        type=str,
        help="Advanced: one already-binned adversarial column, e.g. bmi_bin:4.",
    )
    parser.add_argument(
        "--adversarial_variable",
        default=None,
        choices=ADVERSARIAL_VARIABLE_CHOICES,
        type=str,
        help="One variable to probe.",
    )
    parser.add_argument("--adversarial_column", default=None, type=str)
    parser.add_argument("--adversarial_observed_column", default=None, type=str)
    parser.add_argument("--adversarial_num_classes", default=None, type=int)
    parser.add_argument("--adversarial_loss_weight", default=1.0, type=float)

    parser.add_argument(
        "--model",
        choices=["resnet18", "profound_conv", "profound_vit"],
        required=True,
        type=str,
    )
    parser.add_argument(
        "--train",
        choices=["fintune", "freeze", "scratch"],
        default="fintune",
        type=str,
        help="Encoder construction mode used to match the source Gleason checkpoint.",
    )
    parser.add_argument("--pretrain", default=None, type=str)
    parser.add_argument("--bottleneck_dim", default=256, type=int)

    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--min_lr", default=0.0, type=float)
    parser.add_argument("--warmup_epochs", default=0, type=int)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--output_dir", default=str(CLASSIFICATION_OUTPUT_DIR), type=str)
    parser.add_argument("--log_dir", default=str(CLASSIFICATION_LOG_DIR), type=str)
    parser.add_argument("--val_interval", default=1, type=int)
    parser.add_argument("--save_ckpt_interval", default=0, type=int)
    parser.add_argument(
        "--primary_metric",
        choices=["loss", "balanced_acc", "acc", "auc"],
        default="balanced_acc",
        type=str,
    )
    parser.add_argument(
        "--load_adversarial_head",
        action="store_true",
        help="Load matching adversarial-head weights from checkpoint instead of reinitializing.",
    )
    return parser


def configure_task_args(args):
    args.crop_spatial_size = tuple(args.crop_spatial_size)
    resolve_adversarial_config(args)
    if not args.adversarial_specs_resolved:
        raise ValueError("Provide --adversarial_variable or --adversarial_specs.")
    if len(args.adversarial_specs_resolved) != 1:
        raise ValueError("This probe script supports one adversarial variable per run.")

    if args.task_type == "binary":
        args.num_classes = 1
        args.label_definition = (
            f"{args.binary_label_col}=1"
            if args.binary_label_col
            else f"{args.label_col}>={args.binary_positive_min}"
        )
        definition_name = (
            args.binary_label_col
            if args.binary_label_col
            else f"{args.label_col}_ge_{args.binary_positive_min}"
        )
    elif args.task_type == "ordinal":
        args.num_classes = args.ordinal_levels - 1
        args.label_definition = (
            f"{args.label_col} ordinal, levels={args.ordinal_levels}, "
            f"offset={args.label_offset}"
        )
        definition_name = f"{args.label_col}_ordinal_{args.ordinal_levels}levels"
    else:
        raise NotImplementedError(f"unknown task_type: {args.task_type}")

    adversarial_name = next(iter(args.adversarial_specs_resolved))
    checkpoint_name = Path(args.checkpoint).stem
    args.output_dir = os.path.join(
        args.output_dir,
        "adversarial_probe",
        args.task_type,
        definition_name,
        f"probe_{adversarial_name}",
        args.model,
        checkpoint_name,
        str(args.seed),
    )
    args.log_dir = os.path.join(
        args.log_dir,
        "adversarial_probe",
        args.task_type,
        definition_name,
        f"probe_{adversarial_name}",
        args.model,
        checkpoint_name,
        str(args.seed),
    )
    return args


def build_probe_model(args, device):
    from models.build_gleason_classification import build_encoder
    from models.gleason_classifier import GleasonClassifier

    encoder = build_encoder(args)
    model = GleasonClassifier(
        encoder=encoder,
        main_out_dim=args.num_classes,
        adversarial_num_classes=args.adversarial_specs_resolved,
        bottleneck_dim=args.bottleneck_dim,
        grl_lambda=0.0,
    ).to(device)
    return model


def load_gleason_checkpoint(model, checkpoint_path, device, load_adversarial_head=False):
    checkpoint = misc.load_trusted_checkpoint(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model", checkpoint)
    if not load_adversarial_head:
        state_dict = {
            key: value
            for key, value in state_dict.items()
            if not key.startswith("adversarial_heads.")
        }
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Missing keys: {missing}")
    print(f"Unexpected keys: {unexpected}")
    return checkpoint


def freeze_for_probe(model):
    for param in model.parameters():
        param.requires_grad = False
    for param in model.adversarial_heads.parameters():
        param.requires_grad = True
    model.set_grl_lambda(0.0)


def adversarial_probe_loss(model_output, adversarial_targets, adversarial_masks, args):
    logits_by_name = adversarial_logits(model_output)
    if not logits_by_name:
        raise ValueError("Model has no adversarial logits.")

    loss_fn = torch.nn.CrossEntropyLoss()
    losses = {}
    counts = {}
    total = None
    for name, logits in logits_by_name.items():
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
        counts[name] = int(target.numel())
        total = loss if total is None else total + loss

    if total is None:
        return None, {}, {}
    return total * args.adversarial_loss_weight, losses, counts


def train_one_epoch(model, data_loader, optimizer, device, epoch, log_writer, args):
    model.train(True)
    model.encoder.eval()
    model.main_head.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Probe Epoch: [{epoch}]"
    optimizer.zero_grad(set_to_none=True)

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, 20, header)):
        img, _, adversarial_targets, adversarial_masks, _ = unpack_batch(batch)
        img = img.to(device, non_blocking=True)
        lr_sched.adjust_learning_rate(
            optimizer, data_iter_step / len(data_loader) + epoch, args
        )
        model_output = model(img)
        loss, losses_by_name, counts_by_name = adversarial_probe_loss(
            model_output, adversarial_targets, adversarial_masks, args
        )
        if loss is None:
            metric_logger.update(skipped_batches=1)
            continue
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            raise FloatingPointError(f"Non-finite probe loss: {loss_value}")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        metric_logger.update(loss=loss_value)
        for name, value in losses_by_name.items():
            metric_logger.update(**{f"adv_{name}_loss": value.item()})
        for name, count in counts_by_name.items():
            metric_logger.update(**{f"adv_{name}_observed": count})
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr, skipped_batches=0)

        if log_writer is not None:
            step = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar("train_probe_loss", loss_value, step)
            log_writer.add_scalar("lr", lr, step)

    print("Averaged probe stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_probe(model, data_loader, device, args, phase):
    model.eval()
    losses = []
    logits_by_name = {}
    targets_by_name = {}
    observed_counts = {}

    for batch in data_loader:
        img, _, adversarial_targets, adversarial_masks, _ = unpack_batch(batch)
        img = img.to(device, non_blocking=True)
        model_output = model(img)
        loss, _, counts = adversarial_probe_loss(
            model_output, adversarial_targets, adversarial_masks, args
        )
        if loss is not None:
            losses.append(loss.item())
        for name, count in counts.items():
            observed_counts[name] = observed_counts.get(name, 0) + count
        for name, logits in adversarial_logits(model_output).items():
            target = adversarial_targets[name]
            mask = adversarial_masks.get(name)
            if mask is not None:
                mask = mask.bool()
                if not mask.any():
                    continue
                logits = logits[mask]
                target = target[mask]
            logits_by_name.setdefault(name, []).append(logits.detach().cpu())
            targets_by_name.setdefault(name, []).append(target.detach().cpu())

    stats = {"loss": float(np.mean(losses)) if losses else float("nan")}
    for name, chunks in logits_by_name.items():
        if not chunks:
            continue
        logits = torch.cat(chunks, 0)
        targets = torch.cat(targets_by_name[name], 0).numpy().astype(int)
        probs = torch.softmax(logits, dim=1).numpy()
        preds = probs.argmax(axis=1)
        stats[f"adv_{name}_observed"] = int(observed_counts.get(name, len(targets)))
        stats[f"adv_{name}_acc"] = accuracy_percent(targets, preds)
        stats[f"adv_{name}_balanced_acc"] = balanced_accuracy_percent(targets, preds)
        if probs.shape[1] == 2:
            stats[f"adv_{name}_auc"] = binary_auc_percent(targets, probs[:, 1])
    print(f"{phase} probe stats: {stats}")
    return stats


def metric_value(stats, args):
    adv_name = next(iter(args.adversarial_specs_resolved))
    if args.primary_metric == "loss":
        return stats.get("loss", float("nan"))
    return stats.get(f"adv_{adv_name}_{args.primary_metric}", float("nan"))


def is_better(value, best, args):
    if not np.isfinite(value):
        return False
    if args.primary_metric == "loss":
        return value < best
    return value > best


def save_probe_checkpoint(args, model, optimizer, epoch, filename):
    path = Path(args.output_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    misc.save_on_master(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "args": args,
        },
        path,
    )
    return path


def main(args):
    from dataset.gleason_cls import build_gleason_classification_loaders

    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True

    args = configure_task_args(args)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    data_loader_train, data_loader_val, data_loader_test = (
        build_gleason_classification_loaders(args)
    )
    print("{}".format(args).replace(", ", ",\n"))
    print(f"label definition: {args.label_definition}")
    print(f"adversarial definition: {args.adversarial_definition}")
    print(
        f"train batches: {len(data_loader_train)}, "
        f"val batches: {len(data_loader_val)}, test batches: {len(data_loader_test)}"
    )
    for split_name, loader in [
        ("train", data_loader_train),
        ("val", data_loader_val),
        ("test", data_loader_test),
    ]:
        counts_fn = getattr(loader.dataset, "adversarial_observed_counts", None)
        if counts_fn is not None:
            print(f"{split_name} adversarial observed counts: {counts_fn()}")

    model = build_probe_model(args, device)
    load_gleason_checkpoint(
        model,
        args.checkpoint,
        device,
        load_adversarial_head=args.load_adversarial_head,
    )
    freeze_for_probe(model)
    probe_params = [p for p in model.adversarial_heads.parameters() if p.requires_grad]
    if not probe_params:
        raise ValueError("No trainable adversarial-head parameters found.")
    optimizer = torch.optim.AdamW(
        probe_params, lr=args.lr, weight_decay=args.weight_decay
    )
    print(f"Trainable probe params: {sum(p.numel() for p in probe_params)}")

    log_writer = SummaryWriter(log_dir=args.log_dir)
    start_time = time.time()
    best = np.inf if args.primary_metric == "loss" else -np.inf
    best_epoch = -1

    for epoch in range(args.epochs):
        train_stats = train_one_epoch(
            model, data_loader_train, optimizer, device, epoch, log_writer, args
        )
        misc.write_log(
            log_writer,
            {**{f"train_{k}": v for k, v in train_stats.items()}, "epoch": epoch},
            args,
        )

        if epoch % args.val_interval == 0 or epoch + 1 == args.epochs:
            val_stats = evaluate_probe(model, data_loader_val, device, args, phase="val")
            value = metric_value(val_stats, args)
            best_flag = is_better(value, best, args)
            if best_flag:
                best = value
                best_epoch = epoch
            save_probe_checkpoint(args, model, optimizer, epoch, "last.pth.tar")
            if best_flag:
                save_probe_checkpoint(args, model, optimizer, epoch, "best.pth.tar")
            if args.save_ckpt_interval > 0 and (
                (epoch + 1) % args.save_ckpt_interval == 0
                or epoch + 1 == args.epochs
            ):
                save_probe_checkpoint(
                    args, model, optimizer, epoch, f"checkpoint-{epoch}.pth"
                )
            misc.write_log(
                log_writer,
                {
                    **{f"val_{k}": v for k, v in val_stats.items()},
                    "epoch": epoch,
                    "primary_metric": args.primary_metric,
                    "primary_metric_value": value,
                    "is_best": best_flag,
                    "best_epoch": best_epoch,
                    "label_definition": args.label_definition,
                    "adversarial_definition": args.adversarial_definition,
                },
                args,
            )

    total_time = time.time() - start_time
    print("Probe training time {}".format(str(datetime.timedelta(seconds=int(total_time)))))
    best_path = Path(args.output_dir) / "best.pth.tar"
    if best_path.exists():
        checkpoint = misc.load_trusted_checkpoint(best_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
    test_stats = evaluate_probe(model, data_loader_test, device, args, phase="test")
    misc.write_log(
        log_writer,
        {
            **{f"test_{k}": v for k, v in test_stats.items()},
            "label_definition": args.label_definition,
            "adversarial_definition": args.adversarial_definition,
            "best_epoch": best_epoch,
        },
        args,
    )
    metrics_path = Path(args.output_dir) / "test_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(test_stats, handle, indent=2, sort_keys=True)
    print(f"Wrote test metrics: {metrics_path}")


if __name__ == "__main__":
    main(get_args_parser().parse_args())
