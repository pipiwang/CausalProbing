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
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

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


class MLPHead(nn.Module):
    def __init__(self, embed_dim, out_dim, bottleneck_dim=256):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embed_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.ReLU(),
            nn.Linear(bottleneck_dim, out_dim),
        )

    def forward(self, x):
        return self.layers(x)


class DirectAdversarialClassifier(nn.Module):
    def __init__(self, encoder, adversarial_num_classes, bottleneck_dim=256):
        super().__init__()
        self.encoder = encoder
        self.embed_dim = self.encoder.embed_dim
        self.adversarial_heads = nn.ModuleDict(
            {
                name: MLPHead(self.embed_dim, num_classes, bottleneck_dim)
                for name, num_classes in adversarial_num_classes.items()
            }
        )

    def forward_features(self, x):
        features = self.encoder(x)
        if isinstance(features, tuple):
            features = features[0]
        return features

    def forward(self, x):
        features = self.forward_features(x)
        return {
            "adversarial": {
                name: head(features) for name, head in self.adversarial_heads.items()
            }
        }


def adversarial_logits(model_output):
    return model_output.get("adversarial", {})


def get_args_parser():
    parser = argparse.ArgumentParser(
        "Direct MRI-to-adversarial-variable training",
        description=(
            "Train an MRI model to predict one adversarial/clinical variable. "
            "No Gleason classifier or Gleason loss is used."
        ),
    )
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

    # These label args are only used by the shared MRI dataset validation code.
    parser.add_argument("--task_type", choices=["binary", "ordinal"], default="binary")
    parser.add_argument("--label_col", default="grade_group", type=str)
    parser.add_argument("--binary_label_col", default=None, type=str)
    parser.add_argument("--binary_positive_min", default=2, type=int)
    parser.add_argument("--ordinal_levels", default=5, type=int)
    parser.add_argument("--label_offset", default=1, type=int)

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
        help="One variable to predict directly from MRI.",
    )
    parser.add_argument("--adversarial_column", default=None, type=str)
    parser.add_argument("--adversarial_observed_column", default=None, type=str)
    parser.add_argument("--adversarial_num_classes", default=None, type=int)
    parser.add_argument("--adversarial_loss_weight", default=1.0, type=float)
    parser.add_argument(
        "--adversarial_weighted_sampling",
        action="store_true",
        help="Sample observed adversarial classes with inverse-frequency weights.",
    )
    parser.set_defaults(adversarial_weighted_sampling=False)

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
    )
    parser.add_argument("--pretrain", default=None, type=str)
    parser.add_argument("--bottleneck_dim", default=256, type=int)

    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--min_lr", default=0.0, type=float)
    parser.add_argument("--warmup_epochs", default=5, type=int)
    parser.add_argument("--weight_decay", default=0.05, type=float)
    parser.add_argument("--layer_decay", default=0.6, type=float)
    parser.add_argument(
        "--layer_decay_type",
        choices=["single", "group"],
        default="group",
        type=str,
    )
    parser.add_argument("--output_dir", default=str(CLASSIFICATION_OUTPUT_DIR), type=str)
    parser.add_argument("--log_dir", default=str(CLASSIFICATION_LOG_DIR), type=str)
    parser.add_argument("--val_interval", default=1, type=int)
    parser.add_argument("--save_ckpt_interval", default=10, type=int)
    parser.add_argument(
        "--primary_metric",
        choices=["loss", "balanced_acc", "acc", "auc"],
        default="balanced_acc",
        type=str,
    )
    return parser


def configure_args(args):
    args.crop_spatial_size = tuple(args.crop_spatial_size)
    args.weighted_sampling = False
    resolve_adversarial_config(args)
    if not args.adversarial_specs_resolved:
        raise ValueError("Provide --adversarial_variable or --adversarial_specs.")
    if len(args.adversarial_specs_resolved) != 1:
        raise ValueError("This direct trainer supports one adversarial variable per run.")

    if args.task_type == "binary":
        args.num_classes = 1
        label_name = (
            args.binary_label_col
            if args.binary_label_col
            else f"{args.label_col}_ge_{args.binary_positive_min}"
        )
        args.label_definition = (
            f"{args.binary_label_col}=1"
            if args.binary_label_col
            else f"{args.label_col}>={args.binary_positive_min}"
        )
    elif args.task_type == "ordinal":
        args.num_classes = args.ordinal_levels - 1
        label_name = f"{args.label_col}_ordinal_{args.ordinal_levels}levels"
        args.label_definition = (
            f"{args.label_col} ordinal, levels={args.ordinal_levels}, "
            f"offset={args.label_offset}"
        )
    else:
        raise NotImplementedError(f"unknown task_type: {args.task_type}")

    adversarial_name = next(iter(args.adversarial_specs_resolved))
    args.output_dir = os.path.join(
        args.output_dir,
        "adversarial_direct",
        args.task_type,
        label_name,
        f"predict_{adversarial_name}",
        args.model,
        args.train,
        str(args.seed),
    )
    args.log_dir = os.path.join(
        args.log_dir,
        "adversarial_direct",
        args.task_type,
        label_name,
        f"predict_{adversarial_name}",
        args.model,
        args.train,
        str(args.seed),
    )
    return args


def csv_for_phase(args, phase):
    phase_csv = getattr(args, f"{phase}_csv")
    return phase_csv if phase_csv else args.csv_path


def adversarial_sampler_weights(dataset):
    if len(dataset.adversarial_specs) != 1:
        raise ValueError("Weighted sampling supports one adversarial variable per run.")
    name = next(iter(dataset.adversarial_specs))
    column = dataset.adversarial_columns[name]
    observed_column = dataset.adversarial_observed_columns[name]

    from dataset.gleason_cls import is_missing, is_observed_value

    observed = dataset.df[observed_column].apply(is_observed_value)
    usable = observed & ~dataset.df[column].apply(is_missing)
    values = dataset.df.loc[usable, column].astype(int)
    counts = values.value_counts().to_dict()
    weights = []
    for _, row in dataset.df.iterrows():
        value = row[column]
        if not is_observed_value(row[observed_column]) or is_missing(value):
            weights.append(0.0)
        else:
            weights.append(1.0 / counts[int(value)])
    return torch.as_tensor(weights, dtype=torch.double), int(usable.sum())


def build_loaders(args):
    from dataset.gleason_cls import GleasonClassificationDataset, get_gleason_transforms

    train_transforms, val_transforms, test_transforms = get_gleason_transforms(args)
    train_set = GleasonClassificationDataset(
        args, csv_for_phase(args, "train"), "train", train_transforms
    )
    val_set = GleasonClassificationDataset(
        args, csv_for_phase(args, "val"), "val", val_transforms
    )
    test_set = GleasonClassificationDataset(
        args, csv_for_phase(args, "test"), "test", test_transforms
    )

    sampler = None
    shuffle = True
    num_samples = len(train_set)
    if args.adversarial_weighted_sampling:
        weights, observed_count = adversarial_sampler_weights(train_set)
        if observed_count == 0:
            raise ValueError("No observed adversarial labels available for weighted sampling.")
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=observed_count,
            replacement=True,
        )
        shuffle = False

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=args.num_workers,
        drop_last=args.drop_last,
        pin_memory=args.pin_mem,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=args.pin_mem,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=args.pin_mem,
    )
    return train_loader, val_loader, test_loader


def build_model_and_optimizer(args, device):
    from models.build_gleason_classification import build_encoder

    encoder = build_encoder(args)
    model = DirectAdversarialClassifier(
        encoder=encoder,
        adversarial_num_classes=args.adversarial_specs_resolved,
        bottleneck_dim=args.bottleneck_dim,
    ).to(device)

    head_params = list(model.adversarial_heads.parameters())
    if args.train == "freeze":
        for param in model.encoder.parameters():
            param.requires_grad = False
        optimizer = torch.optim.AdamW(head_params, lr=args.lr, weight_decay=0.0)
    elif args.model == "profound_conv":
        from util.convnext_optim import LayerDecayValueAssigner, get_parameter_groups

        num_layers = sum(model.encoder.depths)
        assigner = LayerDecayValueAssigner(
            [args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)],
            depths=model.encoder.depths,
            layer_decay_type=args.layer_decay_type,
        )
        skip = model.encoder.no_weight_decay() if hasattr(model.encoder, "no_weight_decay") else {}
        backbone_groups = get_parameter_groups(
            model.encoder,
            args.weight_decay,
            skip,
            assigner.get_layer_id,
            assigner.get_scale,
        )
        head_group = {"params": head_params, "weight_decay": 0.0, "lr": args.lr}
        optimizer = torch.optim.AdamW(backbone_groups + [head_group], lr=args.lr)
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model = %s" % str(model))
    print("number of trainable params (M): %.2f" % (trainable / 1.0e6))
    print(f"adversarial heads: {args.adversarial_specs_resolved}")
    return model, optimizer


def adversarial_loss(model_output, adversarial_targets, adversarial_masks, args):
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
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Direct Adv Epoch: [{epoch}]"
    optimizer.zero_grad(set_to_none=True)

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, 20, header)):
        img, _, adversarial_targets, adversarial_masks, _ = unpack_batch(batch)
        img = img.to(device, non_blocking=True)
        lr_sched.adjust_learning_rate(
            optimizer, data_iter_step / len(data_loader) + epoch, args
        )
        model_output = model(img)
        loss, losses_by_name, counts_by_name = adversarial_loss(
            model_output, adversarial_targets, adversarial_masks, args
        )
        if loss is None:
            metric_logger.update(skipped_batches=1)
            continue
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            raise FloatingPointError(f"Non-finite direct adversarial loss: {loss_value}")

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
            log_writer.add_scalar("train_direct_adversarial_loss", loss_value, step)
            log_writer.add_scalar("lr", lr, step)

    print("Averaged direct adversarial stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, data_loader, device, args, phase):
    model.eval()
    losses = []
    logits_by_name = {}
    targets_by_name = {}
    observed_counts = {}

    for batch in data_loader:
        img, _, adversarial_targets, adversarial_masks, _ = unpack_batch(batch)
        img = img.to(device, non_blocking=True)
        model_output = model(img)
        loss, _, counts = adversarial_loss(
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
    print(f"{phase} direct adversarial stats: {stats}")
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


def save_checkpoint(args, model, optimizer, epoch, filename):
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
    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True

    args = configure_args(args)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader = build_loaders(args)
    print("{}".format(args).replace(", ", ",\n"))
    print(f"label definition used only for dataset loading: {args.label_definition}")
    print(f"adversarial definition: {args.adversarial_definition}")
    print(
        f"train batches: {len(train_loader)}, "
        f"val batches: {len(val_loader)}, test batches: {len(test_loader)}"
    )
    for split_name, loader in [
        ("train", train_loader),
        ("val", val_loader),
        ("test", test_loader),
    ]:
        counts_fn = getattr(loader.dataset, "adversarial_observed_counts", None)
        if counts_fn is not None:
            print(f"{split_name} adversarial observed counts: {counts_fn()}")

    model, optimizer = build_model_and_optimizer(args, device)
    log_writer = SummaryWriter(log_dir=args.log_dir)
    start_time = time.time()
    best = np.inf if args.primary_metric == "loss" else -np.inf
    best_epoch = -1

    for epoch in range(args.epochs):
        train_stats = train_one_epoch(
            model, train_loader, optimizer, device, epoch, log_writer, args
        )
        misc.write_log(
            log_writer,
            {**{f"train_{k}": v for k, v in train_stats.items()}, "epoch": epoch},
            args,
        )

        if epoch % args.val_interval == 0 or epoch + 1 == args.epochs:
            val_stats = evaluate(model, val_loader, device, args, phase="val")
            value = metric_value(val_stats, args)
            best_flag = is_better(value, best, args)
            if best_flag:
                best = value
                best_epoch = epoch
            save_checkpoint(args, model, optimizer, epoch, "last.pth.tar")
            if best_flag:
                save_checkpoint(args, model, optimizer, epoch, "best.pth.tar")
            if args.save_ckpt_interval > 0 and (
                (epoch + 1) % args.save_ckpt_interval == 0
                or epoch + 1 == args.epochs
            ):
                save_checkpoint(args, model, optimizer, epoch, f"checkpoint-{epoch}.pth")
            misc.write_log(
                log_writer,
                {
                    **{f"val_{k}": v for k, v in val_stats.items()},
                    "epoch": epoch,
                    "primary_metric": args.primary_metric,
                    "primary_metric_value": value,
                    "is_best": best_flag,
                    "best_epoch": best_epoch,
                    "adversarial_definition": args.adversarial_definition,
                },
                args,
            )

    total_time = time.time() - start_time
    print(
        "Direct adversarial training time {}".format(
            str(datetime.timedelta(seconds=int(total_time)))
        )
    )
    best_path = Path(args.output_dir) / "best.pth.tar"
    if best_path.exists():
        checkpoint = misc.load_trusted_checkpoint(best_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
    test_stats = evaluate(model, test_loader, device, args, phase="test")
    misc.write_log(
        log_writer,
        {
            **{f"test_{k}": v for k, v in test_stats.items()},
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
