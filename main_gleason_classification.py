import argparse
import datetime
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import util.misc as misc
from dataset.gleason_cls import build_gleason_classification_loaders, parse_int_tuple
from engine.gleason_classification import test, train_one_epoch, validation
from models.build_gleason_classification import build_gleason_model
from util.gleason_config import ADVERSARIAL_VARIABLE_CHOICES, resolve_adversarial_config
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.misc import write_log
from util.paths import (
    CLASSIFICATION_LOG_DIR,
    CLASSIFICATION_OUTPUT_DIR,
    DATA_ROOT,
    GLEASON_CLASSIFICATION_CSV,
)


def get_args_parser():
    parser = argparse.ArgumentParser("Gleason classification", add_help=False)

    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument("--pin_mem", action="store_true")
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)
    parser.add_argument("--drop_last", action="store_true")
    parser.set_defaults(drop_last=False)

    parser.add_argument("--data_root", default=str(DATA_ROOT), type=str)
    parser.add_argument("--csv_path", default=str(GLEASON_CLASSIFICATION_CSV), type=str)
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
        help="Advanced: one already-binned adversarial column, e.g. bmi_bin:4",
    )
    parser.add_argument(
        "--adversarial_variable",
        default=None,
        choices=ADVERSARIAL_VARIABLE_CHOICES,
        type=str,
        help="One variable to rule out in this run.",
    )
    parser.add_argument(
        "--adversarial_column",
        default=None,
        type=str,
        help="CSV column for the chosen binned variable. Defaults to <variable>_bin.",
    )
    parser.add_argument(
        "--adversarial_observed_column",
        default=None,
        type=str,
        help="CSV observed-mask column. Defaults to <variable>_observed.",
    )
    parser.add_argument("--adversarial_num_classes", default=None, type=int)
    parser.add_argument("--adversarial_loss_weight", default=1.0, type=float)
    parser.add_argument("--grl_lambda", default=1.0, type=float)
    parser.add_argument(
        "--grl_schedule",
        choices=["constant", "dann"],
        default="constant",
        type=str,
        help="Schedule for the gradient reversal strength.",
    )
    parser.add_argument(
        "--grl_gamma",
        default=10.0,
        type=float,
        help="Gamma for the DANN schedule: 2/(1+exp(-gamma*p))-1.",
    )

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
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--bottleneck_dim", default=256, type=int)

    parser.add_argument("--weight_decay", default=1e-5, type=float)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--min_lr", default=0.0, type=float)
    parser.add_argument("--warmup_epochs", default=10, type=int)
    parser.add_argument("--layer_decay", default=0.6, type=float)
    parser.add_argument(
        "--layer_decay_type",
        choices=["single", "group"],
        default="group",
        type=str,
    )

    parser.add_argument("--output_dir", default=str(CLASSIFICATION_OUTPUT_DIR), type=str)
    parser.add_argument("--log_dir", default=str(CLASSIFICATION_LOG_DIR), type=str)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--val_interval", default=1, type=int)
    parser.add_argument("--save_ckpt_interval", default=10, type=int)
    parser.add_argument(
        "--primary_metric",
        default=None,
        choices=["loss", "auc", "balanced_acc", "qwk"],
        type=str,
        help="Metric used for best.pth.tar. Defaults to auc for binary and qwk for ordinal.",
    )
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://", type=str)
    return parser


def main(args):
    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True

    args.crop_spatial_size = tuple(args.crop_spatial_size)
    resolve_adversarial_config(args)
    if args.task_type == "binary":
        definition_name = (
            args.binary_label_col
            if args.binary_label_col
            else f"{args.label_col}_ge_{args.binary_positive_min}"
        )
    else:
        definition_name = f"{args.label_col}_ordinal_{args.ordinal_levels}levels"
    adversarial_name = "ruleout_none"
    if args.adversarial_specs_resolved:
        adversarial_name = f"ruleout_{next(iter(args.adversarial_specs_resolved))}"
    args.log_dir = os.path.join(
        args.log_dir,
        "gleason",
        args.task_type,
        definition_name,
        adversarial_name,
        args.model,
        args.train,
        str(args.seed),
    )
    args.output_dir = os.path.join(
        args.output_dir,
        "gleason",
        args.task_type,
        definition_name,
        adversarial_name,
        args.model,
        args.train,
        str(args.seed),
    )
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

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

    log_writer = SummaryWriter(log_dir=args.log_dir)
    model, optimizer = build_gleason_model(args=args, device=device)
    loss_scaler = NativeScaler()
    misc.load_model(
        args=args, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scaler
    )

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    primary_metric = args.primary_metric or (
        "qwk" if args.task_type == "ordinal" else "auc"
    )
    best_metrics = {
        "loss": np.inf,
        "auc": -np.inf,
        "balanced_acc": -np.inf,
        "qwk": -np.inf,
    }

    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(
            model,
            data_loader_train,
            optimizer,
            device,
            epoch,
            log_writer=log_writer,
            args=args,
        )
        write_log(
            log_writer,
            {**{f"train_{k}": v for k, v in train_stats.items()}, "epoch": epoch},
            args,
        )

        if epoch % args.val_interval == 0 or epoch + 1 == args.epochs:
            val_loss, val_stats = validation(model, data_loader_val, device, epoch, args)
            best_flags = {}
            for metric in ["loss", "auc", "balanced_acc", "qwk"]:
                value = val_loss if metric == "loss" else val_stats.get(metric)
                if value is None or not np.isfinite(value):
                    best_flags[metric] = False
                    continue
                if metric == "loss":
                    is_best = value < best_metrics[metric]
                else:
                    is_best = value > best_metrics[metric]
                if is_best:
                    best_metrics[metric] = value
                best_flags[metric] = is_best
                if metric in val_stats or metric == "loss":
                    misc.save_best_model(
                        args=args,
                        model=model,
                        model_without_ddp=model,
                        optimizer=optimizer,
                        loss_scaler=loss_scaler,
                        epoch=epoch,
                        is_best=is_best,
                        best_name=f"best_{metric}",
                        update_overall_best=metric == primary_metric,
                    )
            if args.save_ckpt_interval > 0 and (
                (epoch + 1) % args.save_ckpt_interval == 0
                or epoch + 1 == args.epochs
            ):
                misc.save_model(
                    args=args,
                    epoch=epoch,
                    model=model,
                    model_without_ddp=model,
                    optimizer=optimizer,
                    loss_scaler=loss_scaler,
                )
            write_log(
                log_writer,
                {
                    **{f"val_{k}": v for k, v in val_stats.items()},
                    "epoch": epoch,
                    "primary_metric": primary_metric,
                    **{f"is_best_{k}": v for k, v in best_flags.items()},
                    "label_definition": args.label_definition,
                    "adversarial_definition": args.adversarial_definition,
                },
                args,
            )

    total_time = time.time() - start_time
    print("Training time {}".format(str(datetime.timedelta(seconds=int(total_time)))))
    test_stats = test(model=model, test_loader=data_loader_test, args=args)
    write_log(
        log_writer,
        {
            **{f"test_{k}": v for k, v in test_stats.items()},
            "label_definition": args.label_definition,
            "adversarial_definition": args.adversarial_definition,
        },
        args,
    )


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
