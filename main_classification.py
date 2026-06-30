# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import os
import argparse
import datetime
import json
import numpy as np
import time
from pathlib import Path
from typing import Callable, List, Optional, Tuple
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from dataset.dataset_cls import build_Risk_loader, build_promis567_loader,build_Screening_loader, build_Promis_loader,build_Promis3_hist_loader,build_Promis3_gg_loader
import timm.optim.optim_factory as optim_factory
import util.misc as misc
import torch.distributed as dist
from models.build_classifition import buidl_model
from engine.classification import train_one_epoch, validation, test
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.misc import write_log
from util.paths import CLASSIFICATION_LOG_DIR, CLASSIFICATION_OUTPUT_DIR, DATA_ROOT

def tuple_type(strings):
    strings = strings.replace("(", "").replace(")", "")
    mapped_int = map(int, strings.split(","))
    return tuple(mapped_int)

def get_args_parser():
    parser = argparse.ArgumentParser("classification", add_help=False)
    parser.add_argument(
        "--batch_size",
        default=1,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )
    parser.add_argument("--epochs", default=400, type=int)
    parser.add_argument(
        "--root", default=str(DATA_ROOT), type=str
    )
    parser.add_argument(
        "--input_size",
        default=(64, 256, 256),
        type=tuple_type,
        help="images input size",
    )
    parser.add_argument("--crop_spatial_size", default=(64, 256, 256), type=tuple_type)

    # Model parameters
    parser.add_argument("--model", help="model name")
    parser.add_argument(
        "--train", choices=["fintune", "freeze", "scratch"], help="train method"
    )
    parser.add_argument("--pretrain", default=None, type=str)
    parser.add_argument("--tolerance", default=5, type=int)
    parser.add_argument("--spacing", default=(1.0, 0.5, 0.5), type=tuple)
    # Optimizer parameters
    parser.add_argument(
        "--weight_decay", type=float, default=1e-5, help="weight decay (default: 1e-5)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        metavar="LR",
        help="learning rate (absolute lr)",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=0.0,
        metavar="LR",
        help="lower lr bound for cyclic schedulers that hit 0",
    )
    parser.add_argument(
        "--warmup_epochs", type=int, default=40, metavar="N", help="epochs to warmup LR"
    )

    # Dataset parameters
    parser.add_argument(
        "--output_dir",
        default=str(CLASSIFICATION_OUTPUT_DIR),
        help="path where to save, empty for no saving",
    )
    parser.add_argument(
        "--log_dir",
        default=str(CLASSIFICATION_LOG_DIR),
        help="path where to tensorboard log",
    )
    parser.add_argument("--dataset", default="UCL", help="dataset name")
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")

    parser.add_argument("--layer_decay", type=float, default=0.6)
    parser.add_argument(
        "--layer_decay_type",
        type=str,
        choices=["single", "group"],
        default="group",
        help="""Layer decay strategies. The single strategy assigns a distinct decaying value for each layer,
                        whereas the group strategy assigns the same decaying value for three consecutive layers""",
    )

    parser.add_argument("--kfold", type=int, default=None)
    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    parser.add_argument("--data20", action="store_true", help="Use 20 training data")
    parser.set_defaults(data20=False)

    parser.add_argument("--data_num", default=0, type=int, help="number of train data")

    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )
    parser.add_argument("--val_interval", default=1, type=int)
    parser.add_argument("--min_mem", type=int, default=2, help="Minimum GPU memory required (in GB)")
    return parser


def main(args):
    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    device = "cuda"
    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    if args.data20 or args.data_num > 0:
        args.log_dir = os.path.join(
            args.log_dir, args.dataset, args.model, args.train, str(args.epochs), str(args.lr), str(args.seed)
            )
        args.output_dir = os.path.join(
            args.output_dir, args.dataset, args.model, args.train, str(args.epochs), str(args.lr), str(args.seed)
        )
    else: 
        args.log_dir = os.path.join(
            args.log_dir, args.dataset, args.model, args.train, str(args.seed)
        )
        args.output_dir = os.path.join(
            args.output_dir, args.dataset, args.model, args.train, str(args.seed)
        )
        
    if args.kfold is not None:
        args.log_dir = os.path.join(args.log_dir, f"fold{args.kfold}")
        args.output_dir = os.path.join(args.output_dir, f"fold{args.kfold}")
    if args.dataset == "risk":
        data_loader_train, data_loader_val, data_loader_test = build_Risk_loader(args)
    elif args.dataset == "screening":
        data_loader_train, data_loader_val, data_loader_test = build_Screening_loader(args)
    elif args.dataset == "promis":
        data_loader_train, data_loader_val, data_loader_test = build_Promis_loader(args)
    elif args.dataset == "promis3hist":
        data_loader_train, data_loader_val, data_loader_test = build_Promis3_hist_loader(args)
    elif args.dataset == "promis3gg":
        data_loader_train, data_loader_val, data_loader_test = build_Promis3_gg_loader(args)
    elif args.dataset == "promis567":
        data_loader_train, data_loader_val, data_loader_test = build_promis567_loader(args)
    else:
        raise NotImplementedError(f"unknown schedule sampler: {args.dataset}")

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    log_writer = SummaryWriter(log_dir=args.log_dir)

    # define the model
    model, optimizer = buidl_model(args=args, device=device)

    # print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(
        args=args, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scaler
    )

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    best_loss = np.inf

    print(f"train data: {len(data_loader_train)}, val data: {len(data_loader_val)}, test data: {len(data_loader_test)}")
    
    save_epochs = [10,20,50,100]
    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(
            model,
            data_loader_train,
            optimizer,
            device,
            epoch,
            None,
            log_writer=log_writer,
            args=args,
        )
        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            "epoch": epoch,
        }

        write_log(log_writer, log_stats, args)

        if args.output_dir and (epoch % args.val_interval == 0 or epoch + 1 == args.epochs):
            epoch_loss = validation(model, data_loader_val, device, epoch, args)
            is_best = False
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                is_best = True
            if args.data_num == 0:
                misc.save_best_model(
                    args=args,
                    model=model,
                    model_without_ddp=model,
                    optimizer=optimizer,
                    loss_scaler=loss_scaler,
                    epoch=epoch,
                    is_best=is_best,
                )
            else:
                for epoch_interval in save_epochs:
                    if epoch < epoch_interval:
                        current_interval = epoch_interval
                        break
                misc.save_current_best_model(
                    args=args,
                    model=model,
                    model_without_ddp=model,
                    optimizer=optimizer,
                    loss_scaler=loss_scaler,
                    epoch=epoch,
                    is_best=is_best,
                    current_interval=current_interval,
                )

            val_stats = {
                "val_loss": str(epoch_loss),
                "epoch": epoch,
                "is_best": is_best,
            }
            write_log(log_writer, val_stats, args)
        

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))
    test_stats = test(model=model, test_loader=data_loader_test, args=args)
    write_log(log_writer, test_stats, args)


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
