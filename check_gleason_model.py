import argparse
from types import SimpleNamespace

import torch

from dataset.gleason_cls import build_gleason_classification_loaders, parse_int_tuple
from engine.gleason_classification import (
    adversarial_loss,
    criterion_for_task,
    main_logits,
    prepare_targets,
    unpack_batch,
)
from models.build_gleason_classification import build_gleason_model
from util.gleason_config import ADVERSARIAL_VARIABLE_CHOICES
from util.paths import DATA_ROOT, GLEASON_CLASSIFICATION_CSV


def get_args_parser():
    parser = argparse.ArgumentParser("Check Gleason model forward/backward")
    parser.add_argument("--csv_path", default=str(GLEASON_CLASSIFICATION_CSV), type=str)
    parser.add_argument("--train_csv", default=None, type=str)
    parser.add_argument("--val_csv", default=None, type=str)
    parser.add_argument("--test_csv", default=None, type=str)
    parser.add_argument("--data_root", default=str(DATA_ROOT), type=str)
    parser.add_argument("--split_col", default="split", type=str)
    parser.add_argument("--image_path_col", default="image_npy_path", type=str)
    parser.add_argument("--in_channels", default=3, type=int)
    parser.add_argument("--crop_spatial_size", default=(64, 256, 256), type=parse_int_tuple)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--max_batches_to_find_observed_adversary", default=32, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--pin_mem", action="store_true")
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=False)
    parser.add_argument("--drop_last", action="store_true")
    parser.set_defaults(drop_last=False)
    parser.add_argument("--weighted_sampling", action="store_true")
    parser.set_defaults(weighted_sampling=False)

    parser.add_argument("--task_type", choices=["binary", "ordinal"], required=True)
    parser.add_argument("--label_col", default="grade_group", type=str)
    parser.add_argument("--binary_label_col", default=None, type=str)
    parser.add_argument("--binary_positive_min", default=2, type=int)
    parser.add_argument("--ordinal_levels", default=5, type=int)
    parser.add_argument("--label_offset", default=1, type=int)

    parser.add_argument("--adversarial_specs", default="", type=str)
    parser.add_argument(
        "--adversarial_variable",
        default=None,
        choices=ADVERSARIAL_VARIABLE_CHOICES,
        type=str,
    )
    parser.add_argument("--adversarial_column", default=None, type=str)
    parser.add_argument("--adversarial_observed_column", default=None, type=str)
    parser.add_argument("--adversarial_num_classes", default=None, type=int)
    parser.add_argument("--adversarial_loss_weight", default=1.0, type=float)
    parser.add_argument("--grl_lambda", default=1.0, type=float)
    parser.add_argument(
        "--grl_schedule",
        choices=["constant", "dann"],
        default="constant",
        type=str,
    )
    parser.add_argument("--grl_gamma", default=10.0, type=float)

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
    parser.add_argument("--device", default="cuda", type=str)

    parser.add_argument("--weight_decay", default=1e-5, type=float)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--min_lr", default=0.0, type=float)
    parser.add_argument("--warmup_epochs", default=10, type=int)
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--layer_decay", default=0.6, type=float)
    parser.add_argument(
        "--layer_decay_type",
        choices=["single", "group"],
        default="group",
        type=str,
    )
    return parser


def move_adversarial_dict(batch_dict, device):
    return {
        name: value.to(device, non_blocking=True)
        for name, value in batch_dict.items()
    }


def choose_check_batch(train_loader, args):
    if args.adversarial_variable is None:
        return next(iter(train_loader))

    fallback = None
    for step, batch in enumerate(train_loader):
        if fallback is None:
            fallback = batch
        _, _, _, adversarial_masks, _ = unpack_batch(batch)
        mask = adversarial_masks.get(args.adversarial_variable)
        if mask is not None and mask.bool().any():
            print(
                f"Using batch {step} with observed adversarial target "
                f"for {args.adversarial_variable}"
            )
            return batch
        if step + 1 >= args.max_batches_to_find_observed_adversary:
            break

    print(
        "No observed adversarial target found within "
        f"{args.max_batches_to_find_observed_adversary} batch(es); "
        "checking Gleason path and adversarial-head construction only."
    )
    return fallback


def main(args):
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)

    train_loader, _, _ = build_gleason_classification_loaders(args)
    batch = choose_check_batch(train_loader, args)
    image, labels, adversarial_targets, adversarial_masks, sample_idx = unpack_batch(batch)
    image = image.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    adversarial_targets = move_adversarial_dict(adversarial_targets, device)
    adversarial_masks = move_adversarial_dict(adversarial_masks, device)

    print(f"Input image shape: {tuple(image.shape)}")
    print(f"Labels shape: {tuple(labels.shape)}, labels: {labels.detach().cpu().tolist()}")
    print(f"Sample indices: {sample_idx.detach().cpu().tolist()}")
    if adversarial_targets:
        print(f"Adversarial targets: { {k: v.detach().cpu().tolist() for k, v in adversarial_targets.items()} }")
        print(f"Adversarial masks: { {k: v.detach().cpu().tolist() for k, v in adversarial_masks.items()} }")

    model, optimizer = build_gleason_model(args=args, device=device)
    model.train()
    output = model(image)
    logits = main_logits(output)
    targets = prepare_targets(labels, args)
    main_loss = criterion_for_task(args)(logits, targets)
    adv_loss, adv_losses = adversarial_loss(
        output,
        adversarial_targets,
        adversarial_masks,
        args,
    )
    loss = main_loss if adv_loss is None else main_loss + adv_loss
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    print(f"Main logits shape: {tuple(logits.shape)}")
    print(f"Main loss: {float(main_loss.detach().cpu()):.6f}")
    if adv_loss is not None:
        print(f"Adversarial loss: {float(adv_loss.detach().cpu()):.6f}")
        print(
            "Adversarial loss terms: "
            + str({name: float(value.detach().cpu()) for name, value in adv_losses.items()})
        )
    print(f"Total loss: {float(loss.detach().cpu()):.6f}")
    print("Model check passed.")


if __name__ == "__main__":
    parsed = get_args_parser().parse_args()
    main(SimpleNamespace(**vars(parsed)))
