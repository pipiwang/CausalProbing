import os

import torch

from models.convnextv2 import convnextv2_tiny, load_state_dict, remap_checkpoint_keys
from models.gleason_classifier import GleasonClassifier
from models.resnet18 import resnet18_3D
from util.convnext_optim import LayerDecayValueAssigner, get_parameter_groups
from util.gleason_config import resolve_adversarial_config
from util.lars import LARS
from util.paths import PROFOUND_CONV_CHECKPOINT


def default_pretrain_path(model_name):
    if model_name == "profound_conv":
        return str(PROFOUND_CONV_CHECKPOINT)
    return None


def load_trusted_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def build_encoder(args):
    if args.model == "resnet18":
        return resnet18_3D(args.in_channels)

    if args.model == "profound_conv":
        encoder = convnextv2_tiny(in_chans=args.in_channels, drop_path_rate=0.1)
        if args.pretrain is None and args.train != "scratch":
            args.pretrain = default_pretrain_path(args.model)
        if args.pretrain is None:
            return encoder
        if not os.path.exists(args.pretrain):
            raise FileExistsError(f"{args.pretrain} Not exists")
        ckpt = load_trusted_checkpoint(args.pretrain)
        ckpt = remap_checkpoint_keys(ckpt)
        load_state_dict(encoder, ckpt)
        return encoder

    raise NotImplementedError(f"unknown model: {args.model}")


def build_gleason_model(args, device):
    adversarial_num_classes = resolve_adversarial_config(args)
    encoder = build_encoder(args)
    model = GleasonClassifier(
        encoder=encoder,
        main_out_dim=args.num_classes,
        adversarial_num_classes=adversarial_num_classes,
        bottleneck_dim=args.bottleneck_dim,
        grl_lambda=args.grl_lambda,
    ).to(device)

    if args.model == "profound_conv" and args.train == "freeze":
        for _, value in model.encoder.named_parameters():
            value.requires_grad = False
        optimizer = LARS(model.head_parameters(), weight_decay=0, lr=args.lr)
    elif args.model == "profound_conv":
        num_layers = sum(model.encoder.depths)
        assigner = LayerDecayValueAssigner(
            [args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)],
            depths=model.encoder.depths,
            layer_decay_type=args.layer_decay_type,
        )

        skip = {}
        if hasattr(model.encoder, "no_weight_decay"):
            skip = model.encoder.no_weight_decay()

        backbone_param_groups = get_parameter_groups(
            model.encoder,
            args.weight_decay,
            skip,
            assigner.get_layer_id,
            assigner.get_scale,
        )
        head_param_groups = [
            {"params": model.head_parameters(), "weight_decay": 0.0, "lr": args.lr}
        ]
        optimizer = torch.optim.AdamW(backbone_param_groups + head_param_groups, lr=args.lr)
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), args.lr, weight_decay=args.weight_decay
        )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model = %s" % str(model))
    print("number of params (M): %.2f" % (n_parameters / 1.0e6))
    if adversarial_num_classes:
        print(f"adversarial heads: {adversarial_num_classes}")
    return model, optimizer
