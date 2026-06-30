from models.unetr import UNETR3D, UNETR3D_prompt
from models.vit import vit_base_patch16_dec512d8b
from models.convnextv2 import convnextv2_tiny, remap_checkpoint_keys, load_state_dict
from models.convnext_unter import ConvnextUNETR
from models.upernet_module import UperNet, ViTAdapter
import monai
import torch
import os
import timm.optim.optim_factory as optim_factory
from util.lr_decay import param_groups_lrd
from util.convnext_optim import get_parameter_groups, LayerDecayValueAssigner

def buidl_model(args, device):
    if args.model == "unet":
        model = monai.networks.nets.UNet(
            spatial_dims=3,
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )
        model = model.to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), args.lr, weight_decay=args.weight_decay
        )

    elif args.model == "profound_conv_unetr3d":
        if args.pretrain is None:
            raise NotImplementedError(f"No pretrained weight")
        if not os.path.exists(args.pretrain):
            raise FileExistsError(f"{args.pretrain} Not exists")

        convnext = convnextv2_tiny(in_chans=3)
        ckpt = torch.load(args.pretrain, map_location="cpu")
        ckpt = remap_checkpoint_keys(ckpt)
        load_state_dict(convnext, ckpt)
        model = ConvnextUNETR(
            in_channels=args.in_channels, out_channels=args.out_channels, convnext=convnext, feature_size=32
        )
        model = model.to(device)

        if args.train == "freeze":
            for key, value in model.encoder.named_parameters():
                value.requires_grad = False
            optimizer = torch.optim.AdamW(
                model.decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay
            )
        else:
            num_layers = sum(convnext.depths)
            assigner = LayerDecayValueAssigner(
                list(
                    args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)
                ),
                depths=convnext.depths,
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
            decoder_param_groups = decoder_parameters(
                model.decoder, args.weight_decay, lr=args.lr
            )
            optimizer = torch.optim.AdamW(
                backbone_param_groups + decoder_param_groups,
                lr=args.lr
            )

    elif args.model == "profound_conv":
        if args.pretrain is None:
            raise NotImplementedError(f"No pretrained weight")
        if not os.path.exists(args.pretrain):
            raise FileExistsError(f"{args.pretrain} Not exists")

        convnext = convnextv2_tiny(in_chans=3, drop_path_rate=0.1)
        ckpt = torch.load(args.pretrain, map_location="cpu")
        ckpt = remap_checkpoint_keys(ckpt)
        load_state_dict(convnext, ckpt)

        model = UperNet(
            encoder=convnext,
            in_channels=[96, 192, 384, 768],
            out_channels=args.out_channels,
        )
        model = model.to(device)

        if args.train == "freeze":
            for key, value in model.encoder.named_parameters():
                value.requires_grad = False
            optimizer = torch.optim.AdamW(
                model.decode_head.parameters(), lr=args.lr, weight_decay=args.weight_decay
            )
        else:
            num_layers = sum(convnext.depths)
            assigner = LayerDecayValueAssigner(
                list(
                    args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)
                ),
                depths=convnext.depths,
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
            decoder_param_groups = decoder_parameters(
                model.decode_head, args.weight_decay, lr=args.lr
            ) + decoder_parameters(model.auxiliary_head, args.weight_decay, lr=args.lr)
            # decoder_param_groups = decoder_parameters(
            #     model.decoder, args.weight_decay, lr=args.lr
            # )
            optimizer = torch.optim.AdamW(
                backbone_param_groups + decoder_param_groups,
                lr=args.lr,
                weight_decay=args.weight_decay,
            )

    elif args.model == "profound_vit_unetr3d":
        vit = vit_base_patch16_dec512d8b(in_chans=args.in_channels)
        if args.pretrain is None:
            raise NotImplementedError(f"No pretrained weight")
        if not os.path.exists(args.pretrain):
            raise FileExistsError(f"{args.pretrain} Not exists")

        msg = vit.load_state_dict(
            torch.load(args.pretrain, map_location="cpu")["model"], strict=False
        )
        print(msg.missing_keys)
        if args.prompt:
            model = UNETR3D_prompt(
                encoder=vit,
                in_channels=args.in_channels,
                out_channels=args.out_channels,
                img_size=args.crop_spatial_size,
                patch_size=(16, 32, 32),
                feature_size=32,
            )
            model = model.to(device)
            optimizer = torch.optim.AdamW(
                optim_factory.param_groups_weight_decay(model),
                lr=args.lr,
                betas=(0.9, 0.95),
                eps=1e-3,
            )
        else:
            model = UNETR3D(
                encoder=vit,
                in_channels=args.in_channels,
                out_channels=args.out_channels,
                img_size=args.crop_spatial_size,
                patch_size=(16, 32, 32),
                feature_size=32,
            )
            model = model.to(device)
            backbone_param_groups = param_groups_lrd(
                model.encoder, args.weight_decay, layer_decay=args.layer_decay
            )
            decoder_param_groups = decoder_parameters(
                model.decoder, args.weight_decay, lr=args.lr
            )
            optimizer = torch.optim.AdamW(
                backbone_param_groups + decoder_param_groups, lr=args.lr
            )

    elif args.model == "profound_vit":
        vit = vit_base_patch16_dec512d8b(in_chans=args.in_channels)
        if args.pretrain is None:
            raise NotImplementedError(f"No pretrained weight")
        if not os.path.exists(args.pretrain):
            raise FileExistsError(f"{args.pretrain} Not exists")

        msg = vit.load_state_dict(
            torch.load(args.pretrain, map_location="cpu")["model"], strict=False
        )
        print(msg.missing_keys)

        adapter = ViTAdapter(
            img_size=args.crop_spatial_size,
            patch_size=(16, 32, 32),
            embed_dim=768,
        )
        model = UperNet(
            encoder=vit,
            in_channels=[768] * 4,
            out_channels=args.out_channels,
            adapter=adapter,
            out_indices=[3, 5, 7, 11],
        )
        model = model.to(device)
        backbone_param_groups = param_groups_lrd(model.encoder, args.weight_decay)
        decoder_param_groups = (
            decoder_parameters(model.adapter, args.weight_decay, lr=args.lr)
            + decoder_parameters(model.decode_head, args.weight_decay, lr=args.lr)
            + decoder_parameters(model.auxiliary_head, args.weight_decay, lr=args.lr)
        )
        optimizer = torch.optim.AdamW(
            backbone_param_groups + decoder_param_groups,
            lr=args.lr,
        )

    else:
        raise NotImplementedError(f"unknown model: {args.model}")

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model))
    print("number of params (M): %.2f" % (n_parameters / 1.0e6))

    return model, optimizer


def param_groups_weight_decay(
    model: torch.nn.Module, weight_decay=1e-5, no_weight_decay_list=(), lr=1e-3
):
    no_weight_decay_list = set(no_weight_decay_list)
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.ndim <= 1 or name.endswith(".bias") or name in no_weight_decay_list:
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]


def decoder_parameters(
    model: torch.nn.Module, weight_decay=1e-5, no_weight_decay_list=(), lr=1e-3
):
    no_weight_decay_list = set(no_weight_decay_list)
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.ndim <= 1 or name.endswith(".bias") or name in no_weight_decay_list:
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {"params": no_decay, "weight_decay": 0.0, "lr": lr},
        {"params": decay, "weight_decay": weight_decay, "lr": lr},
    ]
