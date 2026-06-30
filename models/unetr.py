import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Sequence, Tuple, Union

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import (
    UnetrBasicBlock,
    UnetrPrUpBlock,
    UnetrUpBlock,
)


class UNETR_decoder(nn.Module):
    """
    UNETR based on: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: Tuple[int] = (96, 96, 96),
        patch_size: Tuple[int] = (16, 16, 16),
        feature_size: int = 16,
        hidden_size: int = 768,
        norm_name: Union[Tuple, str] = "instance",
        conv_block: bool = True,
        res_block: bool = True,
        pool: bool = False,
        spatial_dims: int = 3,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimension of hidden layer.
            norm_name: feature normalization type and arguments.
            conv_block: bool argument to determine if convolutional block is used.
            res_block: bool argument to determine if residual block is used.
            spatial_dims: number of spatial dims.

        Examples::

            # for single channel input 4-channel output with image size of (96,96,96), feature size of 32 and batch norm
            >>> net = UNETR(in_channels=1, out_channels=4, img_size=(96,96,96), feature_size=32, norm_name='batch')

             # for single channel input 4-channel output with image size of (96,96), feature size of 32 and batch norm
            >>> net = UNETR(in_channels=1, out_channels=4, img_size=96, feature_size=32, norm_name='batch', spatial_dims=2)

        """

        super().__init__()

        img_size = img_size
        patch_size = patch_size

        self.grid_size = tuple(img_d // p_d for img_d, p_d in zip(img_size, patch_size))
        self.pool = pool
        self.hidden_size = hidden_size

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=(1, 2, 2),
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 2,
            num_layer=2,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 4,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder4 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.out = UnetOutBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=out_channels,
        )

    def proj_feat(self, x):
        new_view = (x.size(0), *self.grid_size, self.hidden_size)
        x = x.view(new_view)
        new_axes = (0, len(x.shape) - 1) + tuple(d + 1 for d in range(len(self.grid_size)))
        x = x.permute(new_axes).contiguous()
        if self.pool:
            x = F.adaptive_avg_pool3d(x, (4, 8, 8))
        return x

    def forward(self, x_in, x, hidden_states_out):
        enc1 = self.encoder1(x_in)
        x2 = hidden_states_out[3]
        enc2 = self.encoder2(self.proj_feat(x2))
        x3 = hidden_states_out[6]
        enc3 = self.encoder3(self.proj_feat(x3))
        x4 = hidden_states_out[9]
        enc4 = self.encoder4(self.proj_feat(x4))
        dec4 = self.proj_feat(x)
        dec3 = self.decoder5(dec4, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        out = self.decoder2(dec1, enc1)
        mask = self.out(out)
        mask = F.interpolate(
            mask, scale_factor=(1, 2, 2), mode="trilinear", align_corners=False
        )
        return mask


class UNETR3D(nn.Module):
    """General segmenter module for 3D medical images"""

    def __init__(
        self,
        encoder,
        in_channels: int,
        out_channels: int,
        img_size: Union[Sequence[int], int],
        patch_size: Union[Sequence[int], int],
        feature_size: int = 16,
        hidden_size: int = 768,
        norm_name: Union[Tuple, str] = "instance",
        conv_block: bool = True,
        res_block: bool = True,
        pool: bool = False,
        spatial_dims: int = 3,
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = UNETR_decoder(
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=img_size,
            patch_size=patch_size,
            feature_size=feature_size,
            hidden_size=hidden_size,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
            spatial_dims=spatial_dims,
            pool=pool,
        )

    def get_num_layers(self):
        return self.encoder.get_num_layers()

    @torch.jit.ignore
    def no_weight_decay(self):
        total_set = set()
        module_prefix_dict = {self.encoder: "encoder", self.decoder: "decoder"}
        for module, prefix in module_prefix_dict.items():
            if hasattr(module, "no_weight_decay"):
                for name in module.no_weight_decay():
                    total_set.add(f"{prefix}.{name}")
        print(f"{total_set} will skip weight decay")
        return total_set

    def forward(self, x):
        """
        x_in in shape of [BCDHW]
        """
        cls, feat, hidden_states = self.encoder(x, ret_hids=True)
        logits = self.decoder(x, feat, hidden_states)
        return logits


class UNETR3D_prompt(UNETR3D):
    """General segmenter module for 3D medical images"""

    def __init__(
        self,
        encoder,
        in_channels: int,
        out_channels: int,
        img_size: Union[Sequence[int], int],
        patch_size: Union[Sequence[int], int],
        feature_size: int = 16,
        hidden_size: int = 768,
        norm_name: Union[Tuple, str] = "instance",
        conv_block: bool = True,
        res_block: bool = True,
        spatial_dims: int = 3,
    ):
        super().__init__(
            encoder=encoder,
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=img_size,
            patch_size=patch_size,
            feature_size=feature_size,
            hidden_size=hidden_size,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
            spatial_dims=spatial_dims,
        )

        prompt = torch.empty([1, 2, *img_size])
        nn.init.normal_(prompt, std=0.02)
        self.prompt = nn.Parameter(prompt)

    def forward(self, x):
        """
        x_in in shape of [BCDHW]
        """
        prompt = self.prompt.expand(x.shape[0], -1, -1, -1, -1)
        x = torch.cat([x, prompt], 1)
        cls, feat, hidden_states = self.encoder(x, ret_hids=True)
        logits = self.decoder(x, feat, hidden_states)
        return logits
