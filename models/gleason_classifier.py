import torch
import torch.nn as nn
from torch.autograd import Function


class GradientReverse(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


def gradient_reverse(x, lambda_=1.0):
    return GradientReverse.apply(x, lambda_)


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


class GleasonClassifier(nn.Module):
    def __init__(
        self,
        encoder,
        main_out_dim,
        adversarial_num_classes=None,
        bottleneck_dim=256,
        grl_lambda=1.0,
    ):
        super().__init__()
        self.encoder = encoder
        self.embed_dim = self.encoder.embed_dim
        self.main_head = MLPHead(self.embed_dim, main_out_dim, bottleneck_dim)
        self.grl_lambda = grl_lambda
        adversarial_num_classes = adversarial_num_classes or {}
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
        output = {"main": self.main_head(features)}
        if self.adversarial_heads:
            adv_features = gradient_reverse(features, self.grl_lambda)
            output["adversarial"] = {
                name: head(adv_features)
                for name, head in self.adversarial_heads.items()
            }
        return output

    def head_parameters(self):
        return list(self.main_head.parameters()) + list(self.adversarial_heads.parameters())

    def set_grl_lambda(self, value):
        self.grl_lambda = float(value)
