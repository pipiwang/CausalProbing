import torch
from monai.metrics import (
    compute_dice,
    compute_hausdorff_distance,
    compute_average_surface_distance,
)
from monai.networks import one_hot


class DiceMeter:
    def __init__(self, args):
        self.num_classes = args.num_classes
        if self.num_classes > 1:
            self.include_background = False
        else:
            self.include_background = True
        self.reset()

    def reset(self):
        self.count, self.sum = torch.zeros(self.num_classes), torch.zeros(
            self.num_classes
        )

    def one_hot(self, pred, gt):
        if self.num_classes > 1:
            pred_one_hot = one_hot(pred, num_classes=self.num_classes + 1)
            gt_one_hot = one_hot(
                gt, num_classes=self.num_classes + 1
            )  # (1, C, H, W, D)
        else:
            pred_one_hot = pred
            gt_one_hot = gt
        return pred_one_hot, gt_one_hot

    def update(self, pred, gt):
        """
        :param pred_binary: (1, 1, H, W, D)
        :param gt: (1, 1, H, W, D)
        """
        pred_one_hot, gt_one_hot = self.one_hot(pred, gt)
        mean_dice = compute_dice(
            pred_one_hot, gt_one_hot, self.include_background, ignore_empty=True
        ).sum(dim=0)
        nan = torch.isnan(mean_dice)
        n_n = (~nan).float()
        mean_dice[nan] = 0
        self.sum += mean_dice.cpu()
        self.count += n_n.cpu()

    def get_average(self):
        self.count[self.count == 0] = -1
        mean = self.sum / self.count
        metric = torch.mean(mean[self.count > 0])
        self.reset()
        return mean.numpy(), metric.numpy()


class HausdorffMeter(DiceMeter):
    def __init__(self, args):
        super(HausdorffMeter, self).__init__(args)
        self.spacing = args.spacing
        self.reset()

    def update(self, pred, gt):
        """
        :param pred_binary: (1, 1, H, W, D)
        :param gt: (1, 1, H, W, D)
        """
        pred_one_hot, gt_one_hot = self.one_hot(pred, gt)
        mean_dice = compute_hausdorff_distance(
            pred_one_hot,
            gt_one_hot,
            self.include_background,
            percentile=95,
            spacing=self.spacing,
        ).sum(dim=0)
        nan = torch.isnan(mean_dice)
        n_n = (~nan).float()
        mean_dice[nan] = 0
        self.sum += mean_dice.cpu()
        self.count += n_n.cpu()


class SurfaceDistanceMeter(DiceMeter):
    def __init__(self, args):
        super(SurfaceDistanceMeter, self).__init__(args)
        self.spacing = args.spacing
        self.reset()

    def update(self, pred, gt):
        """
        :param pred_binary: (1, 1, H, W, D)
        :param gt: (1, 1, H, W, D)
        """
        pred_one_hot, gt_one_hot = self.one_hot(pred, gt)
        mean_dice = compute_average_surface_distance(
            pred_one_hot, gt_one_hot, self.include_background, spacing=self.spacing
        ).sum(dim=0)
        nan = torch.isnan(mean_dice)
        n_n = (~nan).float()
        mean_dice[nan] = 0
        self.sum += mean_dice.cpu()
        self.count += n_n.cpu()
