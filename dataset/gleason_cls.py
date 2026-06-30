import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
import monai.transforms.compose as monai_compose
import monai.transforms.transform as monai_transform
from monai.transforms import (
    CenterSpatialCropd,
    Compose,
    NormalizeIntensityd,
    RandBiasFieldd,
    RandFlipd,
    RandGaussianSmoothd,
    RandRotated,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    RandZoomd,
    SpatialPadd,
)
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from util.gleason_config import resolve_adversarial_config


def patch_monai_uint32_seed_limit():
    max_uint32 = int(np.iinfo(np.uint32).max)
    monai_compose.MAX_SEED = max_uint32
    monai_transform.MAX_SEED = max_uint32


def parse_int_tuple(value):
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    value = str(value).replace("(", "").replace(")", "")
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def is_missing(value) -> bool:
    if pd.isna(value):
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def is_observed_value(value) -> bool:
    if is_missing(value):
        return False
    if isinstance(value, str):
        value = value.strip().lower()
        return value in {"1", "true", "t", "yes", "y"}
    return int(value) == 1


class GleasonClassificationDataset(Dataset):
    """Gleason classification dataset backed by preprocessed .npy image caches."""

    def __init__(self, args, csv_path: str, phase: str, transforms=None):
        super().__init__()
        if csv_path is None:
            raise ValueError(f"No CSV path provided for {phase} split")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"{csv_path} does not exist")

        self.args = args
        self.phase = phase
        self.root = args.data_root
        self.transforms = transforms
        self.image_path_col = args.image_path_col
        self.adversarial_specs = resolve_adversarial_config(args)
        self.adversarial_columns = args.adversarial_columns
        self.adversarial_observed_columns = args.adversarial_observed_columns

        df = pd.read_csv(csv_path)
        if args.split_col and args.split_col in df.columns:
            df = df[df[args.split_col].astype(str) == phase].reset_index(drop=True)
        if len(df) == 0:
            raise ValueError(f"{csv_path} has no rows for phase '{phase}'")
        self.df = df

        self._validate_columns()
        self._validate_adversarial_values()
        print(f"Loading Gleason {phase} dataset with {len(self.df)} samples")

    def _validate_columns(self):
        missing = [col for col in [self.image_path_col] if col not in self.df.columns]
        if missing:
            raise KeyError(f"Missing image path column(s): {missing}")

        label_cols = []
        if self.args.task_type == "binary" and self.args.binary_label_col:
            label_cols.append(self.args.binary_label_col)
        else:
            label_cols.append(self.args.label_col)

        missing = [col for col in label_cols if col not in self.df.columns]
        if missing:
            raise KeyError(f"Missing label column(s): {missing}")

        missing = [
            col for col in self.adversarial_columns.values() if col not in self.df.columns
        ]
        if missing:
            raise KeyError(f"Missing adversarial column(s): {missing}")

        missing = [
            col
            for col in self.adversarial_observed_columns.values()
            if col not in self.df.columns
        ]
        if missing:
            raise KeyError(f"Missing adversarial observed column(s): {missing}")

    def _validate_adversarial_values(self):
        for name, num_classes in self.adversarial_specs.items():
            column = self.adversarial_columns[name]
            observed_column = self.adversarial_observed_columns[name]
            observed = self.df[observed_column].apply(is_observed_value)
            usable = observed & ~self.df[column].apply(is_missing)
            values = pd.to_numeric(self.df.loc[usable, column], errors="coerce")

            non_integer = values.notna() & ~np.isclose(values, np.round(values))
            out_of_range = values.notna() & (
                (values < 0) | (values >= int(num_classes))
            )
            invalid = values.isna() | non_integer | out_of_range
            if invalid.any():
                bad_values = (
                    self.df.loc[usable, column]
                    .loc[invalid]
                    .drop_duplicates()
                    .head(10)
                    .tolist()
                )
                raise ValueError(
                    f"Invalid adversarial target values for '{name}' in {column}: "
                    f"{bad_values}. Expected integer class ids in "
                    f"[0, {int(num_classes) - 1}]."
                )

            unique_count = int(values.astype(int).nunique()) if len(values) else 0
            if self.phase == "train" and unique_count < 2:
                print(
                    f"Warning: adversarial target '{name}' has {unique_count} "
                    f"observed class(es) in the train split; adversarial training "
                    "will be weak or inactive for this variable."
                )

    def __len__(self):
        return len(self.df)

    def full_image_path(self, path: str) -> str:
        if not isinstance(path, str) or path.strip() == "":
            raise ValueError("empty image cache path")
        return path if os.path.isabs(path) else os.path.join(self.root, path)

    def read_image(self, path: str) -> torch.Tensor:
        full_path = self.full_image_path(path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"image cache does not exist: {full_path}")
        image = np.load(full_path).astype(np.float32)
        if image.ndim != 4:
            raise ValueError(f"expected [C,Z,H,W] image, got shape {image.shape}")
        if image.shape[0] != self.args.in_channels:
            raise ValueError(
                f"expected {self.args.in_channels} channels, got {image.shape[0]}"
            )
        if not np.isfinite(image).all():
            raise ValueError(f"non-finite value in image cache: {full_path}")
        if not np.any(image != 0):
            raise ValueError(f"all-zero image cache: {full_path}")
        return torch.from_numpy(image)

    def get_label(self, row) -> torch.Tensor:
        if self.args.task_type == "binary":
            if self.args.binary_label_col:
                label = int(row[self.args.binary_label_col])
            else:
                label = int(row[self.args.label_col]) >= self.args.binary_positive_min
            return torch.tensor(int(label), dtype=torch.long)

        label = int(row[self.args.label_col]) - self.args.label_offset
        if label < 0 or label >= self.args.ordinal_levels:
            raise ValueError(
                f"Ordinal label {label} is outside [0, {self.args.ordinal_levels - 1}]"
            )
        return torch.tensor(label, dtype=torch.long)

    def get_adversarial_targets(self, row):
        targets = {}
        masks = {}
        for name in self.adversarial_specs:
            column = self.adversarial_columns[name]
            observed_column = self.adversarial_observed_columns[name]
            value = row[column]
            observed = is_observed_value(row[observed_column])
            if not observed or is_missing(value):
                targets[name] = torch.tensor(0, dtype=torch.long)
                masks[name] = torch.tensor(False, dtype=torch.bool)
                continue
            targets[name] = torch.tensor(int(value), dtype=torch.long)
            masks[name] = torch.tensor(True, dtype=torch.bool)
        return targets, masks

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = self.read_image(row[self.image_path_col])
        label = self.get_label(row)
        adversarial_targets, adversarial_masks = self.get_adversarial_targets(row)

        if self.transforms is not None:
            data = self.transforms({"image": image})
            if isinstance(data, list):
                data = data[0]
            image = torch.as_tensor(data["image"], dtype=torch.float32)

        sample_idx = torch.tensor(idx, dtype=torch.long)
        if adversarial_targets:
            return image, label, adversarial_targets, adversarial_masks, sample_idx
        return image, label, sample_idx

    def sampler_weights(self):
        labels = [int(self.get_label(row)) for _, row in self.df.iterrows()]
        counts = pd.Series(labels).value_counts().to_dict()
        return torch.as_tensor([1.0 / counts[label] for label in labels], dtype=torch.double)

    def label_counts(self):
        labels = [int(self.get_label(row)) for _, row in self.df.iterrows()]
        return pd.Series(labels).value_counts(dropna=False).sort_index().to_dict()

    def adversarial_observed_counts(self):
        counts = {}
        for name in self.adversarial_specs:
            column = self.adversarial_columns[name]
            observed_column = self.adversarial_observed_columns[name]
            observed_flag = self.df[observed_column].apply(is_observed_value)
            usable = observed_flag & ~self.df[column].apply(is_missing)
            counts[name] = {
                "observed_flag_1": int(observed_flag.sum()),
                "usable_for_adversarial_loss": int(usable.sum()),
                "not_used_for_adversarial_loss": int((~usable).sum()),
                "target_counts": self.df.loc[usable, column]
                .astype(int)
                .value_counts()
                .sort_index()
                .to_dict(),
            }
        return counts


def get_gleason_transforms(args):
    patch_monai_uint32_seed_limit()
    crop_size = tuple(args.crop_spatial_size)
    train_transforms = Compose(
        [
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            SpatialPadd(keys="image", spatial_size=list(crop_size)),
            RandSpatialCropd(keys="image", roi_size=crop_size, random_size=False),
            RandFlipd(keys="image", prob=0.5, spatial_axis=2),
            RandRotated(
                keys="image",
                prob=0.3,
                range_x=10 / 180 * np.pi,
                range_y=10 / 180 * np.pi,
                range_z=10 / 180 * np.pi,
                keep_size=True,
                mode="bilinear",
            ),
            RandZoomd(
                keys="image",
                prob=0.3,
                min_zoom=[0.9, 0.9, 0.9],
                max_zoom=[1.1, 1.1, 1.1],
                keep_size=True,
                mode="trilinear",
            ),
            RandScaleIntensityd(keys="image", factors=0.1, prob=0.8),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=0.8),
            RandBiasFieldd(keys="image", prob=0.2),
            RandGaussianSmoothd(keys="image", prob=0.2),
        ]
    )
    eval_transforms = Compose(
        [
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            SpatialPadd(keys="image", spatial_size=list(crop_size)),
            CenterSpatialCropd(keys="image", roi_size=crop_size),
        ]
    )
    return train_transforms, eval_transforms, eval_transforms


def _csv_for_phase(args, phase: str) -> Optional[str]:
    phase_csv = getattr(args, f"{phase}_csv")
    return phase_csv if phase_csv else args.csv_path


def configure_gleason_task_args(args):
    args.crop_spatial_size = tuple(parse_int_tuple(args.crop_spatial_size))
    args.in_channels = int(getattr(args, "in_channels", 3))
    resolve_adversarial_config(args)

    if args.task_type == "binary":
        if args.binary_label_col is None and args.binary_positive_min < 1:
            raise ValueError("--binary_positive_min must be >= 1")
        args.label_definition = (
            f"{args.binary_label_col}=1"
            if args.binary_label_col
            else f"{args.label_col}>={args.binary_positive_min}"
        )
        args.num_classes = 1
    elif args.task_type == "ordinal":
        if args.ordinal_levels < 2:
            raise ValueError("--ordinal_levels must be >= 2")
        args.label_definition = (
            f"{args.label_col} ordinal, levels={args.ordinal_levels}, "
            f"offset={args.label_offset}"
        )
        args.num_classes = args.ordinal_levels - 1
    else:
        raise NotImplementedError(f"unknown task_type: {args.task_type}")
    return args


def build_gleason_classification_loaders(args):
    configure_gleason_task_args(args)
    train_transforms, val_transforms, test_transforms = get_gleason_transforms(args)

    train_set = GleasonClassificationDataset(
        args, _csv_for_phase(args, "train"), "train", train_transforms
    )
    val_set = GleasonClassificationDataset(
        args, _csv_for_phase(args, "val"), "val", val_transforms
    )
    test_set = GleasonClassificationDataset(
        args, _csv_for_phase(args, "test"), "test", test_transforms
    )

    sampler = None
    shuffle = True
    if args.weighted_sampling:
        sampler = WeightedRandomSampler(
            weights=train_set.sampler_weights(),
            num_samples=len(train_set),
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
