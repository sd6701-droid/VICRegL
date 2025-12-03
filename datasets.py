# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset
import torch.distributed as dist   
import torchvision.datasets as datasets

from transforms import MultiCropTrainDataTransform, MultiCropValDataTransform


IMAGENET_NUMPY_PATH = "/private/home/abardes/datasets/imagenet1k/"
IMAGENET_PATH = "/datasets01/imagenet_full_size/061417"


class ImageNetNumpyDataset(Dataset):
    def __init__(self, img_file, labels_file, size_dataset=-1, transform=None):
        self.samples = np.load(img_file)
        self.labels = np.load(labels_file)
        if size_dataset > 0:
            self.samples = self.samples[:size_dataset]
            self.labels = self.labels[:size_dataset]
        assert len(self.samples) == len(self.labels)
        self.transform = transform

    def get_img(self, path, transform):
        with open(path, "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")
        if transform is not None:
            img = transform(img)
        return img

    def __getitem__(self, i):
        img = self.get_img(self.samples[i], self.transform)
        lab = self.labels[i]
        return img, lab

    def __len__(self):
        return len(self.samples)


class SingleFolderDataset(Dataset):
    """
    Dataset for the case where *all* images are in a single folder
    (no class subdirectories). We assign a dummy label 0 to every image,
    so num_classes = 1.

    This works fine for VICRegL's self-supervised loss; the online
    classifier just becomes a 1-class dummy head.
    """

    IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp")

    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform

        samples = []
        for fname in sorted(os.listdir(root)):
            path = os.path.join(root, fname)
            if os.path.isfile(path) and fname.lower().endswith(self.IMG_EXTENSIONS):
                samples.append(path)

        if len(samples) == 0:
            raise RuntimeError(f"Found 0 images in folder: {root}")

        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path = self.samples[index]
        with open(path, "rb") as f:
            img = Image.open(f).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        # Dummy label: 0
        target = 0
        return img, target


def build_loader(args, is_train=True):
    dataset = build_dataset(args, is_train)

    batch_size = args.batch_size
    if (not is_train) and args.val_batch_size == -1:
        batch_size = args.batch_size

    # Use DistributedSampler only if torch.distributed is initialized
    if dist.is_available() and dist.is_initialized():
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            shuffle=is_train,
        )
        per_device_batch_size = batch_size // args.world_size
        shuffle = False
    else:
        # Single-process / non-distributed case
        sampler = None
        per_device_batch_size = batch_size
        shuffle = is_train

    loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        shuffle=shuffle,
        batch_size=per_device_batch_size,
        num_workers=args.num_workers,
    )

    return loader, sampler


def build_dataset(args, is_train=True):
    transform = build_transform(args, is_train=is_train)

    # 1) Built-in ImageNet mode (original behavior)
    if args.dataset == "imagenet1k":
        args.num_classes = 1000

        if args.dataset_from_numpy:
            root = IMAGENET_NUMPY_PATH
            prefix = "train" if is_train else "val"
            images_path = os.path.join(root, f"{prefix}_images.npy")
            labels_path = os.path.join(root, f"{prefix}_labels.npy")
            dataset = ImageNetNumpyDataset(
                images_path, labels_path, transform=transform
            )
        else:
            root = IMAGENET_PATH
            prefix = "train" if is_train else "val"
            path = os.path.join(root, prefix)
            dataset = datasets.ImageFolder(path, transform)

        return dataset

    # 2) Treat args.dataset as a PATH to a folder containing all images
    if os.path.isdir(args.dataset):
        # Single-folder dataset, all images share label 0
        args.num_classes = 1
        path = args.dataset  # same folder for train / val unless you change the arg
        dataset = SingleFolderDataset(path, transform=transform)
        return dataset

    # 3) If we get here, the dataset specifier is unknown
    raise ValueError(
        f"Unknown dataset specifier: {args.dataset}. "
        f"Expected 'imagenet1k' or a path to a folder of images."
    )


def build_transform(args, is_train=True):
    transform_args = {
        "size_crops": args.size_crops,
        "num_crops": args.num_crops,
        "min_scale_crops": args.min_scale_crops,
        "max_scale_crops": args.max_scale_crops,
        "return_location_masks": True,
        "no_flip_grid": args.no_flip_grid,
    }
    if is_train:
        transform = MultiCropTrainDataTransform(**transform_args)
    else:
        transform = MultiCropValDataTransform(**transform_args)

    return transform
