"""Buoc 3 - Transformation: resize, augmentation (chi cho train), chuan hoa ImageNet."""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms as T
from src import config as C


def get_transforms(train: bool):
    norm = T.Normalize(C.IMAGENET_MEAN, C.IMAGENET_STD)
    if train:
        return T.Compose([
            T.RandomResizedCrop(C.IMG_SIZE, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(10),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            T.ToTensor(), norm])
    return T.Compose([T.Resize((C.IMG_SIZE, C.IMG_SIZE)), T.ToTensor(), norm])


def get_loaders(batch_size=32, num_workers=2):
    sets = {s: datasets.ImageFolder(C.PROCESSED / s, get_transforms(s == "train"))
            for s in ("train", "val", "test")}
    loaders = {s: DataLoader(d, batch_size=batch_size, shuffle=(s == "train"),
                             num_workers=num_workers) for s, d in sets.items()}
    return loaders, sets["train"].classes, sets["train"].targets


def class_weights(targets, num_classes):
    counts = torch.bincount(torch.tensor(targets), minlength=num_classes).float()
    return counts.sum() / (num_classes * counts)
