import os
from typing import Optional, Callable

import torch
from torch.utils.data import Dataset, random_split

import torchvision
from torchvision import transforms

from skimage.feature import hog
from PIL import Image


class GrayToRGB(object):
    def __call__(self, image: Image.Image):
        if image.mode == "L":
            image = image.convert("RGB")
        return image


def get_dataset(name):
    if name == "MNIST":
        transform = transforms.ToTensor()
        return (
            10,
            torchvision.datasets.MNIST(
                "data", train=True, transform=transform, download=True
            ),
            torchvision.datasets.MNIST(
                "data", train=False, transform=transform, download=True
            ),
        )
    elif name == "FashionMNIST":
        transform = transforms.ToTensor()
        return (
            10,
            torchvision.datasets.FashionMNIST(
                "data", train=True, transform=transform, download=True
            ),
            torchvision.datasets.FashionMNIST(
                "data", train=False, transform=transform, download=True
            ),
        )

    elif name == "Caltech101":
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                GrayToRGB(),
                transforms.ToTensor(),
            ]
        )
        dataset = torchvision.datasets.Caltech101(
            "data", transform=transform, download=True
        )
        train_dataset, valid_dataset = random_split(dataset, [0.8, 0.2])
        return 101, train_dataset, valid_dataset
    elif name == "Caltech256":
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                GrayToRGB(),
                transforms.ToTensor(),
            ]
        )
        dataset = torchvision.datasets.Caltech256(
            "data", transform=transform, download=True
        )
        train_dataset, valid_dataset = random_split(dataset, [0.8, 0.2])
        return 257, train_dataset, valid_dataset
    else:
        raise ValueError(f"Dataset {name} does not exist")
