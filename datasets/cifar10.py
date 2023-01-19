import json
import os

import numpy as np

from torch.utils.data import Dataset
from torchvision.io import read_image


CIFAR10LT_TRAIN_JSON_FILEPATH = "data/json/cifar10_imbalance100/cifar10_imbalance100_train.json"
CIFAR10LT_TRAIN_IMAGES_DIRPATH = "data/json/cifar10_imbalance100/images/"
CIFAR10LT_VALID_JSON_FILEPATH = "data/json/cifar10_imbalance100/cifar10_imbalance100_valid.json"
CIFAR10LT_VALID_IMAGES_DIRPATH = "data/json/cifar10_imbalance100/images/"


class CIFAR10LTDataset(Dataset):
    def __init__(self, json_filepath, images_dirpath, transform=None, target_transform=None):
        self.json_filepath = json_filepath
        self.images_dirpath = images_dirpath
        self.transform = transform
        self.target_transform = target_transform

        with open(self.json_filepath, "r")as f:
            self.json_data = json.load(f)

    def __len__(self):
        return len(self.json_data["annotations"])

    def __getitem__(self, idx):
        label = self.json_data["annotations"][idx]["category_id"]
        image_filepath = self.json_data["annotations"][idx]["fpath"]
        image = read_image(image_filepath)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label


def build_train_dataset(transform):
    return CIFAR10LTDataset(
        json_filepath=CIFAR10LT_TRAIN_JSON_FILEPATH,
        images_dirpath=CIFAR10LT_TRAIN_IMAGES_DIRPATH,
        transform=transform,
    )

def build_valid_dataset(transform):
    return CIFAR10LTDataset(
        json_filepath=CIFAR10LT_VALID_JSON_FILEPATH,
        images_dirpath=CIFAR10LT_VALID_IMAGES_DIRPATH,
        transform=transform,
    )
