import json
import os

import numpy as np

from torch.utils.data import Dataset
from torchvision.io import read_image
from .sampling import count_class_frequency, compute_class_weights, compute_sample_weights


CIFAR10LT_TRAIN_JSON_FILEPATH = "data/json/cifar10_imbalance100/cifar10_imbalance100_train.json"
CIFAR10LT_TRAIN_IMAGES_DIRPATH = "data/json/cifar10_imbalance100/images/"
CIFAR10LT_VALID_JSON_FILEPATH = "data/json/cifar10_imbalance100/cifar10_imbalance100_valid.json"
CIFAR10LT_VALID_IMAGES_DIRPATH = "data/json/cifar10_imbalance100/images/"


class CIFAR10LTDataset(Dataset):
    NUM_CLASSES = 10
    
    def __init__(self, 
                 json_filepath, 
                 images_dirpath, 
                 transform=None, 
                 target_transform=None,
                 use_effective_num_sample_weights=False):
        self.json_filepath = json_filepath
        self.images_dirpath = images_dirpath
        self.transform = transform
        self.target_transform = target_transform

        with open(self.json_filepath, "r")as f:
            self.json_data = json.load(f)
            
        self._set_sample_weights(use_effective_num_sample_weights)

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

    def _set_sample_weights(self, use_effective_num_samples):
        labels = self._get_labels()
        self.class_frequency = count_class_frequency(labels, self.NUM_CLASSES)
        self.class_weights = compute_class_weights(self.class_frequency, use_effective_num_samples)
        self.sample_weights = compute_sample_weights(labels, self.class_weights)
        
    def _get_labels(self):
        return [label for _, label in self]


def build_train_dataset(transform, use_effective_num_sample_weights=False):
    return CIFAR10LTDataset(
        json_filepath=CIFAR10LT_TRAIN_JSON_FILEPATH,
        images_dirpath=CIFAR10LT_TRAIN_IMAGES_DIRPATH,
        transform=transform,
        use_effective_num_sample_weights=use_effective_num_sample_weights
    )

def build_valid_dataset(transform):
    return CIFAR10LTDataset(
        json_filepath=CIFAR10LT_VALID_JSON_FILEPATH,
        images_dirpath=CIFAR10LT_VALID_IMAGES_DIRPATH,
        transform=transform,
    )
