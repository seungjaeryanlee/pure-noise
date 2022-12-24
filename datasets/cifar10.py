import json
import os

import cv2
from torch.utils.data import Dataset
from torchvision.io import read_image


class CIFAR10LTDataset(Dataset):
    def __init__(self, json_filepath, images_dirpath, transform=None, target_transform=None):
        """
        CIFAR-10-LT Dataset
        
        NOTE: transform is NOT expected to be PyTorch transform, but albumentations transform.

        """
        self.json_filepath = json_filepath
        self.images_dirpath = images_dirpath
        self.transform = transform

        with open(self.json_filepath, "r")as f:
            self.json_data = json.load(f)

    def __len__(self):
        return len(self.json_data["annotations"])

    def __getitem__(self, idx):
        label = self.json_data["annotations"][idx]["category_id"]
        image_filepath = self.json_data["annotations"][idx]["fpath"]
        # NOTE: Below line reads image as PyTorch Tensor
        # image = read_image(image_filepath)
        # NOTE: Instead, we use OpenCV to use albumentations library
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image, label
