import json
import os

from torch.utils.data import Dataset
from torchvision.io import read_image


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
