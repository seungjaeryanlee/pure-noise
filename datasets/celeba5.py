import numpy as np
import torch
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision.io import read_image
from sampling import count_class_frequency, compute_class_weights, compute_sample_weights

CELEBA5_TRAIN_DATASET_PATH = 'data/CelebA5_64x64/train'
CELEBA5_VALID_DATASET_PATH = 'data/CelebA5_64x64/valid'
CELEBA5_TEST_DATASET_PATH = 'data/CelebA5_64x64/test'

class CelebA5Dataset(Dataset):
    NUM_CLASSES = 5
    
    def __init__(self,
                 dataset_path, 
                 transform=None, 
                 target_transform=None, 
                 use_effective_num_sample_weights=False):
        self.dataset_path = dataset_path
        self.transform = transform
        self.target_transform = target_transform
        self.img_path_and_labels = []

        labels_path = os.path.join(self.dataset_path, 'labels.txt')
        with open(labels_path, "r") as f:
            for line in f:
                img_path, label = line.split()
                img_path = os.path.join(self.dataset_path, img_path)
                label = int(label)
                self.img_path_and_labels.append((img_path, label))
        
        self._set_sample_weights(use_effective_num_sample_weights)

    def __len__(self):
        return len(self.img_path_and_labels)

    def __getitem__(self, idx):
        img_path, label = self.img_path_and_labels[idx]
        img = Image.open(img_path)
        out_img = self.transform(img) if self.transform else img
        out_label = self.target_transform(label) if self.target_transform else label
        return out_img, out_label
    
    def _set_sample_weights(self, use_effective_num_samples):
        labels = self._get_labels()
        self.class_frequency = count_class_frequency(labels, self.NUM_CLASSES)
        self.class_weights = compute_class_weights(self.class_frequency, use_effective_num_samples)
        self.sample_weights = compute_sample_weights(labels, self.class_weights)
        
    def _get_labels(self):
        return [label for _, label in self]


def build_train_dataset(transform, use_effective_num_sample_weights=False):
    return CelebA5Dataset(CELEBA5_TRAIN_DATASET_PATH, 
                          transform=transform, 
                          use_effective_num_sample_weights=use_effective_num_sample_weights)

def build_valid_dataset(transform):
    return CelebA5Dataset(CELEBA5_VALID_DATASET_PATH, transform=transform)

if __name__ == "__main__":
    from torchvision import transforms
    augmentation_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
    ])
    train_dataset = CelebA5Dataset(CELEBA5_TRAIN_DATASET_PATH, transform=augmentation_transforms)
    print(len(train_dataset))
    img, label = train_dataset[1]
    img.show()