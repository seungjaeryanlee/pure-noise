import json
import os

import numpy as np

from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.datasets import CIFAR10
from .sampling import count_class_frequency

class CIFAR10Dataset(CIFAR10):
    CIFAR10_ROOT = "data/cifar10"
    NUM_CLASSES = 10
    
    def __init__(self, **kwargs):
        super().__init__(root=self.CIFAR10_ROOT, download=True, **kwargs)
        
        self.class_frequency = count_class_frequency(self.get_labels(), self.NUM_CLASSES)
    
    def get_labels(self):
        return [label for _, label in self]


def build_train_dataset(transform):
    return CIFAR10Dataset(train=True, transform=transform)

def build_valid_dataset(transform):
    return CIFAR10Dataset(train=False, transform=transform)

if __name__ == '__main__':
    import torch
    from torchvision import transforms
    train_dataset = build_train_dataset(transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor()]))
    valid_dataset = build_valid_dataset(transform=transforms.Compose([transforms.ToTensor()]))
    
    assert len(train_dataset) == 50000
    assert len(valid_dataset) == 10000
    
    img, label = train_dataset[0]
    assert img.shape == torch.Size([3, 32, 32])
    assert isinstance(label, int)
    
    print('Finished checking dataset')
    