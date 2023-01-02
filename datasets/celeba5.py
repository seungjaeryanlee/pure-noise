import torch
import os

from torch.utils.data import Dataset
from torchvision.io import read_image

class CelebA5Dataset(Dataset):
    def __init__(self, dataset_path, transform=None, target_transform=None):
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

    def __len__(self):
        return len(self.img_path_and_labels)

    def __getitem__(self, idx):
        img_path, label = self.img_path_and_labels[idx]
        img = read_image(img_path)

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)

        return img, label


if __name__ == "__main__":
    train_dataset = CelebA5Dataset('data/CelebA5/train')
    print(len(train_dataset))
    img, label = train_dataset[0]
    print(f'label: {label}, image shape: {img.shape}')
