import torch
import os

from torch.utils.data import Dataset
from torchvision.io import read_image

class CelebA5Dataset(Dataset):
    def __init__(self, dataset_path, transform=None, target_transform=None):
        self.dataset_path = dataset_path
        self.transform = transform
        self.target_transform = target_transform
        self.img_and_labels = []

        labels_path = os.path.join(self.dataset_path, 'labels.txt')
        with open(labels_path, "r") as f:
            for line in f:
                img_path, label = line.split()
                img_path = os.path.join(self.dataset_path, img_path)
                img = read_image(img_path)
                label = int(label)
                self.img_and_labels.append((img, label))

    def __len__(self):
        return len(self.img_and_labels)

    def __getitem__(self, idx):
        img, label = self.img_and_labels[idx]
        out_img = self.transform(img) if self.transform else img
        out_label = self.target_transform(label) if self.target_transform else label
        return out_img, out_label


if __name__ == "__main__":
    train_dataset = CelebA5Dataset('data/CelebA5_64x64/train')
    print(len(train_dataset))
    img, label = train_dataset[0]
    print(f'label: {label}, image shape: {img.shape}')
