import torch
import numpy as np

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def dataset_mean_and_std(dataset):
    imgs = torch.stack([img for img, _ in dataset]).type(torch.FloatTensor)
    dataset_mean = imgs.mean(dim=[0,2,3]).to(device)
    dataset_std = imgs.std(dim=[0,2,3]).to(device)
    return dataset_mean.tolist(), dataset_std.tolist()

def ir_ratio(dataset):
    labels = [label for _, label in dataset]
    label_to_count = [0] * len(np.unique(labels))
    for label in labels:
        label_to_count[label] += 1
    return max(label_to_count) / min(label_to_count)

if __name__ == '__main__':
    from celeba5 import CelebA5Dataset, CELEBA5_TRAIN_DATASET_PATH
    from cifar10 import CIFAR10LTDataset
    from torchvision import transforms

    celeba5_dataset = CelebA5Dataset(
        dataset_path=CELEBA5_TRAIN_DATASET_PATH,
        transform=transforms.ToTensor(),
    )
    print(f"CelebA-5 IR ratio: {ir_ratio(celeba5_dataset)}")
    mean, std = dataset_mean_and_std(celeba5_dataset)
    print(f"CelebA-5 Mean: {mean}")
    print(f"CelebA-5 Std: {std}")

    cifar10lt_dataset = CIFAR10LTDataset(
        json_filepath = "data/json/cifar10_imbalance100/cifar10_imbalance100_train.json",
        images_dirpath = "data/json/cifar10_imbalance100/images/",
        transform=transforms.ConvertImageDtype(float),
    )
    print(f"CIFAR-10-LT IR ratio: {ir_ratio(celeba5_dataset)}")
    mean, std = dataset_mean_and_std(cifar10lt_dataset)
    print(f"CIFAR-10-LT Mean: {mean}")
    print(f"CIFAR-10-LT Std: {std}")
