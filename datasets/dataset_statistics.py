import torch
import numpy as np

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def dataset_mean_and_std(dataset):
    imgs = torch.stack([img for img, _ in dataset]).type(torch.FloatTensor)
    dataset_mean = imgs.mean(dim=[0,2,3]).to(device)
    dataset_std = imgs.std(dim=[0,2,3]).to(device)
    return dataset_mean, dataset_std

def ir_ratio(dataset):
    labels = [label for _, label in dataset]
    label_to_count = [0] * len(np.unique(labels))
    for label in labels:
        label_to_count[label] += 1
    return max(label_to_count) / min(label_to_count)

if __name__ == '__main__':
    from .celeba5 import CELEBA5_TRAIN_DATASET_PATH, CelebA5Dataset
    from .imbalanced_cifar import IMBALANCECIFAR10
    from torchvision.datasets import CIFAR10, CIFAR100
    from torchvision import transforms
    
    torch.set_printoptions(precision=4)

    def compute_for_dataset(dataset):
        print(f"IR ratio: {ir_ratio(dataset)}")
        mean, std = dataset_mean_and_std(dataset)
        print(f"Mean: {mean}, Std: {std}")
        
    print("CelebA-5")
    celeba5_dataset = CelebA5Dataset(
        dataset_path=CELEBA5_TRAIN_DATASET_PATH,
        transform=transforms.ToTensor(),
    )
    compute_for_dataset(celeba5_dataset)

    print("CIFAR10 Long-Tailed")
    cifar10lt_dataset = IMBALANCECIFAR10(
        root="data/",
        transform=transforms.ToTensor()
    )
    compute_for_dataset(cifar10lt_dataset)
    
    print("Balanced CIFAR10")
    cifar10_dataset = CIFAR10(
        root="data/",
        transform=transforms.ToTensor(),
        train=True
    )
    compute_for_dataset(cifar10_dataset)
