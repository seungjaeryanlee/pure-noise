import torch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def dataset_mean_and_std(dataset):
    imgs = torch.stack([img for img, _ in dataset]).type(torch.FloatTensor)
    dataset_mean = imgs.mean(dim=[0,2,3]).to(device)
    dataset_std = imgs.std(dim=[0,2,3]).to(device)
    return dataset_mean.tolist(), dataset_std.tolist()

if __name__ == '__main__':
    from celeba5 import CelebA5Dataset, CELEBA5_TRAIN_DATASET_PATH
    from cifar10 import CIFAR10LTDataset
    from torchvision import transforms

    celeba5_dataset = CelebA5Dataset(
        dataset_path=CELEBA5_TRAIN_DATASET_PATH,
        transform=transforms.ToTensor(),
    )
    mean, std = dataset_mean_and_std(celeba5_dataset)
    print(f"CelebA-5 Mean: {mean}")
    print(f"CelebA-5 Std: {std}")

    cifar10lt_dataset = CIFAR10LTDataset(
        json_filepath = "data/json/cifar10_imbalance100/cifar10_imbalance100_train.json",
        images_dirpath = "data/json/cifar10_imbalance100/images/",
        transform=transforms.ConvertImageDtype(float),
    )
    mean, std = dataset_mean_and_std(cifar10lt_dataset)
    print(f"CIFAR-10-LT Mean: {mean}")
    print(f"CIFAR-10-LT Std: {std}")
