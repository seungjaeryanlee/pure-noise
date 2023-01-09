import torch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def dataset_mean_and_std(dataset):
    imgs = torch.stack([img for img, _ in dataset]).type(torch.FloatTensor)
    dataset_mean = imgs.mean(dim=[0,2,3]).to(device)
    dataset_std = imgs.std(dim=[0,2,3]).to(device)
    return dataset_mean.tolist(), dataset_std.tolist()

if __name__ == '__main__':
    from celeba5 import CelebA5Dataset, CELEBA5_TRAIN_DATASET_PATH
    
    celeba5_dataset = CelebA5Dataset(
        dataset_path=CELEBA5_TRAIN_DATASET_PATH,
    )
    mean, std = dataset_mean_and_std(celeba5_dataset)
    print(mean, std)
