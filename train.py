
def train():
    import torch
    from torchvision import transforms

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    ## Dataset
    from datasets.celeba5 import CelebA5Dataset
    TRAIN_DIR = "data/CelebA5_64x64/train"
    VALID_DIR = "data/CelebA5_64x64/valid"

    shared_transforms = [
        transforms.ConvertImageDtype(torch.float32),
    ]

    import custom_transforms

    # transforms.ToTensor() not needed as we use torchvision.io.read_image,
    # which gives torch.Tensor instead of PIL.Image
    # Data Augmentation transforms are mostly from Bazinga699/NCL
    # https://github.com/Bazinga699/NCL/blob/2bbf193/lib/dataset/cui_cifar.py#L64
    augmentation_transforms = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(64, padding=4),
        custom_transforms.Cutout(n_holes=1, length=16),
        # TODO: Check if this is correct values for SIMCLR augmentation
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=[.1, 2.])
        ], p=0.5),
    ]

    def dataset_mean_and_std(dataset):
        imgs = torch.stack([img for img, _ in dataset]).type(torch.FloatTensor)
        dataset_mean = imgs.mean(dim=[0,2,3]).to(device)
        dataset_std = imgs.std(dim=[0,2,3]).to(device)
        return dataset_mean.tolist(), dataset_std.tolist()

    def no_aug_train_stats():
        dataset = CelebA5Dataset(
            dataset_path=TRAIN_DIR,
            # transform=transforms.Compose(shared_transforms),
        )
        return dataset_mean_and_std(dataset)

    no_aug_train_mean, no_aug_train_std = no_aug_train_stats()
    # no_aug_train_mean, no_aug_train_std = [0.5037, 0.4335, 0.3993], [0.3053, 0.2887, 0.2890]

    print(no_aug_train_mean, no_aug_train_std)

    def aug_train_stats():
        dataset = CelebA5Dataset(
            dataset_path=TRAIN_DIR,
            transform=transforms.Compose(shared_transforms + augmentation_transforms),
        )
        return dataset_mean_and_std(dataset)

    # aug_train_mean, aug_train_std = aug_train_stats()
    aug_train_mean, aug_train_std = [0.4273, 0.3827, 0.3593], [0.3181, 0.3019, 0.2954]

    print(aug_train_mean, aug_train_std)


    train_transform = transforms.Compose(
        # shared_transforms + augmentation_transforms + [transforms.Normalize(no_aug_train_mean, no_aug_train_std)]
        shared_transforms + [transforms.Normalize(no_aug_train_mean, no_aug_train_std)]
    )
    valid_transform = transforms.Compose(
        shared_transforms + [transforms.Normalize(no_aug_train_mean, no_aug_train_std)]
    )
    train_dataset = CelebA5Dataset(
        dataset_path=TRAIN_DIR,
        transform=train_transform,
    )
    valid_dataset = CelebA5Dataset(
        dataset_path=VALID_DIR,
        transform=valid_transform,
    )

    len(train_dataset), len(valid_dataset)


    ## DataLoader
    # DataLoader Hyperparameters
    DATALOADER__NUM_WORKERS = 8
    DATALOADER__BATCH_SIZE = 128

    # Compute weights
    import json
    import os
    import numpy as np

    sample_labels = []
    sample_labels_count = np.arange(5)
    with open(os.path.join(TRAIN_DIR, 'labels.txt'), 'r') as f:
        for line in f:
            _, label = line.split()
            label = int(label)
            sample_labels.append(label)
            sample_labels_count[label] += 1
    weights = 1. / sample_labels_count
    sample_weights = np.array([weights[l] for l in sample_labels])

    from torch.utils.data import DataLoader, WeightedRandomSampler

    train_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=6651, # https://stackoverflow.com/a/67802529
        replacement=True,
    )
    train_loader = DataLoader(
        train_dataset,
        # sampler=train_sampler,
        shuffle=True,
        batch_size=DATALOADER__BATCH_SIZE,
        num_workers=DATALOADER__NUM_WORKERS,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=DATALOADER__BATCH_SIZE,
        num_workers=DATALOADER__NUM_WORKERS,
    )


    ## Model

    # Model hyperparameters
    MODEL__WIDERESNET_DEPTH = 28
    MODEL__WIDERESNET_K = 10
    MODEL__WIDERESNET_DROPOUT = 0.3

    from networks_torchdistill import WideBasicBlock, WideResNet

    net = WideResNet(
        depth=MODEL__WIDERESNET_DEPTH,
        k=MODEL__WIDERESNET_K,
        dropout_p=MODEL__WIDERESNET_DROPOUT,
        block=WideBasicBlock,
        num_classes=5,
    )

    # from networks import WideResNet

    # # TODO: Consider replacing with https://github.com/yoshitomo-matsubara/torchdistill/blob/main/torchdistill/models/classification/wide_resnet.py
    # net = WideResNet(
    #     num_classes=10,
    #     depth=MODEL__WIDERESNET_DEPTH,
    #     widen_factor=MODEL__WIDERESNET_K,
    #     dropRate=MODEL__WIDERESNET_DROPOUT,
    # )

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    count_parameters(net)

    net = net.to(device)


    ## Wandb
    import wandb
    wandb.login()

    # ## Optimizer

    # Optimizer Hyperparameters
    OPTIM__LR = 0.1
    OPTIM__MOMENTUM = 0.9
    OPTIM__WEIGHT_DECAY = 2e-4

    import torch.optim as optim

    optimizer = optim.SGD(
        net.parameters(),
        lr=OPTIM__LR,
        momentum=OPTIM__MOMENTUM,
        weight_decay=OPTIM__WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=1,
        gamma=0.1,
    )

    # ## Prepare Training

    # Training Hyperparameters
    N_EPOCH = 90
    SAVE_CKPT_EVERY_N_EPOCH = 10
    LOAD_CKPT = False
    LOAD_CKPT_FILEPATH = "checkpoints/.pt"
    LOAD_CKPT_EPOCH = 0

    import torch.nn as nn

    criterion = nn.CrossEntropyLoss(reduction="none")


    # ## Training Loop

    def save_checkpoint(
        model,
        optimizer,
        checkpoint_filepath: str,
    ):
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_filepath)


    def load_checkpoint(
        model,
        optimizer,
        checkpoint_filepath: str,
    ):
        checkpoint = torch.load(checkpoint_filepath)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


    if LOAD_CKPT:
        load_checkpoint(net, optimizer, LOAD_CKPT_FILEPATH)

    wandb_run = wandb.init(
        project="pure-noise",
        entity="brianryan",
    )

    wandb.config.update({
        # Data
        "dataloader__num_workers": DATALOADER__NUM_WORKERS,
        "dataloader__batch_size": DATALOADER__BATCH_SIZE,
        # Optimizer
        "optim__lr": OPTIM__LR,
        "optim__momentum": OPTIM__MOMENTUM,
        "optim__weight_decay": OPTIM__WEIGHT_DECAY,
        # Model
        "model__wideresnet_depth": MODEL__WIDERESNET_DEPTH,
        "model__wideresnet_k": MODEL__WIDERESNET_K,
        "model__wideresnet_dropout": MODEL__WIDERESNET_DROPOUT,
        # Training
        "n_epoch": N_EPOCH,
        "save_ckpt_every_n_epoch": SAVE_CKPT_EVERY_N_EPOCH,
        "load_ckpt": LOAD_CKPT,
        "load_ckpt_filepath": LOAD_CKPT_FILEPATH,
        "load_ckpt_epoch": LOAD_CKPT_EPOCH,
    })

    from collections import defaultdict
    import os

    import torch

    start_epoch_i, end_epoch_i = 0, N_EPOCH
    if LOAD_CKPT:
        start_epoch_i += LOAD_CKPT_EPOCH
        end_epoch_i += LOAD_CKPT_EPOCH
    for epoch_i in range(start_epoch_i, end_epoch_i):
        print(f'epoch: {epoch_i}')
        # Save checkpoint
        if epoch_i % SAVE_CKPT_EVERY_N_EPOCH == 0:
            checkpoint_filepath = f"checkpoints/{wandb.run.name}__epoch_{epoch_i}.pt"
            os.makedirs("checkpoints/", exist_ok=True)
            save_checkpoint(net, optimizer, checkpoint_filepath)
            wandb.save(checkpoint_filepath)

        ## Training Phase
        net.train()
        train_losses = []
        train_labels = []
        train_preds = []
        for minibatch_i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.float().to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            losses = criterion(outputs, labels)
            losses.mean().backward()
            optimizer.step()

            preds = torch.argmax(outputs, dim=1)
            train_losses.extend(losses.cpu().detach().tolist())
            train_labels.extend(labels.cpu().detach().tolist())
            train_preds.extend(preds.cpu().detach().tolist())

        train_losses = np.array(train_losses)
        train_labels = np.array(train_labels)
        train_preds = np.array(train_preds)

        # Filter losses by classes
        train_loss_per_class_dict = {
            f"train_loss__class_{class_}": train_losses[np.where(train_labels == class_)[0]].mean()
            for class_ in np.arange(5)
        }
        # Filter preds by classes for accuracy
        train_acc_per_class_dict = {
            f"train_acc__class_{class_}": (train_preds == train_labels)[np.where(train_labels == class_)[0]].mean()
            for class_ in np.arange(5)
        }

        ## Validation Phase
        net.eval()
        with torch.no_grad():
            # Save all losses and labels for each example
            valid_losses = []
            valid_labels = []
            valid_preds = []
            for minibatch_i, (inputs, labels) in enumerate(valid_loader):
                inputs = inputs.float().to(device)
                labels = labels.to(device)

                outputs = net(inputs)
                losses = criterion(outputs, labels)
                preds = torch.argmax(outputs, dim=1)

                valid_losses.extend(losses.cpu().detach().tolist())
                valid_labels.extend(labels.cpu().detach().tolist())
                valid_preds.extend(preds.cpu().detach().tolist())

        valid_losses = np.array(valid_losses)
        valid_labels = np.array(valid_labels)
        valid_preds = np.array(valid_preds)

        # Filter losses by classes
        valid_loss_per_class_dict = {
            f"valid_loss__class_{class_}": valid_losses[np.where(valid_labels == class_)[0]].mean()
            for class_ in np.arange(5)
        }
        # Filter preds by classes for accuracy
        valid_acc_per_class_dict = {
            f"valid_acc__class_{class_}": (valid_preds == valid_labels)[np.where(valid_labels == class_)[0]].mean()
            for class_ in np.arange(5)
        }

        # Logging
        wandb.log({
            "epoch_i": epoch_i,
            "train_loss": np.mean(train_losses),
            "train_acc": np.mean(train_preds == train_labels),
            **train_loss_per_class_dict,
            **train_acc_per_class_dict,
            "valid_loss": np.mean(valid_losses),
            "valid_acc": np.mean(valid_preds == valid_labels),
            **valid_loss_per_class_dict,
            **valid_acc_per_class_dict,
            "lr": optimizer.param_groups[0]['lr'],
        })
        if epoch_i in [30, 60]:
            scheduler.step()

    # Finish wandb run
    wandb_run.finish()

if __name__ == '__main__':
    train()