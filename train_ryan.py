from collections import defaultdict
import logging
import os

import numpy as np
import wandb

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, WeightedRandomSampler

from checkpointing import load_checkpoint, save_checkpoint
from datasets.cifar10 import CIFAR10LTDataset
from ldam_drw_models import resnet32


def _get_cifar10lt_dataloaders(
    dataloader__batch_size,
    dataloader__num_workers,
):
    """
    Get CIFAR-10-LT training and validation dataloaders
    """
    # Create transforms
    # transforms.ConvertImageDtype() used as we use torchvision.io.read_image
    train_transform = transforms.Compose([
        transforms.ConvertImageDtype(torch.float32),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),
    ])
    valid_transform = transforms.Compose([
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),
    ])

    # Create datasets
    train_json_filepath = "data/json/cifar10_imbalance100/cifar10_imbalance100_train.json"
    train_images_dirpath = "data/json/cifar10_imbalance100/images/"
    train_dataset = CIFAR10LTDataset(
        json_filepath=train_json_filepath,
        images_dirpath=train_images_dirpath,
        transform=train_transform,
    )
    valid_json_filepath = "data/json/cifar10_imbalance100/cifar10_imbalance100_valid.json"
    valid_images_dirpath = "data/json/cifar10_imbalance100/images/"
    valid_dataset = CIFAR10LTDataset(
        json_filepath=valid_json_filepath,
        images_dirpath=valid_images_dirpath,
        transform=valid_transform,
    )

    # Create dataloaders
    train_sampler = WeightedRandomSampler(
        weights=train_dataset.sample_weights,
        num_samples=50000, # https://stackoverflow.com/a/67802529
        replacement=True,
    )
    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        # shuffle=True,
        batch_size=dataloader__batch_size,
        num_workers=dataloader__num_workers,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=dataloader__batch_size,
        num_workers=dataloader__num_workers,
    )
    
    return train_loader, valid_loader


def _get_resnet32_model():
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    net = resnet32()
    logging.info(f"Loaded ResNet-32 model with {count_parameters(net)} parameters")

    return net


def _get_optimizer(
    net,
    optim__lr,
    optim__momentum,
    optim__weight_decay,
):
    optimizer = optim.SGD(
        net.parameters(),
        lr=optim__lr,
        momentum=optim__momentum,
        weight_decay=optim__weight_decay,
    )

    return optimizer


def _get_lr_scheduler(optimizer):
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        # NOTE(ryanlee): 0 causes ZeroDivisionError
        start_factor=torch.finfo().tiny,
        end_factor=1,
        total_iters=5,
    )
    multistep_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[160,180],
        gamma=0.01,
    )
    scheduler = optim.lr_scheduler.ChainedScheduler([
        warmup_scheduler,
        multistep_scheduler,
    ])

    return scheduler


def main():
    # TODO (ryanlee): Use OmegaConf
    DATALOADER__BATCH_SIZE = 128
    DATALOADER__NUM_WORKERS = 8
    OPTIM__LR = 0.1
    OPTIM__MOMENTUM = 0.9
    OPTIM__WEIGHT_DECAY = 2e-4
    
    # Training Hyperparameters
    N_EPOCH = 200
    SAVE_CKPT_EVERY_N_EPOCH = 10
    LOAD_CKPT = False
    LOAD_CKPT_FILEPATH = "checkpoints/.pt"
    LOAD_CKPT_EPOCH = 0

    # Setup wandb
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
        # Training
        "n_epoch": N_EPOCH,
        "save_ckpt_every_n_epoch": SAVE_CKPT_EVERY_N_EPOCH,
        "load_ckpt": LOAD_CKPT,
        "load_ckpt_filepath": LOAD_CKPT_FILEPATH,
        "load_ckpt_epoch": LOAD_CKPT_EPOCH,
    })

    # Setup components
    train_loader, valid_loader = _get_cifar10lt_dataloaders(
        dataloader__batch_size=DATALOADER__BATCH_SIZE,
        dataloader__num_workers=DATALOADER__NUM_WORKERS,
    )
    net = _get_resnet32_model()
    net = net.cuda()
    optimizer = _get_optimizer(
        net,
        optim__lr=OPTIM__LR,
        optim__momentum=OPTIM__MOMENTUM,
        optim__weight_decay=OPTIM__WEIGHT_DECAY,
    )
    scheduler = _get_lr_scheduler(optimizer)
    criterion = nn.CrossEntropyLoss(reduction="none")

    start_epoch_i, end_epoch_i = 0, N_EPOCH
    if LOAD_CKPT:
        load_checkpoint(net, optimizer, LOAD_CKPT_FILEPATH)
        start_epoch_i += LOAD_CKPT_EPOCH
        end_epoch_i += LOAD_CKPT_EPOCH

    # Training
    for epoch_i in range(start_epoch_i, end_epoch_i):
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
            inputs = inputs.float().cuda()
            labels = labels.cuda()

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
            for class_ in np.arange(10)
        }
        # Filter preds by classes for accuracy
        train_acc_per_class_dict = {
            f"train_acc__class_{class_}": (train_preds == train_labels)[np.where(train_labels == class_)[0]].mean()
            for class_ in np.arange(10)
        }

        ## Validation Phase
        net.eval()
        with torch.no_grad():
            # Save all losses and labels for each example
            valid_losses = []
            valid_labels = []
            valid_preds = []
            for minibatch_i, (inputs, labels) in enumerate(valid_loader):
                inputs = inputs.float().cuda()
                labels = labels.cuda()

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
            for class_ in np.arange(10)
        }
        # Filter preds by classes for accuracy
        valid_acc_per_class_dict = {
            f"valid_acc__class_{class_}": (valid_preds == valid_labels)[np.where(valid_labels == class_)[0]].mean()
            for class_ in np.arange(10)
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
            "lr": scheduler.get_last_lr()[0],
        })
        scheduler.step()

    # Save the last epoch
    checkpoint_filepath = f"checkpoints/{wandb.run.name}__epoch_{end_epoch_i}.pt"
    save_checkpoint(net, optimizer, checkpoint_filepath)
    wandb.save(checkpoint_filepath)

    # Finish wandb run
    wandb_run.finish()


if __name__ == "__main__":
    main()
