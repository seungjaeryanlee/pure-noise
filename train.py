from collections import defaultdict
import os

import numpy as np
from omegaconf import OmegaConf

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler

from checkpointing import load_checkpoint, save_checkpoint
from initializers import (
    initialize_lr_scheduler,
    initialize_model,
    initialize_transforms,
) 


def train(CONFIG):
    # Wandb
    if not CONFIG.disable_wandb:
        import wandb
        wandb.login()
        wandb_run = wandb.init(
            project="pure-noise",
            entity="brianryan",
            config=OmegaConf.to_container(CONFIG),
        )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    ######################################### Dataset ############################################
    from torchvision import transforms

    if CONFIG.dataset == 'CelebA-5':
        NUM_CLASSES = 5
         
        from datasets.celeba5 import build_train_dataset, build_valid_dataset
        train_dataset_builder = build_train_dataset
        valid_dataset_builder = build_valid_dataset

    ## Find dataset statistics with and without augmentation.
    # import custom_transforms

    train_transform = initialize_transforms(CONFIG.train_transform_repr)
    valid_transform = initialize_transforms(CONFIG.valid_transform_repr)

    from datasets.dataset_statistics import dataset_mean_and_std
    def train_mean_and_std():
        dataset = build_train_dataset(transforms.Compose(train_transform))
        return dataset_mean_and_std(dataset)
    def valid_mean_and_std():
        dataset = build_valid_dataset(transforms.Compose(valid_transform))
        return dataset_mean_and_std(dataset)

    train_dataset_mean, train_dataset_std = train_mean_and_std()
    print(f'Train dataset mean: {train_dataset_mean} std: {train_dataset_std}')
    valid_dataset_mean, valid_dataset_std = valid_mean_and_std()
    print(f'Valid dataset mean: {valid_dataset_mean} std: {valid_dataset_std}')

    print(f'Normalize using train stats: {CONFIG.normalize_using_train_stats}')
    mean = train_dataset_mean if CONFIG.normalize_using_train_stats else valid_dataset_mean
    std = train_dataset_std if CONFIG.normalize_using_train_stats else valid_dataset_std
    train_dataset = build_train_dataset(transform=transforms.Compose(train_transform + [transforms.Normalize(mean, std)]))
    valid_dataset = build_valid_dataset(transform=transforms.Compose(valid_transform + [transforms.Normalize(mean, std)]))

    print(f'Train dataset length: {len(train_dataset)}, Valid dataset length: {len(valid_dataset)}')
    print(f"Train dataset class weights: {train_dataset.weights}")

    ######################################### DataLoader ############################################

    train_sampler = WeightedRandomSampler(
        weights=train_dataset.sample_weights,
        num_samples=len(train_dataset), # https://stackoverflow.com/a/67802529
        replacement=True,
    )
    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler if CONFIG.use_oversampling else None,
        shuffle=False if CONFIG.use_oversampling else True,
        batch_size=CONFIG.batch_size,
        num_workers=CONFIG.num_workers,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=CONFIG.batch_size,
        num_workers=CONFIG.num_workers,
    )

    ######################################### Model #########################################

    net = initialize_model(model_name=CONFIG.model, num_classes=NUM_CLASSES)
    net = net.to(device)

    ######################################### Optimizer #########################################

    optimizer = optim.SGD(
        net.parameters(),
        lr=CONFIG.lr,
        momentum=CONFIG.momentum,
        weight_decay=CONFIG.weight_decay,
    )
    scheduler = initialize_lr_scheduler(
        optimizer,
        enable_linear_warmup=CONFIG.enable_linear_warmup,
        lr_decay=CONFIG.lr_decay,
        lr_decay_epochs=CONFIG.lr_decay_epochs,
    )

    ######################################### Loss #########################################

    criterion = nn.CrossEntropyLoss(reduction="none")

    ######################################### Training #########################################

    start_epoch_i, end_epoch_i = 0, CONFIG.num_epochs
    if CONFIG.load_ckpt:
        load_checkpoint(net, optimizer, CONFIG.load_ckpt_filepath)
        start_epoch_i += LOAD_CKPT_EPOCH
        end_epoch_i += LOAD_CKPT_EPOCH

    for epoch_i in range(start_epoch_i, end_epoch_i):
        print(f'epoch: {epoch_i}')
        # Save checkpoint
        # TODO: fix checkpoint error.
        if CONFIG.enable_checkpoint and (epoch_i % CONFIG.save_ckpt_every_n_epoch == 0):
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
            for class_ in np.arange(NUM_CLASSES)
        }
        # Filter preds by classes for accuracy
        train_acc_per_class_dict = {
            f"train_acc__class_{class_}": (train_preds == train_labels)[np.where(train_labels == class_)[0]].mean()
            for class_ in np.arange(NUM_CLASSES)
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
            for class_ in np.arange(NUM_CLASSES)
        }
        # Filter preds by classes for accuracy
        valid_acc_per_class_dict = {
            f"valid_acc__class_{class_}": (valid_preds == valid_labels)[np.where(valid_labels == class_)[0]].mean()
            for class_ in np.arange(NUM_CLASSES)
        }

        # Logging
        if not CONFIG.disable_wandb:
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

        scheduler.step()

    # Finish wandb run
    if not CONFIG.disable_wandb:
        wandb_run.finish()


if __name__ == '__main__':
    DEFAULT_CONFIG = OmegaConf.load("default2.yaml")
    CLI_CONFIG = OmegaConf.from_cli()
    CONFIG = OmegaConf.merge(DEFAULT_CONFIG, CLI_CONFIG)

    train(CONFIG)
