from collections import defaultdict
import logging
import os

import numpy as np
from omegaconf import OmegaConf

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms

from checkpointing import load_checkpoint, load_finished_epoch_from_checkpoint, save_checkpoint
from initializers import (
    compute_learning_rate,
    set_learning_rate,
    initialize_model,
    initialize_transforms,
    InputNormalize,
)
from replace_with_pure_noise import replace_with_pure_noise

from torchvision.datasets import CIFAR10, CIFAR100
from datasets.imbalanced_cifar import IMBALANCECIFAR10, IMBALANCECIFAR100
from datasets.sampling import count_class_frequency, compute_class_weights_on_effective_num_samples, compute_sample_weights
from models.noise_bn_option import NoiseBnOption

logging.getLogger().setLevel(logging.INFO)

DATA_ROOT = './data'

def train(CONFIG):
    # Wandb
    if CONFIG.enable_wandb:
        import wandb
        wandb.login()
        wandb_run = wandb.init(
            entity=CONFIG.wandb_entity,
            project=CONFIG.wandb_project,
            name=None if not CONFIG.wandb_name else CONFIG.wandb_name,
            config=OmegaConf.to_container(CONFIG),
        )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    ######################################### Dataset ############################################
    
    train_transform = initialize_transforms(CONFIG.train_transform_reprs)
    valid_transform = initialize_transforms(CONFIG.valid_transform_reprs)

    if CONFIG.dataset == 'CelebA-5':
        from datasets.celeba5 import build_train_dataset, build_valid_dataset
        train_dataset = build_train_dataset(transform=train_transform)
        valid_dataset = build_valid_dataset(transform=valid_transform)
        NUM_CLASSES = 5
    elif CONFIG.dataset == "CIFAR-10-LT":
        train_dataset = IMBALANCECIFAR10(root=DATA_ROOT, train=True, transform=train_transform, download=True, ir_ratio=CONFIG.ir_ratio)
        valid_dataset = CIFAR10(root=DATA_ROOT, train=False, transform=valid_transform, download=True)
        NUM_CLASSES = 10
    elif CONFIG.dataset == "CIFAR-10":
        train_dataset = CIFAR10(root=DATA_ROOT, train=True, transform=train_transform, download=True)
        valid_dataset = CIFAR10(root=DATA_ROOT, train=False, transform=valid_transform, download=True)
        NUM_CLASSES = 10
    elif CONFIG.dataset == "CIFAR-100-LT":
        train_dataset = IMBALANCECIFAR100(root=DATA_ROOT, train=True, transform=train_transform, download=True, ir_ratio=CONFIG.ir_ratio)
        valid_dataset = CIFAR100(root=DATA_ROOT, train=False, transform=valid_transform, download=True)
        NUM_CLASSES = 100

    print(f'Train dataset length: {len(train_dataset)}, Valid dataset length: {len(valid_dataset)}')

    ######################################### DataLoader ############################################
    
    train_default_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=CONFIG.batch_size,
        num_workers=CONFIG.num_workers,
        pin_memory=CONFIG.enable_pin_memory,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=CONFIG.batch_size,
        num_workers=CONFIG.num_workers,
        pin_memory=CONFIG.enable_pin_memory,
    )
    
    if CONFIG.enable_oversampling or CONFIG.enable_open:
        class_frequency = count_class_frequency(train_dataset.targets, NUM_CLASSES)
    
    if CONFIG.enable_oversampling:
        if CONFIG.oversample_majority_class_num_samples:
            num_samples = int(max(class_frequency) * NUM_CLASSES)
        else:
            num_samples = len(train_dataset)
        
        if CONFIG.oversample_use_effective_num_sample_weights:
            class_weights = compute_class_weights_on_effective_num_samples(class_frequency)
        else:
            class_weights = 1. / class_frequency
        sample_weights = compute_sample_weights(train_dataset.targets, class_weights)

        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=num_samples, # https://stackoverflow.com/a/67802529
            replacement=True,
        )
        train_oversampling_loader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            shuffle=False,
            batch_size=CONFIG.batch_size,
            num_workers=CONFIG.num_workers,
            pin_memory=CONFIG.enable_pin_memory,
        )
        logging.info(f"Initialized WeightedRandomSampler with weights {class_weights}")
        logging.info(f"From epoch {CONFIG.oversampling_start_epoch}, each epoch has {num_samples} samples.")

    ######################################### Model #########################################

    net = initialize_model(
        model_name=CONFIG.model, 
        num_classes=NUM_CLASSES, 
        noise_bn_option=NoiseBnOption[CONFIG.noise_bn_option],
        dropout_rate=CONFIG.dropout_rate)
    net = net.to(device)
    
    normalizer = InputNormalize(
        torch.Tensor(CONFIG.normalize_mean).to(device), 
        torch.Tensor(CONFIG.normalize_std).to(device)
    ).to(device)

    ######################################### Optimizer #########################################

    optimizer = optim.SGD(
        net.parameters(),
        lr=CONFIG.lr,
        momentum=CONFIG.momentum,
        weight_decay=CONFIG.weight_decay,
    )
    if CONFIG.use_adam:
        optimizer = optim.Adam(
            net.parameters(), 
            lr=CONFIG.lr, 
            betas=CONFIG.adam_betas,
            weight_decay=CONFIG.weight_decay)

    ######################################### Loss #########################################

    criterion = nn.CrossEntropyLoss(reduction="none")

    ######################################### Training #########################################

    if CONFIG.enable_open:
        num_samples_per_class = torch.Tensor(class_frequency).to(device)
        pure_noise_mean = torch.Tensor(CONFIG.pure_noise_mean).to(device)
        pure_noise_std = torch.Tensor(CONFIG.pure_noise_std).to(device)
    
    start_epoch_i = 0
    if CONFIG.load_ckpt:
        # Start epoch is one after the checkpointed epoch, because we checkpoint after finishing the epoch.
        start_epoch_i = load_finished_epoch_from_checkpoint(CONFIG.load_ckpt_filepath) + 1
        load_checkpoint(net, optimizer, CONFIG.load_ckpt_filepath)
    
    for epoch_i in range(start_epoch_i, CONFIG.num_epochs):
        print(f'epoch: {epoch_i}')
        
        # Update learning rate
        set_learning_rate(optimizer,
                          compute_learning_rate(
                              epoch=epoch_i,
                              default_lr=CONFIG.lr,
                              lr_decay=CONFIG.lr_decay,
                              lr_decay_epochs=CONFIG.lr_decay_epochs,
                              enable_linear_warmup=CONFIG.enable_linear_warmup,
                              enable_lr_decay=CONFIG.enable_lr_decay))

        # Choose dataloader
        if CONFIG.enable_oversampling and CONFIG.oversampling_start_epoch <= epoch_i:
            train_loader = train_oversampling_loader
        else:
            train_loader = train_default_loader

        ## Training Phase
        net.train()
        train_losses = []
        train_labels = []
        train_preds = []
        for minibatch_i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.float().to(device)
            labels = labels.to(device)
            
            inputs = normalizer(inputs)

            optimizer.zero_grad()
            if CONFIG.enable_open:
                if epoch_i < CONFIG.open_start_epoch:
                    noise_mask = None
                else:
                    noise_mask = replace_with_pure_noise(
                        images=inputs,
                        targets=labels,
                        delta=CONFIG.delta,
                        num_samples_per_class=num_samples_per_class,
                        dataset_mean=pure_noise_mean,
                        dataset_std=pure_noise_std,
                        image_size=CONFIG.pure_noise_image_size,
                    )
                outputs = net(inputs, noise_mask=noise_mask)
            else:
                outputs = net(inputs)
            losses = criterion(outputs, labels)
            losses.mean().backward()
            optimizer.step()

            preds = torch.argmax(outputs, dim=1)
            train_losses.extend(losses.cpu().detach().tolist())
            train_labels.extend(labels.cpu().detach().tolist())
            train_preds.extend(preds.cpu().detach().tolist())

            if CONFIG.debug_run and minibatch_i > 20:
                break

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
                
                inputs = normalizer(inputs)
                
                if CONFIG.enable_open:
                    noise_mask = torch.zeros(inputs.size(0), dtype=torch.bool).to(device)
                    outputs = net(inputs, noise_mask=noise_mask)
                else:
                    outputs = net(inputs)
                losses = criterion(outputs, labels)
                preds = torch.argmax(outputs, dim=1)

                valid_losses.extend(losses.cpu().detach().tolist())
                valid_labels.extend(labels.cpu().detach().tolist())
                valid_preds.extend(preds.cpu().detach().tolist())

                if CONFIG.debug_run and minibatch_i > 20:
                    break

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
        if CONFIG.enable_wandb:
            wandb.log(data={
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
            }, step=epoch_i)
        
        # Save checkpoint
        if CONFIG.save_ckpt and (epoch_i in CONFIG.save_ckpt_epochs):
            checkpoint_filepath = f"checkpoints/{wandb.run.name}__epoch_{epoch_i}.pt"
            os.makedirs("checkpoints/", exist_ok=True)
            save_checkpoint(net, optimizer, checkpoint_filepath, finished_epoch=epoch_i)
            wandb.save(checkpoint_filepath)

        if CONFIG.debug_run:
            break

    # Finish wandb run
    if CONFIG.enable_wandb:
        wandb_run.finish()

if __name__ == '__main__':
    DEFAULT_CONFIG_FILEPATH = "default_cifar10lt.yaml"

    CLI_CONFIG = OmegaConf.from_cli()
    if "config_filepath" in CLI_CONFIG:
        DEFAULT_CONFIG = OmegaConf.load(CLI_CONFIG.config_filepath)
        logging.info(f"Loaded config from {CLI_CONFIG.config_filepath}")
    else:
        CLI_CONFIG.config_filepath = DEFAULT_CONFIG_FILEPATH
        DEFAULT_CONFIG = OmegaConf.load(CLI_CONFIG.config_filepath)
        logging.info(f"No config specified. Loading config from {CLI_CONFIG.config_filepath}")

    CONFIG = OmegaConf.merge(DEFAULT_CONFIG, CLI_CONFIG)
    print(OmegaConf.to_yaml(CONFIG))

    train(CONFIG)
