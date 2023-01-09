
def train(args):
    import torch

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    ######################################### Dataset ############################################
    from torchvision import transforms

    if args.dataset == 'CelebA-5':
        NUM_CLASSES = 5
        
        from datasets.celeba5 import build_train_dataset, build_valid_dataset
        train_dataset_builder = build_train_dataset
        valid_dataset_builder = build_valid_dataset

    ## Find dataset statistics with and without augmentation.
    # import custom_transforms

    # TODO: argparse.
    train_transform = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        # custom_transforms.Cutout(n_holes=1, length=16),
        # # TODO: Check if this is correct values for SIMCLR augmentation
        # transforms.RandomApply([
        #     transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        # ], p=0.8),
        # transforms.RandomGrayscale(p=0.2),
        # transforms.RandomApply([
        #     transforms.GaussianBlur(kernel_size=3, sigma=[.1, 2.])
        # ], p=0.5),
    ]
    valid_transform = [
        transforms.ToTensor(),
    ]

    from datasets.dataset_statistics import dataset_mean_and_std

    def train_mean_and_std():
        dataset = build_train_dataset(transforms.Compose(train_transform))
        return dataset_mean_and_std(dataset)
    
    train_dataset_mean, train_dataset_std = train_mean_and_std()
    print(f'Train dataset mean: {train_dataset_mean} std: {train_dataset_std}')

    def valid_mean_and_std():
        dataset = build_valid_dataset(transforms.Compose(valid_transform))
        return dataset_mean_and_std(dataset)
    
    valid_dataset_mean, valid_dataset_std = valid_mean_and_std()
    print(f'Valid dataset mean: {valid_dataset_mean} std: {valid_dataset_std}')

    train_dataset = build_train_dataset(
        transform=transforms.Compose(
            train_transform + [transforms.Normalize(train_dataset_mean, train_dataset_std)]
        )
    )
    valid_dataset = build_valid_dataset(
        transform=transforms.Compose(
            valid_transform + [transforms.Normalize(valid_dataset_mean, valid_dataset_std)]
        )
    )
    print(f'Train dataset length: {len(train_dataset)}, Valid dataset length: {len(valid_dataset)}')

    ######################################### DataLoader ############################################
    
    # DataLoader Hyperparameters
    DATALOADER__NUM_WORKERS = args.num_workers
    DATALOADER__BATCH_SIZE = args.batch_size

    # Compute weights
    import numpy as np

    sample_labels = []
    sample_labels_count = np.arange(NUM_CLASSES)
    for _, label in train_dataset:
        sample_labels.append(label)
        sample_labels_count[label] += 1
    weights = 1. / sample_labels_count
    sample_weights = np.array([weights[l] for l in sample_labels])
    print(f'Class weights: {weights}')

    from torch.utils.data import DataLoader, WeightedRandomSampler

    train_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset), # https://stackoverflow.com/a/67802529
        replacement=True,
    )
    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler if args.use_oversampling else None,
        shuffle=False if args.use_oversampling else True,
        batch_size=DATALOADER__BATCH_SIZE,
        num_workers=DATALOADER__NUM_WORKERS,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=DATALOADER__BATCH_SIZE,
        num_workers=DATALOADER__NUM_WORKERS,
    )

    ######################################### Model #########################################

    # Model hyperparameters
    MODEL__WIDERESNET_DEPTH = 28
    MODEL__WIDERESNET_K = 10
    MODEL__WIDERESNET_DROPOUT = args.dropout

    from networks_torchdistill import WideBasicBlock, WideResNet

    net = WideResNet(
        depth=MODEL__WIDERESNET_DEPTH,
        k=MODEL__WIDERESNET_K,
        dropout_p=MODEL__WIDERESNET_DROPOUT,
        block=WideBasicBlock,
        num_classes=NUM_CLASSES,
    )

    net = net.to(device)

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

    print(f'Model parameter count: {count_parameters(net)}')

    ######################################### Optimizer #########################################

    # Optimizer Hyperparameters
    OPTIM__LR = args.lr
    OPTIM__MOMENTUM = args.momentum
    OPTIM__WEIGHT_DECAY = args.weight_decay

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
        gamma=args.lr_decay,
    )

    ######################################### Loss #########################################
    import torch.nn as nn

    criterion = nn.CrossEntropyLoss(reduction="none")

    ######################################### Logging & Checkpoint #########################################

    # Training Hyperparameters
    N_EPOCH = args.num_epochs
    SAVE_CKPT_EVERY_N_EPOCH = args.save_ckpt_every_n_epoch
    LOAD_CKPT = args.load_ckpt
    LOAD_CKPT_FILEPATH = args.load_ckpt_filepath
    LOAD_CKPT_EPOCH = args.load_ckpt_epoch

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

    ## Wandb
    import wandb
    wandb.login()

    wandb_run = wandb.init(
        project="pure-noise",
        entity="brianryan",
    )

    wandb.config.update({
        # Data
        "dataloader__num_workers": DATALOADER__NUM_WORKERS,
        "dataloader__batch_size": DATALOADER__BATCH_SIZE,
        "dataloader__use_oversampling": args.use_oversampling,
        # Optimizer
        "optim__lr": OPTIM__LR,
        "optim__momentum": OPTIM__MOMENTUM,
        "optim__weight_decay": OPTIM__WEIGHT_DECAY,
        "optim__lr_decay": args.lr_decay,
        "optim__lr_decay_epochs": args.lr_decay_epochs,
        # Model
        "model__wideresnet_depth": MODEL__WIDERESNET_DEPTH,
        "model__wideresnet_k": MODEL__WIDERESNET_K,
        "model__wideresnet_dropout": MODEL__WIDERESNET_DROPOUT,
        # Checkpoint
        "save_ckpt_every_n_epoch": SAVE_CKPT_EVERY_N_EPOCH,
        "load_ckpt": LOAD_CKPT,
        "load_ckpt_filepath": LOAD_CKPT_FILEPATH,
        "load_ckpt_epoch": LOAD_CKPT_EPOCH,
        # Training
        "n_epoch": N_EPOCH,
    })

    ######################################### Training #########################################

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
        if epoch_i in args.lr_decay_epochs:
            scheduler.step()

    # Finish wandb run
    wandb_run.finish()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument('--dataset', default='CelebA-5', choices=['CIFAR-10-LT', 'CelebA-5'], type=str)

    # DataLoader
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--use_oversampling', default=False, type=bool)

    # Model
    parser.add_argument('--model', default='WideResNet-28-10', choices=['WideResNet-28-10', 'ResNet-32'], type=str)
    parser.add_argument('--dropout', default=0.3, type=float)

    # Optimizer
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=2e-4, type=float)
    parser.add_argument('--lr_decay', default=0.1, type=float)
    parser.add_argument('--lr_decay_epochs', default=[30, 60], type=int, nargs='*')

    # Checkpoint
    parser.add_argument('--save_ckpt_every_n_epoch', default=10, type=int)
    parser.add_argument('--load_ckpt', default=False, type=bool)
    parser.add_argument('--load_ckpt_filepath', default='checkpoints/.pt', type=str)
    parser.add_argument('--load_ckpt_epoch', default=0, type=int)

    # Training
    parser.add_argument('--num_epochs', default=200, type=int)

    args = parser.parse_args()
    train(args)