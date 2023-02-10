## Cui et al.
WANDB_NAME="compare_dataset-cui"

python train.py wandb_name="${WANDB_NAME}" \
use_subset_to_train=True train_subset_filepath="cifar10ir100_indices_cui.txt"

python train.py wandb_name="${WANDB_NAME}-drs" \
load_ckpt=True load_ckpt_filepath="checkpoints/${WANDB_NAME}__epoch_159.pt" \
enable_oversampling=True \
use_subset_to_train=True train_subset_filepath="cifar10ir100_indices_cui.txt"

python train.py wandb_name="${WANDB_NAME}-open" \
load_ckpt=True load_ckpt_filepath="checkpoints/${WANDB_NAME}__epoch_159.pt" \
enable_oversampling=True enable_open=True \
use_subset_to_train=True train_subset_filepath="cifar10ir100_indices_cui.txt"

## LDAM
WANDB_NAME="compare_dataset-ldam"

python train.py wandb_name="${WANDB_NAME}" \
use_subset_to_train=True train_subset_filepath="cifar10ir100_indices_ldam.txt"

python train.py wandb_name="${WANDB_NAME}-drs" \
load_ckpt=True load_ckpt_filepath="checkpoints/${WANDB_NAME}__epoch_159.pt" \
enable_oversampling=True \
use_subset_to_train=True train_subset_filepath="cifar10ir100_indices_ldam.txt"

python train.py wandb_name="${WANDB_NAME}-open" \
load_ckpt=True load_ckpt_filepath="checkpoints/${WANDB_NAME}__epoch_159.pt" \
enable_oversampling=True enable_open=True \
use_subset_to_train=True train_subset_filepath="cifar10ir100_indices_ldam.txt"

## M2m
WANDB_NAME="compare_dataset-m2m"

python train.py wandb_name="${WANDB_NAME}" \
use_subset_to_train=True train_subset_filepath="cifar10ir100_indices_m2m.txt"

python train.py wandb_name="${WANDB_NAME}-drs" \
load_ckpt=True load_ckpt_filepath="checkpoints/${WANDB_NAME}__epoch_159.pt" \
enable_oversampling=True \
use_subset_to_train=True train_subset_filepath="cifar10ir100_indices_m2m.txt"

python train.py wandb_name="${WANDB_NAME}-open" \
load_ckpt=True load_ckpt_filepath="checkpoints/${WANDB_NAME}__epoch_159.pt" \
enable_oversampling=True enable_open=True \
use_subset_to_train=True train_subset_filepath="cifar10ir100_indices_m2m.txt"
