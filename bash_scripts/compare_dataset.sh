## Cui et al.
WANDB_NAME="compare_dataset-cui"

python train.py wandb_name="${WANDB_NAME}"

python train.py wandb_name="${WANDB_NAME}-drs" \
load_ckpt=True load_ckpt_filepath="checkpoints/${WANDB_NAME}__epoch_159.pt" \
enable_oversampling=True \
use_subset_to_train=True train_subset_filepath="cui_indices.txt"

python train.py wandb_name="${WANDB_NAME}-open" \
load_ckpt=True load_ckpt_filepath="checkpoints/${WANDB_NAME}__epoch_159.pt" \
enable_oversampling=True enable_open=True \
use_subset_to_train=True train_subset_filepath="cui_indices.txt"

## LDAM
WANDB_NAME="compare_dataset-ldam"

python train.py wandb_name="${WANDB_NAME}"

python train.py wandb_name="${WANDB_NAME}-drs" \
load_ckpt=True load_ckpt_filepath="checkpoints/${WANDB_NAME}__epoch_159.pt" \
enable_oversampling=True \
use_subset_to_train=True train_subset_filepath="ldam_indices.txt"

python train.py wandb_name="${WANDB_NAME}-open" \
load_ckpt=True load_ckpt_filepath="checkpoints/${WANDB_NAME}__epoch_159.pt" \
enable_oversampling=True enable_open=True \
use_subset_to_train=True train_subset_filepath="ldam_indices.txt"

## M2m
WANDB_NAME="compare_dataset-m2m"

python train.py wandb_name="${WANDB_NAME}"

python train.py wandb_name="${WANDB_NAME}-drs" \
load_ckpt=True load_ckpt_filepath="checkpoints/${WANDB_NAME}__epoch_159.pt" \
enable_oversampling=True \
use_subset_to_train=True train_subset_filepath="m2m_indices.txt"

python train.py wandb_name="${WANDB_NAME}-open" \
load_ckpt=True load_ckpt_filepath="checkpoints/${WANDB_NAME}__epoch_159.pt" \
enable_oversampling=True enable_open=True \
use_subset_to_train=True train_subset_filepath="m2m_indices.txt"
