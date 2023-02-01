WANDB_NAME='normalizer-resnet-0131'

# ERM
python train.py wandb_name="${WANDB_NAME}"

# Oversampling
python train.py wandb_name="${WANDB_NAME}-rs" \
load_ckpt=True load_ckpt_filepath="checkpoints/${WANDB_NAME}__epoch_159.pt" \
enable_oversampling=True

# OPeN
python train.py wandb_name="${WANDB_NAME}-open" \
load_ckpt=True load_ckpt_filepath="checkpoints/${WANDB_NAME}__epoch_159.pt" \
enable_oversampling=True enable_open=True