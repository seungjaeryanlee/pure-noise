WANDB_NAME='Delta'

python train.py wandb_name="${WANDB_NAME}" delta=0.33333 \
enable_oversampling=True enable_open=True

python train.py wandb_name="${WANDB_NAME}-0.66666" delta=0.66666 \
load_ckpt=True load_ckpt_filepath="checkpoints/${WANDB_NAME}__epoch_159.pt" \
enable_oversampling=True enable_open=True

python train.py wandb_name="${WANDB_NAME}-1.0" delta=1.0 \
load_ckpt=True load_ckpt_filepath="checkpoints/${WANDB_NAME}__epoch_159.pt" \
enable_oversampling=True enable_open=True

python train.py wandb_name="${WANDB_NAME}-0.5" delta=0.5 \
load_ckpt=True load_ckpt_filepath="checkpoints/${WANDB_NAME}__epoch_159.pt" \
enable_oversampling=True enable_open=True

python train.py wandb_name="${WANDB_NAME}-0.16666" delta=0.16666 \
load_ckpt=True load_ckpt_filepath="checkpoints/${WANDB_NAME}__epoch_159.pt" \
enable_oversampling=True enable_open=True
