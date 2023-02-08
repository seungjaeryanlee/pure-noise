# Reproduce Table 1.
# Compare ERM, (non-deferred) oversampling, and OPeN on
# CIFAR-10-LT with IR=50,100 and CIFAR-100-LT with IR=50,100.

WANDB_NAME="CIFAR10IR100"

python train.py dataset="CIFAR-10-LT" ir_ratio=100 wandb_name="${WANDB_NAME}"

python train.py dataset="CIFAR-10-LT" ir_ratio=100 wandb_name="${WANDB_NAME}-rs" \
enable_oversampling=True oversampling_start_epoch=0

python train.py dataset="CIFAR-10-LT" ir_ratio=100 wandb_name="${WANDB_NAME}-open" \
load_ckpt=True load_ckpt_filepath="checkpoints/${WANDB_NAME}__epoch_159.pt" \
enable_oversampling=True enable_open=True

WANDB_NAME="CIFAR10IR50"

python train.py dataset="CIFAR-10-LT" ir_ratio=50 wandb_name="${WANDB_NAME}"

python train.py dataset="CIFAR-10-LT" ir_ratio=50 wandb_name="${WANDB_NAME}-rs" \
enable_oversampling=True oversampling_start_epoch=0

python train.py dataset="CIFAR-10-LT" ir_ratio=50 wandb_name="${WANDB_NAME}-open" \
load_ckpt=True load_ckpt_filepath="checkpoints/${WANDB_NAME}__epoch_159.pt" \
enable_oversampling=True enable_open=True

WANDB_NAME="CIFAR100IR100"

python train.py dataset="CIFAR-100-LT" ir_ratio=100 wandb_name="${WANDB_NAME}"

python train.py dataset="CIFAR-100-LT" ir_ratio=100 wandb_name="${WANDB_NAME}-rs" \
enable_oversampling=True oversampling_start_epoch=0

python train.py dataset="CIFAR-100-LT" ir_ratio=100 wandb_name="${WANDB_NAME}-open" \
load_ckpt=True load_ckpt_filepath="checkpoints/${WANDB_NAME}__epoch_159.pt" \
enable_oversampling=True enable_open=True

WANDB_NAME="CIFAR100IR50"

python train.py dataset="CIFAR-100-LT" ir_ratio=50 wandb_name="${WANDB_NAME}"

python train.py dataset="CIFAR-100-LT" ir_ratio=50 wandb_name="${WANDB_NAME}-rs" \
enable_oversampling=True oversampling_start_epoch=0

python train.py dataset="CIFAR-100-LT" ir_ratio=50 wandb_name="${WANDB_NAME}-open" \
load_ckpt=True load_ckpt_filepath="checkpoints/${WANDB_NAME}__epoch_159.pt" \
enable_oversampling=True enable_open=True
