# CIFAR10LT: IR=50,100
# CIFAR100-LT: IR=50,100
# CELEBA5

WANDB_NAME="CIFAR10IR50"

python train.py dataset="CIFAR-10-LT" ir_ratio=50 wandb_name="${WANDB_NAME}"

python train.py dataset="CIFAR-10-LT" ir_ratio=50 wandb_name="${WANDB_NAME}-rs" \
load_ckpt=True load_ckpt_filepath="checkpoints/${WANDB_NAME}__epoch_159.pt" \
enable_oversampling=True

python train.py dataset="CIFAR-10-LT" ir_ratio=50 wandb_name="${WANDB_NAME}-open" \
load_ckpt=True load_ckpt_filepath="checkpoints/${WANDB_NAME}__epoch_159.pt" \
enable_oversampling=True enable_open=True

WANDB_NAME="CIFAR100IR100"

python train.py dataset="CIFAR-100-LT" ir_rato=100 wandb_name="${WANDB_NAME}"

python train.py dataset="CIFAR-100-LT" ir_rato=100 wandb_name="${WANDB_NAME}-rs" \
load_ckpt=True load_ckpt_filepath="checkpoints/${WANDB_NAME}__epoch_159.pt" \
enable_oversampling=True

python train.py dataset="CIFAR-100-LT" ir_rato=100 wandb_name="${WANDB_NAME}-open" \
load_ckpt=True load_ckpt_filepath="checkpoints/${WANDB_NAME}__epoch_159.pt" \
enable_oversampling=True enable_open=True

WANDB_NAME="CIFAR100IR50"

python train.py dataset="CIFAR-100-LT" ir_rato=50 wandb_name="${WANDB_NAME}"

python train.py dataset="CIFAR-100-LT" ir_rato=50 wandb_name="${WANDB_NAME}-rs" \
load_ckpt=True load_ckpt_filepath="checkpoints/${WANDB_NAME}__epoch_159.pt" \
enable_oversampling=True

python train.py dataset="CIFAR-100-LT" ir_rato=50 wandb_name="${WANDB_NAME}-open" \
load_ckpt=True load_ckpt_filepath="checkpoints/${WANDB_NAME}__epoch_159.pt" \
enable_oversampling=True enable_open=True