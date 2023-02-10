# CIFAR10
WANDB_NAME='search_inputnorm-cifar10'
python train.py wandb_name="${WANDB_NAME}" \
"normalize_mean=[0.4914, 0.4822, 0.4465]" \
"normalize_std=[0.2470, 0.2435, 0.2616]"

python train.py wandb_name="${WANDB_NAME}-drs" \
load_ckpt=True load_ckpt_filepath="checkpoints/${WANDB_NAME}__epoch_159.pt" \
enable_oversampling=True \
"normalize_mean=[0.4914, 0.4822, 0.4465]" \
"normalize_std=[0.2470, 0.2435, 0.2616]"

python train.py wandb_name="${WANDB_NAME}-open" \
load_ckpt=True load_ckpt_filepath="checkpoints/${WANDB_NAME}__epoch_159.pt" \
enable_oversampling=True enable_open=True \
"normalize_mean=[0.4914, 0.4822, 0.4465]" \
"normalize_std=[0.2470, 0.2435, 0.2616]"

python train.py wandb_name="${WANDB_NAME}-cifar10ltir100" \
enable_oversampling=True enable_open=True \

# CIFAR10 (IR100)
WANDB_NAME='search_inputnorm-cifar10ltir100'
python train.py wandb_name="${WANDB_NAME}" \
"normalize_mean=[0.4989, 0.5044, 0.4926]" \
"normalize_std=[0.2513, 0.2485, 0.2734]"

python train.py wandb_name="${WANDB_NAME}-drs" \
load_ckpt=True load_ckpt_filepath="checkpoints/${WANDB_NAME}__epoch_159.pt" \
enable_oversampling=True \
"normalize_mean=[0.4989, 0.5044, 0.4926]" \
"normalize_std=[0.2513, 0.2485, 0.2734]"

python train.py wandb_name="${WANDB_NAME}-open" \
load_ckpt=True load_ckpt_filepath="checkpoints/${WANDB_NAME}__epoch_159.pt" \
enable_oversampling=True enable_open=True \
"normalize_mean=[0.4989, 0.5044, 0.4926]" \
"normalize_std=[0.2513, 0.2485, 0.2734]"
