## 512
WANDB_NAME="batch_size-bs512"
python train.py wandb_name="${WANDB_NAME}" \
batch_size=512

python train.py wandb_name="${WANDB_NAME}-drs" \
load_ckpt=True load_ckpt_filepath="checkpoints/${WANDB_NAME}__epoch_159.pt" \
enable_oversampling=True \
batch_size=512

python train.py wandb_name="${WANDB_NAME}-open" \
load_ckpt=True load_ckpt_filepath="checkpoints/${WANDB_NAME}__epoch_159.pt" \
enable_oversampling=True enable_open=True \
batch_size=512

## 256
WANDB_NAME="batch_size-bs256"
python train.py wandb_name="${WANDB_NAME}" \
batch_size=256

python train.py wandb_name="${WANDB_NAME}-drs" \
load_ckpt=True load_ckpt_filepath="checkpoints/${WANDB_NAME}__epoch_159.pt" \
enable_oversampling=True \
batch_size=256

python train.py wandb_name="${WANDB_NAME}-open" \
load_ckpt=True load_ckpt_filepath="checkpoints/${WANDB_NAME}__epoch_159.pt" \
enable_oversampling=True enable_open=True \
batch_size=256

## 64
WANDB_NAME="batch_size-bs64"
python train.py wandb_name="${WANDB_NAME}" \
batch_size=64

python train.py wandb_name="${WANDB_NAME}-drs" \
load_ckpt=True load_ckpt_filepath="checkpoints/${WANDB_NAME}__epoch_159.pt" \
enable_oversampling=True \
batch_size=64

python train.py wandb_name="${WANDB_NAME}-open" \
load_ckpt=True load_ckpt_filepath="checkpoints/${WANDB_NAME}__epoch_159.pt" \
enable_oversampling=True enable_open=True \
batch_size=64

## 32
WANDB_NAME="batch_size-bs32"
python train.py wandb_name="${WANDB_NAME}" \
batch_size=32

python train.py wandb_name="${WANDB_NAME}-drs" \
load_ckpt=True load_ckpt_filepath="checkpoints/${WANDB_NAME}__epoch_159.pt" \
enable_oversampling=True \
batch_size=32

python train.py wandb_name="${WANDB_NAME}-open" \
load_ckpt=True load_ckpt_filepath="checkpoints/${WANDB_NAME}__epoch_159.pt" \
enable_oversampling=True enable_open=True \
batch_size=32
