### Reproduce Figure 3 (data augmentation ablation study) in the paper.
# Model: WideResNet 
# Dataset: CIFAR-10-LT

### HorizontalFlip + RandomCrop
WANDB_NAME='flipcrop-0201'

# ERM
python train.py wandb_name="${WANDB_NAME}" \
'train_transform_reprs=["RandomHorizontalFlip()","RandomCrop(32, padding=4)","ToTensor()"]'

# Oversampling
python train.py wandb_name="${WANDB_NAME}-rs" \
'train_transform_reprs=["RandomHorizontalFlip()","RandomCrop(32, padding=4)","ToTensor()"]' \
load_ckpt=True load_ckpt_filepath="checkpoints/${WANDB_NAME}__epoch_159.pt" \
enable_oversampling=True

# OPeN
python train.py wandb_name="${WANDB_NAME}-open" \
'train_transform_reprs=["RandomHorizontalFlip()","RandomCrop(32, padding=4)","ToTensor()"]' \
load_ckpt=True load_ckpt_filepath="checkpoints/${WANDB_NAME}__epoch_159.pt" \
enable_oversampling=True enable_open=True

### Add CutOut
WANDB_NAME='cutout-0201'

# ERM
python train.py wandb_name="${WANDB_NAME}" \
'train_transform_reprs=["RandomHorizontalFlip()","RandomCrop(32, padding=4)","ToTensor()","Cutout(n_holes=1, length=16)"]'

# Oversampling
python train.py wandb_name="${WANDB_NAME}-rs" \
'train_transform_reprs=["RandomHorizontalFlip()","RandomCrop(32, padding=4)","ToTensor()","Cutout(n_holes=1, length=16)"]' \
load_ckpt=True load_ckpt_filepath="checkpoints/${WANDB_NAME}__epoch_159.pt" \
enable_oversampling=True

# OPeN
python train.py wandb_name="${WANDB_NAME}-open" \
'train_transform_reprs=["RandomHorizontalFlip()","RandomCrop(32, padding=4)","ToTensor()","Cutout(n_holes=1, length=16)"]' \
load_ckpt=True load_ckpt_filepath="checkpoints/${WANDB_NAME}__epoch_159.pt" \
enable_oversampling=True enable_open=True

### Add CutOut + SimCLR
WANDB_NAME='simclr-0201'

# ERM
python train.py wandb_name="${WANDB_NAME}" \
'train_transform_reprs=["RandomHorizontalFlip()","RandomCrop(32, padding=4)","ToTensor()","Cutout(n_holes=1, length=16)","RandomApply([ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)","RandomGrayscale(p=0.2)","RandomApply([GaussianBlur(kernel_size=3, sigma=[.1, 2.])], p=0.5)"]'

# Oversampling
python train.py wandb_name="${WANDB_NAME}-rs" \
'train_transform_reprs=["RandomHorizontalFlip()","RandomCrop(32, padding=4)","ToTensor()","Cutout(n_holes=1, length=16)","RandomApply([ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)","RandomGrayscale(p=0.2)","RandomApply([GaussianBlur(kernel_size=3, sigma=[.1, 2.])], p=0.5)"]' \
load_ckpt=True load_ckpt_filepath="checkpoints/${WANDB_NAME}__epoch_159.pt" \
enable_oversampling=True

# OPeN
python train.py wandb_name="${WANDB_NAME}-open" \
'train_transform_reprs=["RandomHorizontalFlip()","RandomCrop(32, padding=4)","ToTensor()","Cutout(n_holes=1, length=16)","RandomApply([ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)","RandomGrayscale(p=0.2)","RandomApply([GaussianBlur(kernel_size=3, sigma=[.1, 2.])], p=0.5)"]' \
load_ckpt=True load_ckpt_filepath="checkpoints/${WANDB_NAME}__epoch_159.pt" \
enable_oversampling=True enable_open=True