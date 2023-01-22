### Reproduce Figure 3 (data augmentation ablation study) in the paper.
# Model: WideResNet 
# Dataset: CIFAR-10-LT

### HorizontalFlip + RandomCrop 
# # ERM
# python train.py save_ckpt=True wandb_name='flipcrop-0121' \
# 'train_transform_reprs=["ConvertImageDtype(float)","RandomHorizontalFlip()","RandomCrop(32, padding=4)","Normalize((0.4988, 0.5040, 0.4926), (0.2498, 0.2480, 0.2718))"]' \ 
# 'valid_transform_reprs=["ConvertImageDtype(float)","Normalize((0.4988, 0.5040, 0.4926), (0.2498, 0.2480, 0.2718))"]' \
# 'pure_noise_mean=[0.4988, 0.5040, 0.4926]' 'pure_noise_std=[0.2498, 0.2480, 0.2718]'

# # Oversampling
# python train.py wandb_name='flipcrop-rs-0121' \
# load_ckpt=True load_ckpt_filepath='checkpoints/flipcrop-0121__epoch_159.pt' \
# 'train_transform_reprs=["ConvertImageDtype(float)","RandomHorizontalFlip()","RandomCrop(32, padding=4)","Normalize((0.4988, 0.5040, 0.4926), (0.2498, 0.2480, 0.2718))"]' \ 
# 'valid_transform_reprs=["ConvertImageDtype(float)","Normalize((0.4988, 0.5040, 0.4926), (0.2498, 0.2480, 0.2718))"]' \
# 'pure_noise_mean=[0.4988, 0.5040, 0.4926]' 'pure_noise_std=[0.2498, 0.2480, 0.2718]' \
# enable_oversampling=True

# # OPeN
# python train.py wandb_name='flipcrop-open-0121' \
# load_ckpt=True load_ckpt_filepath='checkpoints/flipcrop-0121__epoch_159.pt' \
# 'train_transform_reprs=["ConvertImageDtype(float)","RandomHorizontalFlip()","RandomCrop(32, padding=4)","Normalize((0.4988, 0.5040, 0.4926), (0.2498, 0.2480, 0.2718))"]' \ 
# 'valid_transform_reprs=["ConvertImageDtype(float)","Normalize((0.4988, 0.5040, 0.4926), (0.2498, 0.2480, 0.2718))"]' \
# 'pure_noise_mean=[0.4988, 0.5040, 0.4926]' 'pure_noise_std=[0.2498, 0.2480, 0.2718]'
# enable_oversampling=True enable_open=True

### Add CutOut
# ERM
python train.py save_ckpt=True wandb_name='cutout-0121' \
'train_transform_reprs=["ConvertImageDtype(float)","RandomHorizontalFlip()","RandomCrop(32, padding=4)","Cutout(n_holes=1, length=16)","Normalize((0.4988, 0.5040, 0.4926), (0.2498, 0.2480, 0.2718))"]' \ 
'valid_transform_reprs=["ConvertImageDtype(float)","Normalize((0.4988, 0.5040, 0.4926), (0.2498, 0.2480, 0.2718))"]' \
'pure_noise_mean=[0.4988, 0.5040, 0.4926]' 'pure_noise_std=[0.2498, 0.2480, 0.2718]'

# Oversampling
python train.py wandb_name='cutout-rs-0121' \
load_ckpt=True load_ckpt_filepath='checkpoints/cutout-0121__epoch_159.pt' \
'train_transform_reprs=["ConvertImageDtype(float)","RandomHorizontalFlip()","RandomCrop(32, padding=4)","Cutout(n_holes=1, length=16)","Normalize((0.4988, 0.5040, 0.4926), (0.2498, 0.2480, 0.2718))"]' \ 
'valid_transform_reprs=["ConvertImageDtype(float)","Normalize((0.4988, 0.5040, 0.4926), (0.2498, 0.2480, 0.2718))"]' \
'pure_noise_mean=[0.4988, 0.5040, 0.4926]' 'pure_noise_std=[0.2498, 0.2480, 0.2718]' \
enable_oversampling=True

# OPeN
python train.py wandb_name='cutout-open-0121' \
load_ckpt=True load_ckpt_filepath='checkpoints/cutout-0121__epoch_159.pt' \
'train_transform_reprs=["ConvertImageDtype(float)","RandomHorizontalFlip()","RandomCrop(32, padding=4)","Cutout(n_holes=1, length=16)","Normalize((0.4988, 0.5040, 0.4926), (0.2498, 0.2480, 0.2718))"]' \ 
'valid_transform_reprs=["ConvertImageDtype(float)","Normalize((0.4988, 0.5040, 0.4926), (0.2498, 0.2480, 0.2718))"]' \
'pure_noise_mean=[0.4988, 0.5040, 0.4926]' 'pure_noise_std=[0.2498, 0.2480, 0.2718]'
enable_oversampling=True enable_open=True

### Add CutOut + SimCLR
# ERM
python train.py save_ckpt=True wandb_name='simclr-0121' \
'train_transform_reprs=["ConvertImageDtype(float)","RandomHorizontalFlip()","RandomCrop(32, padding=4)","Cutout(n_holes=1, length=16)","RandomApply([ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)","RandomGrayscale(p=0.2)","RandomApply([GaussianBlur(kernel_size=3, sigma=[.1, 2.])], p=0.5)","Normalize((0.4988, 0.5040, 0.4926), (0.2498, 0.2480, 0.2718))"]' \ 
'valid_transform_reprs=["ConvertImageDtype(float)","Normalize((0.4988, 0.5040, 0.4926), (0.2498, 0.2480, 0.2718))"]' \
'pure_noise_mean=[0.4988, 0.5040, 0.4926]' 'pure_noise_std=[0.2498, 0.2480, 0.2718]'

# Oversampling
python train.py wandb_name='simclr-rs-0121' \
load_ckpt=True load_ckpt_filepath='checkpoints/simclr-0121__epoch_159.pt' \
'train_transform_reprs=["ConvertImageDtype(float)","RandomHorizontalFlip()","RandomCrop(32, padding=4)","Cutout(n_holes=1, length=16)","RandomApply([ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)","RandomGrayscale(p=0.2)","RandomApply([GaussianBlur(kernel_size=3, sigma=[.1, 2.])], p=0.5)","Normalize((0.4988, 0.5040, 0.4926), (0.2498, 0.2480, 0.2718))"]' \ 
'valid_transform_reprs=["ConvertImageDtype(float)","Normalize((0.4988, 0.5040, 0.4926), (0.2498, 0.2480, 0.2718))"]' \
'pure_noise_mean=[0.4988, 0.5040, 0.4926]' 'pure_noise_std=[0.2498, 0.2480, 0.2718]' \
enable_oversampling=True

# OPeN
python train.py wandb_name='simclr-open-0121' \
load_ckpt=True load_ckpt_filepath='checkpoints/simclr-0121__epoch_159.pt' \
'train_transform_reprs=["ConvertImageDtype(float)","RandomHorizontalFlip()","RandomCrop(32, padding=4)","Cutout(n_holes=1, length=16)","RandomApply([ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)","RandomGrayscale(p=0.2)","RandomApply([GaussianBlur(kernel_size=3, sigma=[.1, 2.])], p=0.5)","Normalize((0.4988, 0.5040, 0.4926), (0.2498, 0.2480, 0.2718))"]' \ 
'valid_transform_reprs=["ConvertImageDtype(float)","Normalize((0.4988, 0.5040, 0.4926), (0.2498, 0.2480, 0.2718))"]' \
'pure_noise_mean=[0.4988, 0.5040, 0.4926]' 'pure_noise_std=[0.2498, 0.2480, 0.2718]'
enable_oversampling=True enable_open=True