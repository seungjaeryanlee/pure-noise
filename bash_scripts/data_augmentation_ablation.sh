### Reproduce Figure 3 (data augmentation ablation study) in the paper.
# Currently, this is same as main.sh, because we only apply horizontal flip and random crop by default.
# Model: WideResNet 
# Dataset: CIFAR-10-LT

### HorizontalFlip + RandomCrop 
# ERM
python train.py \
'train_transform_reprs=["ConvertImageDtype(float)","RandomHorizontalFlip()","RandomCrop(32, padding=4)","Normalize((0.4988, 0.5040, 0.4926), (0.2498, 0.2480, 0.2718))"]' \ 
'valid_transform_reprs=["ConvertImageDtype(float)","Normalize((0.4988, 0.5040, 0.4926), (0.2498, 0.2480, 0.2718))"]' \
'pure_noise_mean=[0.4988, 0.5040, 0.4926]' 'pure_noise_std=[0.2498, 0.2480, 0.2718]'

# Oversampling
python train.py \
'train_transform_reprs=["ConvertImageDtype(float)","RandomHorizontalFlip()","RandomCrop(32, padding=4)","Normalize((0.4988, 0.5040, 0.4926), (0.2498, 0.2480, 0.2718))"]' \ 
'valid_transform_reprs=["ConvertImageDtype(float)","Normalize((0.4988, 0.5040, 0.4926), (0.2498, 0.2480, 0.2718))"]' \
'pure_noise_mean=[0.4988, 0.5040, 0.4926]' 'pure_noise_std=[0.2498, 0.2480, 0.2718]' \
enable_oversampling=True

# OPeN
python train.py \
'train_transform_reprs=["ConvertImageDtype(float)","RandomHorizontalFlip()","RandomCrop(32, padding=4)","Normalize((0.4988, 0.5040, 0.4926), (0.2498, 0.2480, 0.2718))"]' \ 
'valid_transform_reprs=["ConvertImageDtype(float)","Normalize((0.4988, 0.5040, 0.4926), (0.2498, 0.2480, 0.2718))"]' \
'pure_noise_mean=[0.4988, 0.5040, 0.4926]' 'pure_noise_std=[0.2498, 0.2480, 0.2718]'
enable_oversampling=True enable_open=True
