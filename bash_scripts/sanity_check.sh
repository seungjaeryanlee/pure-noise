# Run this script to check if everything in the repo is set up correctly.

# Check all three datasets work
python train.py enable_wandb=False debug_run=True \
dataset="CIFAR-10-LT" config_filepath="default_cifar10lt.yaml"

python train.py enable_wandb=False debug_run=True \
dataset="CIFAR-10" config_filepath="default_cifar10.yaml"

# python train.py enable_wandb=False debug_run=True \
# dataset="CelebA-5" config_filepath="default_celeba5.yaml"


# Check both IR ratios work
python train.py enable_wandb=False debug_run=True \
ir_ratio=50

python train.py enable_wandb=False debug_run=True \
ir_ratio=100


# Check all data augmentations work
python train.py enable_wandb=False debug_run=True \
'train_transform_reprs=["RandomHorizontalFlip()","RandomCrop(32, padding=4)","ToTensor()","Cutout(n_holes=1, length=16)","RandomApply([ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)","RandomGrayscale(p=0.2)","RandomApply([GaussianBlur(kernel_size=3, sigma=[.1, 2.])], p=0.5)"]'

python train.py enable_wandb=False debug_run=True \
'train_transform_reprs=["AutoAugment(policy=AutoAugmentPolicy.CIFAR10)","ToTensor()"]'


# Check oversampling and OPeN works
python train.py enable_wandb=False debug_run=True \
enable_oversampling=True oversampling_start_epoch=0

python train.py enable_wandb=False debug_run=True \
enable_oversampling=True oversampling_start_epoch=0 \
enable_open=True open_start_epoch=0


# Check both models work
python train.py enable_wandb=False debug_run=True \
model='ResNet-32-akamaster'

python train.py enable_wandb=False debug_run=True \
model='WideResNet-28-10-torchdistill'


# Check all three batch norm layers work
python train.py enable_wandb=False debug_run=True \
noise_bn_option='STANDARD'

python train.py enable_wandb=False debug_run=True \
noise_bn_option='AUXBN'

python train.py enable_wandb=False debug_run=True \
noise_bn_option='DARBN'


# Check all three indices work
python train.py enable_wandb=False debug_run=True \
use_subset_to_train=True train_subset_filepath=cifar10ir100_indices_cui.txt

python train.py enable_wandb=False debug_run=True \
use_subset_to_train=True train_subset_filepath=cifar10ir100_indices_ldam.txt

python train.py enable_wandb=False debug_run=True \
use_subset_to_train=True train_subset_filepath=cifar10ir100_indices_m2m.txt
