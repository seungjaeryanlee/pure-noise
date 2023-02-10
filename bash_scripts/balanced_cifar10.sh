### Compare Baseline Augmentation, AutoAugment, and pure-noise augmentation on balanced CIFAR10.
# Section 5, "Pure Noise Images - a General Useful Augmentation"

# Baseline Augmentation
python train.py config_filepath="default_cifar10.yaml" wandb_name='balanced-baseaug'

# Baseline + pure-noise
python train.py config_filepath="default_cifar10.yaml" wandb_name='balanced-autoaug-noise' \
enable_open=True

python train.py config_filepath="default_cifar10.yaml" wandb_name='balanced-autoaug-noise' \
enable_open=True open_start_epoch=0

# AutoAugment
python train.py config_filepath="default_cifar10.yaml" wandb_name='balanced-autoaug' \
'train_transform_reprs=["AutoAugment(policy=AutoAugmentPolicy.CIFAR10)","ToTensor()"]'

# AutoAugment + Pure-noise
python train.py config_filepath="default_cifar10.yaml" wandb_name='balanced-autoaug-noise' \
'train_transform_reprs=["AutoAugment(policy=AutoAugmentPolicy.CIFAR10)","ToTensor()"]' \
enable_open=True

python train.py config_filepath="default_cifar10.yaml" wandb_name='balanced-autoaug-noise-start0' \
'train_transform_reprs=["AutoAugment(policy=AutoAugmentPolicy.CIFAR10)","ToTensor()"]' \
enable_open=True open_start_epoch=0
