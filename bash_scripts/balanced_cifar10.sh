### Compare Baseline Augmentation, AutoAugment, and pure-noise augmentation on balanced CIFAR10.
# Section 5, "Pure Noise Images - a General Useful Augmentation"

# Baseline
python train.py config_filepath="default_cifar10.yaml" wandb_name='balanced-baseaug-0122'

# AutoAugment
python train.py config_filepath="default_cifar10.yaml" wandb_name='balanced-autoaug-0122' \
'train_transform_reprs=["AutoAugment(policy=AutoAugmentPolicy.CIFAR10)","ToTensor()"]'

# Pure-noise
# python train.py config_filepath="default_cifar10.yaml" wandb_name='balanced-noise-0122' \
# 'train_transform_reprs=["AutoAugment(policy=AutoAugmentPolicy.CIFAR10)"]'