### Compare Baseline Augmentation, AutoAugment, and pure-noise augmentation on balanced CIFAR10.
# Section 5, "Pure Noise Images - a General Useful Augmentation"

# Baseline Augmentation
python train.py config_filepath="default_cifar10.yaml" wandb_name='balanced-baseaug'

# AutoAugment
# TODO: does AutoAugment need normalize?
python train.py config_filepath="default_cifar10.yaml" wandb_name='balanced-autoaug' \
'train_transform_reprs=["AutoAugment(policy=AutoAugmentPolicy.CIFAR10)","ToTensor()"]'

# Pure-noise
python train.py config_filepath="default_cifar10.yaml" wandb_name='balanced-noise' \
'train_transform_reprs=["AutoAugment(policy=AutoAugmentPolicy.CIFAR10)","ToTensor()"]' \
enable_open=True