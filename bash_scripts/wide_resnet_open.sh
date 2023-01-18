python train.py enable_wandb=True config_filepath=default_cifar10lt.yaml model=WideResNet-28-10-torchdistill || true
python train.py enable_wandb=True config_filepath=default_cifar10lt.yaml model=WideResNet-28-10-torchdistill enable_overampling=True || true
python train.py enable_wandb=True config_filepath=default_cifar10lt.yaml model=WideResNet-28-10-torchdistill enable_overampling=True enable_open=True || true

python train.py enable_wandb=True config_filepath=default_celeba5.yaml model=WideResNet-28-10-torchdistill || true
python train.py enable_wandb=True config_filepath=default_celeba5.yaml model=WideResNet-28-10-torchdistill enable_overampling=True || true
python train.py enable_wandb=True config_filepath=default_celeba5.yaml model=WideResNet-28-10-torchdistill enable_overampling=True enable_open=True || true