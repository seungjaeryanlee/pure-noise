python train.py enable_wandb=True config_filepath=default_cifar10lt.yaml model=ResNet-32-akamaster oversample_ldam_weights=True oversample_num_majority_class_samples=False oversampling_start_epoch=180 || true
python train.py enable_wandb=True config_filepath=default_cifar10lt.yaml model=ResNet-32-akamaster oversample_ldam_weights=True oversample_num_majority_class_samples=False oversampling_start_epoch=180 enable_overampling=True || true
python train.py enable_wandb=True config_filepath=default_cifar10lt.yaml model=ResNet-32-akamaster oversample_ldam_weights=True oversample_num_majority_class_samples=False oversampling_start_epoch=180 enable_overampling=True enable_open=True || true

python train.py enable_wandb=True config_filepath=default_celeba5.yaml model=ResNet-32-akamaster oversample_ldam_weights=True oversample_num_majority_class_samples=False oversampling_start_epoch=60 || true
python train.py enable_wandb=True config_filepath=default_celeba5.yaml model=ResNet-32-akamaster oversample_ldam_weights=True oversample_num_majority_class_samples=False oversampling_start_epoch=60 enable_overampling=True || true
python train.py enable_wandb=True config_filepath=default_celeba5.yaml model=ResNet-32-akamaster oversample_ldam_weights=True oversample_num_majority_class_samples=False oversampling_start_epoch=60 enable_overampling=True enable_open=True || true
