# Sanity check: run each config for few batches then exit.
python train.py debug_run=True batch_size=32 config_filepath=default_celeba5.yaml model=ResNet-32-akamaster
python train.py debug_run=True batch_size=32 config_filepath=default_celeba5.yaml model=ResNet-32-akamaster enable_oversampling=True oversampling_start_epoch=0
python train.py debug_run=True batch_size=32 config_filepath=default_celeba5.yaml model=ResNet-32-akamaster enable_oversampling=True oversampling_start_epoch=0 enable_open=True

python train.py debug_run=True batch_size=32 config_filepath=default_cifar10lt.yaml model=ResNet-32-akamaster
python train.py debug_run=True batch_size=32 config_filepath=default_cifar10lt.yaml model=ResNet-32-akamaster enable_oversampling=True oversampling_start_epoch=0
python train.py debug_run=True batch_size=32 config_filepath=default_cifar10lt.yaml model=ResNet-32-akamaster enable_oversampling=True oversampling_start_epoch=0 enable_open=True

python train.py debug_run=True batch_size=32 config_filepath=default_celeba5.yaml model=WideResNet-28-10-torchdistill
python train.py debug_run=True batch_size=32 config_filepath=default_celeba5.yaml model=WideResNet-28-10-torchdistill enable_oversampling=True oversampling_start_epoch=0
python train.py debug_run=True batch_size=32 config_filepath=default_celeba5.yaml model=WideResNet-28-10-torchdistill enable_oversampling=True oversampling_start_epoch=0 enable_open=True

python train.py debug_run=True batch_size=32 config_filepath=default_cifar10lt.yaml model=WideResNet-28-10-torchdistill
python train.py debug_run=True batch_size=32 config_filepath=default_cifar10lt.yaml model=WideResNet-28-10-torchdistill enable_oversampling=True oversampling_start_epoch=0
python train.py debug_run=True batch_size=32 config_filepath=default_cifar10lt.yaml model=WideResNet-28-10-torchdistill enable_oversampling=True oversampling_start_epoch=0 enable_open=True

python train.py debug_run=True batch_size=32 config_filepath=default_celeba5.yaml model=ResNet-32-m2m
python train.py debug_run=True batch_size=32 config_filepath=default_celeba5.yaml model=ResNet-32-ldam
