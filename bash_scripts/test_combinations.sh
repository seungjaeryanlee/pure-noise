# Sanity check: run each config for few batches then exit.
# TODO: clean this up.
python train.py debug_run=True batch_size=32 config_filepath=default_celeba5.yaml model=ResNet-32-akamaster && \
python train.py debug_run=True batch_size=32 config_filepath=default_celeba5.yaml model=ResNet-32-akamaster enable_overampling=True && \
python train.py debug_run=True batch_size=32 config_filepath=default_celeba5.yaml model=ResNet-32-akamaster enable_overampling=True enable_open=True && \

python train.py debug_run=True batch_size=32 config_filepath=default_cifar10lt.yaml model=ResNet-32-akamaster && \
python train.py debug_run=True batch_size=32 config_filepath=default_cifar10lt.yaml model=ResNet-32-akamaster enable_overampling=True && \
python train.py debug_run=True batch_size=32 config_filepath=default_cifar10lt.yaml model=ResNet-32-akamaster enable_overampling=True enable_open=True && \

python train.py debug_run=True batch_size=32 config_filepath=default_celeba5.yaml model=WideResNet-28-10-torchdistill && \
python train.py debug_run=True batch_size=32 config_filepath=default_celeba5.yaml model=WideResNet-28-10-torchdistill enable_overampling=True && \
python train.py debug_run=True batch_size=32 config_filepath=default_celeba5.yaml model=WideResNet-28-10-torchdistill enable_overampling=True enable_open=True && \

python train.py debug_run=True batch_size=32 config_filepath=default_cifar10lt.yaml model=WideResNet-28-10-torchdistill && \
python train.py debug_run=True batch_size=32 config_filepath=default_cifar10lt.yaml model=WideResNet-28-10-torchdistill enable_overampling=True && \
python train.py debug_run=True batch_size=32 config_filepath=default_cifar10lt.yaml model=WideResNet-28-10-torchdistill enable_overampling=True enable_open=True

python train.py debug_run=True batch_size=32 config_filepath=default_celeba5.yaml model=ResNet-32-m2m && \
python train.py debug_run=True batch_size=32 config_filepath=default_celeba5.yaml model=ResNet-32-ldam
# TODO: fix?
# python train.py debug_run=True batch_size=32 config_filepath=default_celeba5.yaml model=WideResNet-28-10-xternalz
