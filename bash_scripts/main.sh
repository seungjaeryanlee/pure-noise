### Run ERM, Oversampling, and OPeN. 
# Model: WideResNet
# Dataset: CIFAR-10-LT
python train.py || true
python train.py enable_oversampling=True || true
python train.py enable_oversampling=True enable_open=True || true
