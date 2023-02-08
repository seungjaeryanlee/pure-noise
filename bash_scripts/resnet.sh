# Run DAR-BN ablation using ResNet. 

WANDB_NAME='ResNetBnAblation'

python train.py enable_oversampling=True enable_open=True model='ResNet-32-akamaster' \
wandb_name="${WANDB_NAME}-auxbn" noise_bn_option='AUXBN'

python train.py enable_oversampling=True enable_open=True model='ResNet-32-akamaster' \
wandb_name="${WANDB_NAME}-standard" noise_bn_option='STANDARD'

python train.py enable_oversampling=True enable_open=True model='ResNet-32-akamaster' \
wandb_name="${WANDB_NAME}-darbn" noise_bn_option='DARBN'

# Run ERM vs RS vs OPeN using ResNet.

WANDB_NAME='ResNet'

python train.py model='ResNet-32-akamaster' wandb_name="${WANDB_NAME}"

python train.py model='ResNet-32-akamaster' wandb_name="${WANDB_NAME}-rs" \
load_ckpt=True load_ckpt_filepath="checkpoints/${WANDB_NAME}__epoch_159.pt" \
enable_oversampling=True

# OPeN
python train.py model='ResNet-32-akamaster' wandb_name="${WANDB_NAME}-open" \
load_ckpt=True load_ckpt_filepath="checkpoints/${WANDB_NAME}__epoch_159.pt" \
enable_oversampling=True enable_open=True
