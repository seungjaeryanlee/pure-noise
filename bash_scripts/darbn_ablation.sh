# Ablation: replace DAR-BN with Standard BN and Auxiliary BN.

WANDB_NAME='BnAblation'

python train.py enable_oversampling=True enable_open=True \
wandb_name="${WANDB_NAME}-auxbn" noise_bn_option='AUXBN'

python train.py enable_oversampling=True enable_open=True \
wandb_name="${WANDB_NAME}-standard" noise_bn_option='STANDARD'

python train.py enable_oversampling=True enable_open=True \
wandb_name="${WANDB_NAME}-darbn" noise_bn_option='DARBN'