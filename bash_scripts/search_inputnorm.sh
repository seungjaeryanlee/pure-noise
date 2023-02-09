
WANDB_NAME='search_inputnorm'

# ERM
python train.py wandb_name="${WANDB_NAME}" \

python train.py wandb_name="${WANDB_NAME}-ldam" \
enable_oversampling=True enable_open=True \
"normalize_mean=[0.4914, 0.4822, 0.4465]" \
"normalize_std=[0.2023, 0.1994, 0.2010]"

python train.py wandb_name="${WANDB_NAME}-cifar10" \
enable_oversampling=True enable_open=True \
"normalize_mean=[0.4914, 0.4822, 0.4465]" \
"normalize_std=[0.2470, 0.2435, 0.2616]"

python train.py wandb_name="${WANDB_NAME}-cifar10ltir100" \
enable_oversampling=True enable_open=True \
"normalize_mean=[0.4989, 0.5044, 0.4926]" \
"normalize_std=[0.2513, 0.2485, 0.2734]"




