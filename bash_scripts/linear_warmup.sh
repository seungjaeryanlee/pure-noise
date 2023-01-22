### Warmup off
# ERM
python train.py enable_linear_warmup=False save_ckpt=True wandb_name='no-warmup-0121'

# Oversampling
python train.py enable_linear_warmup=False wandb_name='no-warmup-rs-0121' \
load_ckpt=True load_ckpt_filepath='checkpoints/no-warmup-0121__epoch_159.pt' \
enable_oversampling=True

# OPeN
python train.py enable_linear_warmup=False wandb_name='no-warmup-open-0121' \
load_ckpt=True load_ckpt_filepath='checkpoints/no-warmup-0121__epoch_159.pt' \
enable_oversampling=True enable_open=True

### Warmup on
# ERM
python train.py enable_linear_warmup=True save_ckpt=True wandb_name='warmup-0121'

# Oversampling
python train.py enable_linear_warmup=True wandb_name='warmup-rs-0121' \
load_ckpt=True load_ckpt_filepath='checkpoints/warmup-0121__epoch_159.pt' \
enable_oversampling=True

# OPeN
python train.py enable_linear_warmup=True wandb_name='warmup-open-0121' \
load_ckpt=True load_ckpt_filepath='checkpoints/warmup-0121__epoch_159.pt' \
enable_oversampling=True enable_open=True