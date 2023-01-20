# Oversampling
python train.py \
load_ckpt=True load_ckpt_filepath='checkpoints/bumbling-vortex-221__epoch_160.pt' load_ckpt_epoch=160 \
enable_oversampling=True

# OPeN
python train.py \
load_ckpt=True load_ckpt_filepath='checkpoints/bumbling-vortex-221__epoch_160.pt' load_ckpt_epoch=160 \
enable_oversampling=True enable_open=True
