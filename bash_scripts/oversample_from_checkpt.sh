# OPeN
python train.py \
load_ckpt=True load_ckpt_filepath='checkpoints/bumbling-vortex-221__epoch_160.pt' load_ckpt_epoch=160 \
enable_oversampling=True enable_open=True

# Oversampling
python train.py \
load_ckpt=True load_ckpt_filepath='checkpoints/bumbling-vortex-221__epoch_160.pt' load_ckpt_epoch=160 \
enable_oversampling=True

# # Oversampling
# python train.py oversample_majority_class_num_samples=False \
# load_ckpt=True load_ckpt_filepath='checkpoints/bumbling-vortex-221__epoch_160.pt' load_ckpt_epoch=160 \
# enable_oversampling=True 
# # OPeN
# python train.py oversample_majority_class_num_samples=False \
# load_ckpt=True load_ckpt_filepath='checkpoints/bumbling-vortex-221__epoch_160.pt' load_ckpt_epoch=160 \
# enable_oversampling=True enable_open=True

# # Oversampling
# python train.py oversample_use_effective_num_sample_weights=True \
# load_ckpt=True load_ckpt_filepath='checkpoints/bumbling-vortex-221__epoch_160.pt' load_ckpt_epoch=160 \
# enable_oversampling=True 
# # OPeN
# python train.py oversample_use_effective_num_sample_weights=True \
# load_ckpt=True load_ckpt_filepath='checkpoints/bumbling-vortex-221__epoch_160.pt' load_ckpt_epoch=160 \
# enable_oversampling=True enable_open=True

# # Oversampling
# python train.py oversample_majority_class_num_samples=False oversample_use_effective_num_sample_weights=True \
# load_ckpt=True load_ckpt_filepath='checkpoints/bumbling-vortex-221__epoch_160.pt' load_ckpt_epoch=160 \
# enable_oversampling=True
# # OPeN
# python train.py oversample_majority_class_num_samples=False oversample_use_effective_num_sample_weights=True \
# load_ckpt=True load_ckpt_filepath='checkpoints/bumbling-vortex-221__epoch_160.pt' load_ckpt_epoch=160 \
# enable_oversampling=True enable_open=True
