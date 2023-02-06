# Cui et al.
python train.py wandb_name='cui-erm' \
'train_transform_reprs=["RandomHorizontalFlip()","RandomCrop(32, padding=4)","ToTensor()","Cutout(n_holes=1, length=16)","RandomApply([ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)","RandomGrayscale(p=0.2)","RandomApply([GaussianBlur(kernel_size=3, sigma=[.1, 2.])], p=0.5)"]' \
train_subset_filepath=cui_indices.txt

python train.py wandb_name='cui-rs' \
'train_transform_reprs=["RandomHorizontalFlip()","RandomCrop(32, padding=4)","ToTensor()","Cutout(n_holes=1, length=16)","RandomApply([ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)","RandomGrayscale(p=0.2)","RandomApply([GaussianBlur(kernel_size=3, sigma=[.1, 2.])], p=0.5)"]' \
train_subset_filepath=cui_indices.txt \
load_ckpt=True load_ckpt_filepath=checkpoints/cui-erm__epoch_159.pt \
enable_oversampling=True

python train.py wandb_name='cui-open' \
'train_transform_reprs=["RandomHorizontalFlip()","RandomCrop(32, padding=4)","ToTensor()","Cutout(n_holes=1, length=16)","RandomApply([ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)","RandomGrayscale(p=0.2)","RandomApply([GaussianBlur(kernel_size=3, sigma=[.1, 2.])], p=0.5)"]' \
train_subset_filepath=cui_indices.txt \
load_ckpt=True load_ckpt_filepath=checkpoints/cui-erm__epoch_159.pt \
enable_oversampling=True enable_open=True

# M2m
python train.py wandb_name='m2m-erm' \
'train_transform_reprs=["RandomHorizontalFlip()","RandomCrop(32, padding=4)","ToTensor()","Cutout(n_holes=1, length=16)","RandomApply([ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)","RandomGrayscale(p=0.2)","RandomApply([GaussianBlur(kernel_size=3, sigma=[.1, 2.])], p=0.5)"]' \
train_subset_filepath=m2m_indices.txt

python train.py wandb_name='m2m-rs' \
'train_transform_reprs=["RandomHorizontalFlip()","RandomCrop(32, padding=4)","ToTensor()","Cutout(n_holes=1, length=16)","RandomApply([ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)","RandomGrayscale(p=0.2)","RandomApply([GaussianBlur(kernel_size=3, sigma=[.1, 2.])], p=0.5)"]' \
train_subset_filepath=m2m_indices.txt \
load_ckpt=True load_ckpt_filepath=checkpoints/m2m-erm__epoch_159.pt \
enable_oversampling=True

python train.py wandb_name='m2m-open' \
'train_transform_reprs=["RandomHorizontalFlip()","RandomCrop(32, padding=4)","ToTensor()","Cutout(n_holes=1, length=16)","RandomApply([ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)","RandomGrayscale(p=0.2)","RandomApply([GaussianBlur(kernel_size=3, sigma=[.1, 2.])], p=0.5)"]' \
train_subset_filepath=m2m_indices.txt \
load_ckpt=True load_ckpt_filepath=checkpoints/m2m-erm__epoch_159.pt \
enable_oversampling=True enable_open=True

# LDAM
python train.py wandb_name='ldam-erm' \
'train_transform_reprs=["RandomHorizontalFlip()","RandomCrop(32, padding=4)","ToTensor()","Cutout(n_holes=1, length=16)","RandomApply([ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)","RandomGrayscale(p=0.2)","RandomApply([GaussianBlur(kernel_size=3, sigma=[.1, 2.])], p=0.5)"]' \
train_subset_filepath=ldam_indices.txt

python train.py wandb_name='ldam-rs' \
'train_transform_reprs=["RandomHorizontalFlip()","RandomCrop(32, padding=4)","ToTensor()","Cutout(n_holes=1, length=16)","RandomApply([ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)","RandomGrayscale(p=0.2)","RandomApply([GaussianBlur(kernel_size=3, sigma=[.1, 2.])], p=0.5)"]' \
train_subset_filepath=ldam_indices.txt \
load_ckpt=True load_ckpt_filepath=checkpoints/ldam-erm__epoch_159.pt \
enable_oversampling=True

python train.py wandb_name='ldam-open' \
'train_transform_reprs=["RandomHorizontalFlip()","RandomCrop(32, padding=4)","ToTensor()","Cutout(n_holes=1, length=16)","RandomApply([ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)","RandomGrayscale(p=0.2)","RandomApply([GaussianBlur(kernel_size=3, sigma=[.1, 2.])], p=0.5)"]' \
train_subset_filepath=ldam_indices.txt \
load_ckpt=True load_ckpt_filepath=checkpoints/ldam-erm__epoch_159.pt \
enable_oversampling=True enable_open=True

