
## ERM
```
python train.py \ 
--dataset CelebA-5 \
--train_transform '[RandomHorizontalFlip(), RandomCrop(32, padding=4), ToTensor()]' \
--use_oversampling False \
--model WideResNet-28-10 \
--num_epochs 90 \
```

## Oversampling
```
python train.py \ 
--dataset CelebA-5 \
--train_transform '[RandomHorizontalFlip(), RandomCrop(32, padding=4), ToTensor()]' \
--use_oversampling True \
--model WideResNet-28-10 \
--num_epochs 90 \
```
