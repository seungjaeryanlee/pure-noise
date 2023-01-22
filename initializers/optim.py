import torch
import torch.optim as optim


def initialize_lr_scheduler(
    optimizer,
    enable_linear_warmup: bool,
    lr_decay: float,
    lr_decay_epochs: list,
):
    multistep_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=lr_decay_epochs,
        gamma=lr_decay,
    )

    if not enable_linear_warmup:
        return multistep_scheduler

    # Add learning rate warm-up
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        # NOTE(ryanlee): 0 causes ZeroDivisionError
        start_factor=torch.finfo().tiny,
        end_factor=1,
        total_iters=5,
    )
    scheduler = optim.lr_scheduler.ChainedScheduler([
        warmup_scheduler,
        multistep_scheduler,
    ])

    return scheduler

"""
Modified from https://github.com/kaidic/LDAM-DRW/blob/master/cifar_train.py
"""
def compute_learning_rate( 
    epoch, # starts at 0
    default_lr,
    lr_decay, 
    lr_decay_epochs, 
    enable_linear_warmup):
    if enable_linear_warmup and (epoch < 5):
        return default_lr * (epoch + 1) / 5
    elif epoch >= lr_decay_epochs[1]:
        return default_lr * (lr_decay ** 2)
    elif epoch >= lr_decay_epochs[0]:
        return default_lr * (lr_decay)
    else:
        return default_lr


def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    

if __name__ == '__main__':
    default_lr = 0.1 
    lr_decay = 0.01
    lr_decay_epochs = [160, 180]
    assert compute_learning_rate(0, default_lr, lr_decay, lr_decay_epochs, False) == 0.1
    assert compute_learning_rate(0, default_lr, lr_decay, lr_decay_epochs, True) == 0.02
    assert compute_learning_rate(3, default_lr, lr_decay, lr_decay_epochs, True) == 0.08
    assert compute_learning_rate(4, default_lr, lr_decay, lr_decay_epochs, True) == 0.1
    assert compute_learning_rate(159, default_lr, lr_decay, lr_decay_epochs, True) == 0.1
    assert compute_learning_rate(160, default_lr, lr_decay, lr_decay_epochs, True) == 0.001
    assert compute_learning_rate(179, default_lr, lr_decay, lr_decay_epochs, True) == 0.001
    assert compute_learning_rate(180, default_lr, lr_decay, lr_decay_epochs, True) == 0.00001
    assert compute_learning_rate(199, default_lr, lr_decay, lr_decay_epochs, True) == 0.00001
    print('compute_learning_rate done.')