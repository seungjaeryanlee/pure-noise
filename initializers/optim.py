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
