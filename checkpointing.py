import torch


def save_checkpoint(
    model,
    optimizer,
    checkpoint_filepath: str,
    finished_epoch,
):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'finished_epoch': finished_epoch,
    }, checkpoint_filepath)


def load_checkpoint(
    model,
    optimizer,
    checkpoint_filepath: str,
):
    checkpoint = torch.load(checkpoint_filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    
def load_finished_epoch_from_checkpoint(checkpoint_filepath: str):
    checkpoint = torch.load(checkpoint_filepath)
    return checkpoint['finished_epoch']
