import torch


def save_checkpoint(
    model,
    optimizer,
    checkpoint_filepath: str,
):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_filepath)


def load_checkpoint(
    model,
    optimizer,
    checkpoint_filepath: str,
):
    checkpoint = torch.load(checkpoint_filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
