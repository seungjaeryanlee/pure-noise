"""
InputNormalize from M2m (Kim et al., 2020)

From https://github.com/alinlab/M2m/tree/4cfd52d9298f9e77c77a18bb23d0f8d860c260f7
"""

import torch

class InputNormalize(torch.nn.Module):
    '''
    A module (custom layer) for normalizing the input to have a fixed
    mean and standard deviation (user-specified).
    '''
    def __init__(self, new_mean, new_std):
        super(InputNormalize, self).__init__()
        new_std = new_std[..., None, None].cuda()
        new_mean = new_mean[..., None, None].cuda()

        # To prevent the updates the mean, std
        self.register_buffer("new_mean", new_mean)
        self.register_buffer("new_std", new_std)

    def forward(self, x):
        x = torch.clamp(x, 0, 1)
        x_normalized = (x - self.new_mean)/self.new_std
        return x_normalized