'''
A nn.Sequential that can take additional noise_mask as input and returns only the output after Sequential iteration.

Modified from https://github.com/pytorch/pytorch/issues/19808#issuecomment-487291323
'''
import torch.nn as nn

class DarBnSequential(nn.Sequential):
    def forward(self, x, noise_mask=None):
        for block in self._modules.values():
            # assert (noise_mask is not None) == self.enable_dar_bn
            x, noise_mask = block(x, noise_mask)
        return x