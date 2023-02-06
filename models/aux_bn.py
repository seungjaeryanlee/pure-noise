import torch

def aux_bn(natural_bn_layer, noise_bn_layer, x, noise_mask):
    """Implementation of Auxiliary BN (Xie et al., 2020).
    
    natural_bn_layer : torch.nn.BatchNorm2d
        Batch norm layer operating on activation maps of natural images
    noise_bn_layer : torch.nn.BatchNorm2d
        Batch norm layer operating on activation maps of noise images
    x : torch.FloatTensor of size: (N, C, H, W)
        2D activation maps obtained from both natural images and noise images
    noise_mask: torch.BoolTensor of size: (N)
        Boolean 1D tensor indicates which activation map is obtained from noise

    """
    # Batch norm for activation maps of natural images
    out_natural = natural_bn_layer(x[torch.logical_not(noise_mask)])
    out_noise = noise_bn_layer(x[noise_mask])

    # Concatenate activation maps in original order
    out = torch.empty_like(torch.cat([out_natural, out_noise], dim=0))
    out[torch.logical_not(noise_mask)] = out_natural
    out[noise_mask] = out_noise

    return out