import torch

def batch_norm_with_adaptive_parameters(x_noise, adaptive_parameters):
    """Applies batch normalization to x_noise according to adaptive_parameters

    x_noise : torch.FloatTensor of size: (N, C, H, W)
        2D activation maps obtained from noise images only
    adaptive_parameters:
        a dictionary containing:
            weight: scale parameter for the adaptive affine
            bias: bias parameter for the adaptive affine
            eps: a value added to the denominator for numerical stability.

    """
    # Calculate mean and variance for the noise activations batch per channel
    mean = x_noise.mean([0, 2, 3])
    var = x_noise.var([0, 2, 3], unbiased=False)
    # Normalize the noise activations batch per channel
    out = x_noise - mean[None, :, None, None]
    out = out / torch.sqrt(var[None, :, None, None] + adaptive_parameters["eps"])

    # Scale and shift using adaptive affine per channel
    scale = adaptive_parameters["weight"][None, :, None, None]
    shift = adaptive_parameters["bias"][None, :, None, None]
    out = out * scale + shift

    return out


def dar_bn(bn_layer, x, noise_mask):
    """Applies DAR-BN normalization to a 4D input (a mini-batch of 2D inputs with
    additional channel dimension)

    bn_layer : torch.nn.BatchNorm2d
        Batch norm layer operating on activation maps of natural images
    x : torch.FloatTensor of size: (N, C, H, W)
        2D activation maps obtained from both natural images and noise images
    noise_mask: torch.BoolTensor of size: (N)
        Boolean 1D tensor indicates which activation map is obtained from noise

    """
    # Batch norm for activation maps of natural images
    out_natural = bn_layer(x[torch.logical_not(noise_mask)])
    # Batch norm for activation maps of noise images
    # Do not compute gradients for this operation
    with torch.no_grad():
        adaptive_params = {
            "weight": bn_layer.weight,
            "bias": bn_layer.bias,
            "eps": bn_layer.eps,
        }
        out_noise = batch_norm_with_adaptive_parameters(
            x[noise_mask],
            adaptive_params,
        )

    # Concatenate activation maps in original order
    out = torch.empty_like(torch.cat([out_natural, out_noise], dim=0))
    out[torch.logical_not(noise_mask)] = out_natural
    out[noise_mask] = out_noise

    return out