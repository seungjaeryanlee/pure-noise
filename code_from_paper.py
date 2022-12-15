import torch


def sample_noise_images(image_size, mean, std, count):
    """Samples pure noise images from the normal distribution N(mean,std)"""
    r = torch.normal(mean[0], std[0], size=(count, 1, image_size, image_size))
    g = torch.normal(mean[1], std[1], size=(count, 1, image_size, image_size))
    b = torch.normal(mean[2], std[2], size=(count, 1, image_size, image_size))
    pure_noise_images = torch.cat((r, g, b), 1)

    return pure_noise_images


def oversampling_with_pure_noise_train_epoch(
    model, balanced_loader, criterion, optimizer, delta, num_samples_per_class,
    dataset_mean, dataset_std, image_size,
):
    """Trains model for one epoch according to the OPeN scheme

        model : torch.nn.Module;
            Model to train
        balanced_loader: torch.utils.data.DataLoader
            A class balanced loader - samples each class with equal probability
        delta: float
            Hyper-parameter for OPeN (see description in paper)
        num_samples_per_class: torch.IntTensor
            Number of samples in each class in the original imbalanced dataset
        dataset_mean: torch.FloatTensor of size: (3)
            Dataset mean per color channel
        dataset_std: torch.FloatTensor of size: (3)
            Dataset standard deviation per color channel
        image_size: int
            Image size - Assumes squared images of size (image_size , image_size)
    """
    for images, targets in balanced_loader:
        # Compute representation ratio
        max_class_size = torch.max(num_samples_per_class)
        representation_ratio = num_samples_per_class[targets] / max_class_size
        # Compute probabilities to replace natural images with pure noise images
        noise_probs = (1 - representation_ratio) * delta
        # Sample indexes to replace with noise according to Bernoulli distribution
        noise_indices = torch.nonzero(torch.bernoulli(noise_probs)).view(-1)
        # Replace natural images with sampled pure noise images
        noise_images = sample_noise_images(image_size=image_size,
        mean=dataset_mean, std=dataset_std, count=len(noise_indices))
        images[noise_indices] = noise_images
        # Create mask for noise images - later used by DAR-BN
        noise_mask = torch.zeros(images.size(0), dtype=torch.bool)
        noise_mask[noise_indices] = True
        # Train model
        outputs = model(images, noise_mask)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


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
