import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def sample_noise_images(image_size, mean, std, count):
    """Samples pure noise images from the normal distribution N(mean,std)"""
    r = torch.normal(mean[0], std[0], size=(count, 1, image_size, image_size))
    g = torch.normal(mean[1], std[1], size=(count, 1, image_size, image_size))
    b = torch.normal(mean[2], std[2], size=(count, 1, image_size, image_size))
    pure_noise_images = torch.cat((r, g, b), 1)
    return torch.clamp(pure_noise_images, min=0.0, max=1.0)


def replace_with_fixed_ratio_pure_noise(
    images, targets, noise_ratio, dataset_mean, dataset_std, image_size):
    """Replaces images with pure noise and returns noise mask for use by DAR-BN in model.

        noise_ratio: float
            Fixed noise ratio.
        dataset_mean: torch.FloatTensor of size: (3)
            Dataset mean per color channel
        dataset_std: torch.FloatTensor of size: (3)
            Dataset standard deviation per color channel
        image_size: int
            Image size - Assumes squared images of size (image_size , image_size)
    """
    noise_probs = torch.full((images.size(0),), noise_ratio).to(device)
    # Sample indexes to replace with noise according to Bernoulli distribution
    noise_indices = torch.nonzero(torch.bernoulli(noise_probs)).view(-1)
    # Replace natural images with sampled pure noise images
    noise_images = sample_noise_images(
        image_size=image_size,
        mean=dataset_mean, 
        std=dataset_std, 
        count=len(noise_indices)
        ).to(device)
    images[noise_indices] = noise_images
    # Create mask for noise images - later used by DAR-BN
    noise_mask = torch.zeros(images.size(0), dtype=torch.bool).to(device)
    noise_mask[noise_indices] = True
    return noise_mask
