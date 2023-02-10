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
    images, targets, noise_ratio, dataset_mean, dataset_std, image_size, num_classes):
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
    batch_size = images.size(0)
    noise_images_count = int(batch_size * noise_ratio)
    discrete_uniform_noise_target_probs = torch.full((num_classes,), 1. / num_classes)
    noise_targets = torch.distributions.categorical.Categorical(
        probs=discrete_uniform_noise_target_probs).sample(sample_shape=(noise_images_count,)).to(device)
    noise_images = sample_noise_images(
        image_size=image_size,
        mean=dataset_mean, 
        std=dataset_std, 
        count=noise_images_count
        ).to(device)
    extended_targets = torch.cat((targets, noise_targets), 0)
    extended_images = torch.cat((images, noise_images), 0)
    # Create mask for noise images - later used by DAR-BN
    noise_mask = torch.ones(batch_size + noise_images_count, dtype=torch.bool).to(device)
    noise_mask[:batch_size] = True
    return extended_images, extended_targets, noise_mask
