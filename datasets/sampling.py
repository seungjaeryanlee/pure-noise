import numpy as np

def count_class_frequency(labels, num_classes):
    counts = np.zeros(num_classes)
    for label in labels:
        counts[label] += 1
    return counts

def compute_class_weights_on_effective_num_samples(class_frequency, beta=0.9999):
    effective_num = 1.0 - np.power(beta, class_frequency)
    return (1.0 - beta) / np.array(effective_num)

def compute_sample_weights(labels, class_weights):
    return np.array([class_weights[label] for label in labels])
    
if __name__ == '__main__':
    from .celeba5 import CELEBA5_TRAIN_DATASET_PATH, CelebA5Dataset
    from .imbalanced_cifar import IMBALANCECIFAR10
    from torchvision.datasets import CIFAR10, CIFAR100
    from torchvision import transforms
    
    np.set_printoptions(precision=4)

    def compute_for_dataset(dataset, num_classes):
        class_frequency = count_class_frequency(dataset.targets, num_classes)
        print(f"Class frequency: {class_frequency}")
        
        class_weights = 1. / class_frequency
        sample_weights = compute_sample_weights(dataset.targets, class_weights)
        print(f"Class weights, inverse frequency: {class_weights}")
        print(f"Example sample weights, inverse frequency: {sample_weights[:10]}")
        
        class_weights = compute_class_weights_on_effective_num_samples(class_frequency)
        sample_weights = compute_sample_weights(dataset.targets, class_weights)
        print(f"Class weights, effective number of samples: {class_weights}")
        print(f"Example sample weights, effective number of samples: {sample_weights[:10]}")
        
    print("CelebA-5")
    celeba5_dataset = CelebA5Dataset(
        dataset_path=CELEBA5_TRAIN_DATASET_PATH,
        transform=transforms.ToTensor(),
    )
    compute_for_dataset(celeba5_dataset, 5)

    print("CIFAR10 Long-Tailed")
    cifar10lt_dataset = IMBALANCECIFAR10(
        root="data/",
        transform=transforms.ToTensor()
    )
    compute_for_dataset(cifar10lt_dataset, 10)
    
    print("Balanced CIFAR10")
    cifar10_dataset = CIFAR10(
        root="data/",
        transform=transforms.ToTensor(),
        train=True
    )
    compute_for_dataset(cifar10_dataset, 10)