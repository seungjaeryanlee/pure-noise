import numpy as np

def count_class_frequency(labels, num_classes):
    counts = np.arange(num_classes)
    for label in labels:
        counts[label] += 1
    return counts

def compute_class_weights_on_effective_num_samples(class_frequency, beta=0.9999):
    effective_num = 1.0 - np.power(beta, class_frequency)
    return (1.0 - beta) / np.array(effective_num)

def compute_sample_weights(labels, class_weights):
    return np.array([class_weights[label] for label in labels])
    
if __name__ == '__main__':
    from cifar10 import CIFAR10LTDataset
    from torchvision import transforms

    dataset = CIFAR10LTDataset(
        json_filepath = "data/json/cifar10_imbalance100/cifar10_imbalance100_train.json",
        images_dirpath = "data/json/cifar10_imbalance100/images/",
        transform=transforms.ConvertImageDtype(float),
    )
    
    labels = [label for _, label in dataset]
    class_frequency = count_class_frequency(labels, dataset.NUM_CLASSES)
    
    class_weights = 1. / class_frequency
    sample_weights = compute_sample_weights(labels, class_weights)
    print(f"Class weights: {class_weights}")
    print(f"Sample weights: {sample_weights[:10]}")
    
    class_weights = compute_class_weights_on_effective_num_samples(class_frequency)
    sample_weights = compute_sample_weights(labels, class_weights)
    print(f"Class weights on effective num samples: {class_weights}")
    print(f"Sample weights on effective num samples: {sample_weights[:10]}")