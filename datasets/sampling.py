import numpy as np

def count_labels(labels, num_classes):
    counts = np.arange(num_classes)
    for label in labels:
        counts[label] += 1
    return counts

def compute_ldam_class_weights(sample_labels_count, beta=0.9999):
    effective_num = 1.0 - np.power(beta, sample_labels_count)
    return (1.0 - beta) / np.array(effective_num)

def compute_weights_info(dataset, num_classes, use_ldam_weights):
    sample_labels = [label for _, label in dataset]
    sample_labels_count = count_labels(sample_labels, num_classes)
    if use_ldam_weights:
        label_weights = compute_ldam_class_weights(sample_labels_count)
    else:
        label_weights = 1. / sample_labels_count
    sample_weights = np.array([label_weights[l] for l in sample_labels])
    return {
        'sample_labels_count': sample_labels_count,
        'label_weights': label_weights,
        'sample_weights': sample_weights
    }

if __name__ == '__main__':
    from cifar10 import CIFAR10LTDataset
    from torchvision import transforms

    cifar10lt_dataset = CIFAR10LTDataset(
        json_filepath = "data/json/cifar10_imbalance100/cifar10_imbalance100_train.json",
        images_dirpath = "data/json/cifar10_imbalance100/images/",
        transform=transforms.ConvertImageDtype(float),
    )
    cifar10_weights = compute_weights_info(cifar10lt_dataset, 10, False)
    print(f"CIFAR-10-LT label weights: {cifar10_weights['label_weights']}")
    cifar10_ldam_weights = compute_weights_info(cifar10lt_dataset, 10, True)
    print(f"CIFAR-10-LT label LDAM-DRS weights: {cifar10_ldam_weights['label_weights']}")