from data_handler import load_only_cifar, load_only_synthetic, load_cifar10_synthetic

processing = {
    "cifar10": load_only_cifar(),
    "synthetic": load_only_synthetic(),
    "cifar10-synthetic": load_cifar10_synthetic()
}