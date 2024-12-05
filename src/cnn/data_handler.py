import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, ConcatDataset

import config


# strictly only cifar10 dataset
def load_only_cifar():

    all_transforms = transforms.Compose([transforms.Resize((32,32)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                                            std=[0.2023, 0.1994, 0.2010])
                                        ])
    # Create Training dataset
    train_dataset = torchvision.datasets.CIFAR10(root = '../../data',
                                                train = True,
                                                transform = all_transforms,
                                                download = True)

    # Create Testing dataset
    test_dataset = torchvision.datasets.CIFAR10(root = '../../data',
                                                train = False,
                                                transform = all_transforms,
                                                download=True)

    # Instantiate loader objects to facilitate processing
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                            batch_size = config.BATCH_SIZE,
                                            shuffle = True)


    test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                            batch_size = config.BATCH_SIZE,
                                            shuffle = True)
    
    return train_loader, test_loader

def load_only_synthetic():
    pass

# TODO: Need to figure out SyntheticData structure
def load_cifar10_synthetic():
    return True
