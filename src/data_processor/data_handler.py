import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.datasets import CIFAR10
import config


class CIFAR10Dataset:
    def __init__(self, root=config.DATA_DIR, transform=None):
        self.transform = transform or transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])
        self.dataset = CIFAR10(root=root, train=True, download=True, transform=self.transform)
        self.test_dataset = CIFAR10(root=root, train=False, download=True, transform=self.transform)

    def get_train_val_split(self, train_size=0.8):
        train_length = int(len(self.dataset) * train_size)
        val_length = len(self.dataset) - train_length
        return random_split(self.dataset, [train_length, val_length])

    def get_test_dataset(self):
        return self.test_dataset

    def get_dataloader(self, dataset, batch_size=config.BATCH_SIZE, shuffle=True):
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class NoisyCIFAR10(Dataset):
    def __init__(self, dataset, noise_level=config.NOISE_LEVEL):
        self.dataset = dataset
        self.noise_level = noise_level

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        clean_image, label = self.dataset[idx]
        # Add Gaussian noise
        noise = torch.randn_like(clean_image) * self.noise_level
        noisy_image = clean_image + noise
        # Clamp to [-1, 1]
        noisy_image = torch.clamp(noisy_image, -1., 1.)
        return noisy_image, label, clean_image
