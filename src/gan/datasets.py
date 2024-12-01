import torch
from torchvision import datasets

class NoisyCIFAR10(torch.utils.data.Dataset):
    def __init__(self, dataset, noise_level=0.1):
        self.dataset = dataset
        self.noise_level = noise_level

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        clean_image, label = self.dataset[idx]
        # Add Gaussian noise
        noise = torch.randn_like(clean_image) * self.noise_level
        noisy_image = clean_image + noise
        # Clamp to [0, 1]
        noisy_image = torch.clamp(noisy_image, 0., 1.)
        return noisy_image, label, clean_image