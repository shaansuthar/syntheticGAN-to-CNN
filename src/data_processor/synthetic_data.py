import torch
import config
from torch.utils.data import Dataset

class SyntheticDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels  # List of integers

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]  # Return label as int


def generate_synthetic_data(generator, num_samples):
    generator.eval()
    synthetic_images = []
    synthetic_labels = []
    with torch.no_grad():
        for class_label in range(config.NUM_CLASSES):
            num_samples_per_class = num_samples // config.NUM_CLASSES
            for _ in range(num_samples_per_class // config.BATCH_SIZE):
                # Generate noisy images
                noisy_batch = torch.randn(config.BATCH_SIZE, 3, 32, 32).to(config.DEVICE)
                labels_batch = torch.full((config.BATCH_SIZE,), class_label, dtype=torch.long).to(config.DEVICE)
                fake_images = generator(noisy_batch, labels_batch)
                synthetic_images.append(fake_images.cpu())
                synthetic_labels.extend([class_label] * config.BATCH_SIZE)

    # Stack synthetic images and keep labels as list of ints
    synthetic_images = torch.cat(synthetic_images)

    synthetic_dataset = SyntheticDataset(synthetic_images, synthetic_labels)
    return synthetic_dataset
