import torch
import config
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np

class SyntheticDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels  # List of integers

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]  # Return label as int


def generate_synthetic_data(generator, num_samples, show_samples=False):
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

    if show_samples:
        visualize_batch(synthetic_images, synthetic_labels)
    return synthetic_dataset

def visualize_batch(images, labels, num_samples=10):
    
    # Select random indices if we have more than num_samples
    if len(images) > num_samples:
        indices = np.random.choice(len(images), num_samples, replace=False)
        images = images[indices]
        labels = [labels[i] for i in indices]
    
    # Create a grid of images
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                  'dog', 'frog', 'horse', 'ship', 'truck']
    
    for idx, (ax, img, label) in enumerate(zip(axes.flat, images, labels)):
        # Denormalize from [-1,1] to [0,1]
        img = (img + 1) / 2.0
        img = img.permute(1, 2, 0).numpy()
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f'{class_names[label]}')
    
    plt.tight_layout()
    plt.show()