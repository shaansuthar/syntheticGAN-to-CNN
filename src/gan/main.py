# main.py

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

from datasets import NoisyCIFAR10
from models import Generator, Discriminator
from utils import weights_init, imshow
from train import train_gan

import torch.nn as nn
import torch.optim as optim

# Create an augmented dataset by combining original and synthetic data
class AugmentedDataset(Dataset):
    def __init__(self, original_dataset, synthetic_images, synthetic_labels, transform=None):
        self.original_dataset = original_dataset
        self.synthetic_images = synthetic_images
        self.synthetic_labels = synthetic_labels
        self.transform = transform

    def __len__(self):
        return len(self.original_dataset) + len(self.synthetic_labels)

    def __getitem__(self, idx):
        if idx < len(self.original_dataset):
            image, label = self.original_dataset[idx]
        else:
            synthetic_idx = idx - len(self.original_dataset)
            image = self.synthetic_images[synthetic_idx]
            label = torch.tensor(self.synthetic_labels[synthetic_idx], dtype=torch.long)
            if self.transform:
                image = self.transform(image)
        return image, label
    
# Define a simple CNN classifier
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def main():
    # Set random seed for reproducibility
    manualSeed = 999
    torch.manual_seed(manualSeed)
    
    # Device configuration
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters
    num_epochs = 5
    batch_size = 64
    learning_rate = 0.0002
    beta1 = 0.5
    noise_level = 0.1
    num_classes = 10

    # Transformation: Convert images to tensors and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load CIFAR-10 dataset
    train_dataset = torchvision.datasets.CIFAR10(
        root='../../data', train=True, download=True, transform=transform)

    # Create noisy dataset
    noisy_train_dataset = NoisyCIFAR10(train_dataset, noise_level=noise_level)

    # Data loader
    train_loader = DataLoader(noisy_train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=0)

    # Initialize networks
    netG = Generator(num_classes).to(device)
    netD = Discriminator(num_classes).to(device)

    netG.apply(weights_init)
    netD.apply(weights_init)

    # Train the GAN
    G_losses, D_losses = train_gan(netG, netD, train_loader, device,
                                   num_epochs, learning_rate, beta1, num_classes)

    # Visualization
    # Get some random samples from the test set
    test_dataset = torchvision.datasets.CIFAR10(
        root='../../data', train=False, download=True, transform=transform)
    noisy_test_dataset = NoisyCIFAR10(test_dataset, noise_level=noise_level)
    test_loader = DataLoader(noisy_test_dataset, batch_size=8, shuffle=True, num_workers=0)

    # Get a batch of test images
    dataiter = iter(test_loader)
    noisy_images, labels, clean_images = next(dataiter)

    # Generate denoised images using the trained generator
    with torch.no_grad():
        netG.eval()
        fake_images = netG(noisy_images.to(device), labels.to(device)).cpu()

    # Display images
    imshow(torchvision.utils.make_grid(noisy_images, nrow=4), title='Noisy Images')
    imshow(torchvision.utils.make_grid(fake_images, nrow=4), title='Denoised Images (Generated)')
    imshow(torchvision.utils.make_grid(clean_images, nrow=4), title='Original Clean Images')

    # Plot the losses
    plt.figure()
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # ---------------------------
    # Generate Synthetic Images
    # ---------------------------
    print("Generating synthetic images...")
    netG.eval()
    synthetic_images = []
    synthetic_labels = []
    with torch.no_grad():
        for class_label in range(num_classes):
            print(f"Generating images for class {class_label}")
            num_samples = 5000  # Number of images to generate per class
            for _ in range(num_samples // batch_size):
                # Generate noisy images
                noisy_batch = torch.randn(batch_size, 3, 32, 32).to(device)
                labels_batch = torch.full((batch_size,), class_label, dtype=torch.long).to(device)
                fake_images = netG(noisy_batch, labels_batch)
                synthetic_images.append(fake_images.cpu())
                synthetic_labels.extend([class_label] * batch_size)

    # Stack synthetic images and labels
    synthetic_images = torch.cat(synthetic_images)
    synthetic_labels = torch.tensor(synthetic_labels)

    # Combine datasets
    augmented_dataset = AugmentedDataset(train_dataset, synthetic_images, synthetic_labels, transform=None)
    augmented_loader = DataLoader(augmented_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Train the classifier with the augmented dataset
    def train_classifier(model, train_loader, test_loader, num_epochs=10):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)
            epoch_loss = running_loss / len(train_loader.dataset)
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

            # Evaluate on test set
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total
            print(f'Accuracy on test set: {accuracy:.2f}%')

    # Prepare test loader
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Create classifier model
    classifier = SimpleCNN(num_classes=num_classes).to(device)

    # Train classifier with augmented dataset
    print("Training classifier with augmented dataset...")
    train_classifier(classifier, augmented_loader, test_loader, num_epochs=10)

    # For comparison, train classifier with original dataset
    original_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    classifier_original = SimpleCNN(num_classes=num_classes).to(device)
    print("Training classifier with original dataset...")
    train_classifier(classifier_original, original_loader, test_loader, num_epochs=10)

if __name__ == '__main__':
    main()
