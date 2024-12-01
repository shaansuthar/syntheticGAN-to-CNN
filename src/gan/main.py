import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from datasets import NoisyCIFAR10
from models import Generator, Discriminator
from utils import weights_init, imshow
from train import train_gan

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
    beta1 = 0.5  # Beta1 hyperparam for Adam optimizers
    noise_level = 0.1  # Standard deviation of Gaussian noise

    # Transformation: Convert images to tensors
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Load CIFAR-10 dataset
    train_dataset = torchvision.datasets.CIFAR10(
        root='../../data', train=True, download=True, transform=transform)

    # Create noisy dataset
    noisy_train_dataset = NoisyCIFAR10(train_dataset, noise_level=noise_level)

    # Data loader
    train_loader = DataLoader(noisy_train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=2)

    # Initialize networks
    netG = Generator().to(device)
    netD = Discriminator().to(device)

    netG.apply(weights_init)
    netD.apply(weights_init)

    # Train the GAN
    G_losses, D_losses = train_gan(netG, netD, train_loader, device,
                                   num_epochs, learning_rate, beta1, noise_level)

    # Visualization
    # Get some random samples from the test set
    test_dataset = torchvision.datasets.CIFAR10(
        root='../../data', train=False, download=True, transform=transform)
    noisy_test_dataset = NoisyCIFAR10(test_dataset, noise_level=noise_level)
    test_loader = DataLoader(noisy_test_dataset, batch_size=8, shuffle=True, num_workers=2)

    # Get a batch of test images
    dataiter = iter(test_loader)
    noisy_images, clean_images = next(dataiter)

    # Generate denoised images using the trained generator
    with torch.no_grad():
        netG.eval()
        fake_images = netG(noisy_images.to(device)).cpu()

    # Display noisy images
    imshow(torchvision.utils.make_grid(noisy_images, nrow=4),
           title='Noisy Images')

    # Display denoised images
    imshow(torchvision.utils.make_grid(fake_images, nrow=4),
           title='Denoised Images (Generated)')

    # Display original clean images
    imshow(torchvision.utils.make_grid(clean_images, nrow=4),
           title='Original Clean Images')

    # Plot the losses
    plt.figure()
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
