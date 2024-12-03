import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

from datasets import NoisyCIFAR10
from models import Generator, Discriminator
from utils import weights_init, imshow

import torch.nn as nn
import torch.optim as optim

# Hyperparameters
num_epochs = 5
batch_size = 64
learning_rate = 0.0002
beta1 = 0.5
noise_level = 0.1
num_classes = 10

# Device configuration
device = torch.device("mps" if torch.mps.is_available() else "cpu")
print(f"Using device: {device}")

def preprocessing():
    # Set random seed for reproducibility
    manualSeed = 999
    torch.manual_seed(manualSeed)

    # Transformation: Convert images to tensors and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load CIFAR-10 dataset
    train_dataset = torchvision.datasets.CIFAR10(
        root='../data', train=True, download=True, transform=transform)

    test_dataset = torchvision.datasets.CIFAR10(
        root='../data', train=False, download=True, transform=transform)
    noisy_test_dataset = NoisyCIFAR10(test_dataset, noise_level=noise_level)
    test_loader = DataLoader(noisy_test_dataset, batch_size=8, shuffle=True, num_workers=0)    

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

    return netG, netD, train_loader, test_loader

def train_gan(netD, netG, train_loader):   

    # Loss function
    criterion = nn.BCELoss()

    # Optimizers
    optimizerD = optim.Adam(netD.parameters(), lr=learning_rate, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=learning_rate, betas=(beta1, 0.999))

    # Labels
    real_label = 1.
    fake_label = 0.

    G_losses = []
    D_losses = []

    print("Starting Training Loop...")
    for epoch in range(num_epochs):
        for i, data in enumerate(train_loader, 0):
            ############################
            # (1) Update D network
            ###########################
            netD.zero_grad()
            # Format batch
            noisy_images, labels, real_images = data
            noisy_images = noisy_images.to(device)
            real_images = real_images.to(device)
            labels = labels.to(device)
            b_size = real_images.size(0)
            label_real = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            label_fake = torch.full((b_size,), fake_label, dtype=torch.float, device=device)

            # Forward pass real batch through D
            output_real = netD(real_images, labels).view(-1)
            lossD_real = criterion(output_real, label_real)
            lossD_real.backward()

            # Generate fake images
            fake_images = netG(noisy_images, labels)

            # Classify fake images with D
            output_fake = netD(fake_images.detach(), labels).view(-1)
            lossD_fake = criterion(output_fake, label_fake)
            lossD_fake.backward()
            optimizerD.step()

            ############################
            # (2) Update G network
            ###########################
            netG.zero_grad()
            # Generate fake images again for G update
            fake_images = netG(noisy_images, labels)
            output_fake_for_G = netD(fake_images, labels).view(-1)
            lossG_adv = criterion(output_fake_for_G, label_real)
            # L1 loss for reconstruction
            lossG_L1 = nn.L1Loss()(fake_images, real_images) * 100  # Weight of L1 loss
            # Total generator loss
            lossG = lossG_adv + lossG_L1
            lossG.backward()
            optimizerG.step()

            # Save losses for plotting later
            G_losses.append(lossG.item())
            D_losses.append((lossD_real + lossD_fake).item())

            # Output training stats
            if i % 100 == 0:
                print(f'[{epoch}/{num_epochs}][{i}/{len(train_loader)}] '
                      f'Loss_D: {(lossD_real + lossD_fake):.4f} '
                      f'Loss_G: {lossG:.4f}')
                
    # save the generator model
    torch.save(netG.state_dict(), "../models/generator.pt")

    return G_losses, D_losses

def evaluate(G_losses, D_losses, netG, test_loader):
    # Get a batch of test images
    dataiter = iter(test_loader)
    noisy_images, labels, clean_images = next(dataiter)

    # Generate denoised images using the trained generator
    with torch.no_grad():
        netG.eval()
        fake_images = netG(noisy_images.to(device), labels.to(device)).cpu()

    # Display images all in one grid and save to a pdf 
    fig, axes = plt.subplots(nrows=3, ncols=8, figsize=(20, 8))
    for i in range(8):
        # Noisy images
        axes[0, i].imshow(noisy_images[i].permute(1, 2, 0) * 0.5 + 0.5)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Noisy Images')

        # Clean images
        axes[1, i].imshow(clean_images[i].permute(1, 2, 0) * 0.5 + 0.5)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Clean Images')

        # Fake images
        axes[2, i].imshow(fake_images[i].permute(1, 2, 0) * 0.5 + 0.5)
        axes[2, i].axis('off')
        if i == 0:
            axes[2, i].set_title('Fake Images')

    plt.tight_layout()
    plt.savefig("../results/comparison_grid.pdf")

    # Plot the losses
    plt.figure()
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("../results/gan_training_losses.pdf")    

if __name__ == "__main__":
    netG, netD, train_loader, test_loader = preprocessing()
    G_losses, D_losses = train_gan(netD, netG, train_loader)
    evaluate(G_losses, D_losses, netG, test_loader)