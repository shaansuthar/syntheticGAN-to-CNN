import torch
import torch.nn as nn
import torch.optim as optim
from gan.gan import Generator, Discriminator
import config
import os
import matplotlib.pyplot as plt

def weights_init(m):
    if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.Linear)):
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)

class GANTrainer:
    def __init__(self, noise_level=config.NOISE_LEVEL, device=config.DEVICE,
                 learning_rate=config.GAN_LEARNING_RATE, beta1=config.GAN_BETA1, beta2=config.GAN_BETA2, num_epochs=config.GAN_NUM_EPOCHS):
        self.device = device
        self.num_epochs = num_epochs
        self.noise_level = noise_level
        self.netG = Generator().to(self.device)
        self.netD = Discriminator().to(self.device)
        self.netG.apply(weights_init)
        self.netD.apply(weights_init)
        self.criterion = nn.BCELoss()
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=learning_rate, betas=(beta1, beta2))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=learning_rate, betas=(beta1, beta2))
        self.real_label = 1.
        self.fake_label = 0.

    def train(self, train_loader):
        G_losses = []
        D_losses = []

        print("Starting GAN Training...")
        for epoch in range(self.num_epochs):
            for i, data in enumerate(train_loader, 0):
                ############################
                # (1) Update D network
                ###########################
                self.netD.zero_grad()
                noisy_images, labels, real_images = data
                noisy_images = noisy_images.to(self.device)
                real_images = real_images.to(self.device)
                labels = labels.to(self.device)
                b_size = real_images.size(0)
                label_real = torch.full((b_size,), self.real_label, dtype=torch.float, device=self.device)
                label_fake = torch.full((b_size,), self.fake_label, dtype=torch.float, device=self.device)

                # Forward pass real batch through D
                output_real = self.netD(real_images, labels).view(-1)
                lossD_real = self.criterion(output_real, label_real)
                lossD_real.backward()

                # Generate fake images
                fake_images = self.netG(noisy_images, labels)

                # Classify fake images with D
                output_fake = self.netD(fake_images.detach(), labels).view(-1)
                lossD_fake = self.criterion(output_fake, label_fake)
                lossD_fake.backward()
                self.optimizerD.step()

                ############################
                # (2) Update G network
                ###########################
                self.netG.zero_grad()
                # Generate fake images again for G update
                fake_images = self.netG(noisy_images, labels)
                output_fake_for_G = self.netD(fake_images, labels).view(-1)
                lossG_adv = self.criterion(output_fake_for_G, label_real)
                # L1 loss for reconstruction
                lossG_L1 = nn.L1Loss()(fake_images, real_images) * 100  # Weight of L1 loss
                # Total generator loss
                lossG = lossG_adv + lossG_L1
                lossG.backward()
                self.optimizerG.step()

                # Save losses for plotting later
                G_losses.append(lossG.item())
                D_losses.append((lossD_real + lossD_fake).item())

                # Output training stats
                if i % 100 == 0:
                    print(f'[{epoch}/{self.num_epochs}][{i}/{len(train_loader)}] '
                          f'Loss_D: {(lossD_real + lossD_fake):.4f} '
                          f'Loss_G: {lossG:.4f}')

        # Save the generator model
        if not os.path.exists(config.MODEL_DIR):
            os.makedirs(config.MODEL_DIR)
        torch.save(self.netG.state_dict(), os.path.join(config.MODEL_DIR, 'generator.pt'))

        return G_losses, D_losses

    def get_generator(self):
        return self.netG      
    
    def view_sample_images(self, test_loader):
        # Get a batch of test images
        dataiter = iter(test_loader)

        noisy_images, labels, clean_images = next(dataiter)

        # Generate denoised images using the trained generator
        with torch.no_grad():
            self.netG.eval()
            fake_images = self.netG(noisy_images.to(self.device), labels.to(self.device))

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
        plt.savefig(f'{config.RESULTS_DIR}/generated_images.pdf')     
    
    def plot_gan_losses(self, G_losses, D_losses):
        plt.figure()
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(G_losses, label="G")
        plt.plot(D_losses, label="D")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"{config.RESULTS_DIR}/gan_training_losses.pdf")        