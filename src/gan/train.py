import torch
import torch.optim as optim
import torch.nn as nn

def train_gan(netG, netD, dataloader, device, num_epochs, learning_rate, beta1, noise_level):
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
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network
            ###########################
            netD.zero_grad()
            # Format batch
            noisy_images, real_images = data
            noisy_images = noisy_images.to(device)
            real_images = real_images.to(device)
            b_size = real_images.size(0)
            label_real = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            label_fake = torch.full((b_size,), fake_label, dtype=torch.float, device=device)

            # Forward pass real batch through D
            output_real = netD(real_images).view(-1)
            lossD_real = criterion(output_real, label_real)
            lossD_real.backward()

            # Generate fake images (denoised images)
            fake_images = netG(noisy_images)

            # Classify all fake batch with D
            output_fake = netD(fake_images.detach()).view(-1)
            lossD_fake = criterion(output_fake, label_fake)
            lossD_fake.backward()
            optimizerD.step()

            ############################
            # (2) Update G network
            ###########################
            netG.zero_grad()
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output_fake_for_G = netD(fake_images).view(-1)
            # Calculate G's loss based on this output
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
                print(f'[{epoch}/{num_epochs}][{i}/{len(dataloader)}] '
                      f'Loss_D: {(lossD_real + lossD_fake):.4f} '
                      f'Loss_G: {lossG:.4f}')

    return G_losses, D_losses
