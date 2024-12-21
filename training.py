import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from torchvision.datasets import CIFAR10
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pandas as pd
import seaborn as sns

#############################
# Configuration and Hyperparameters
#############################
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 10
BATCH_SIZE = 100
DATA_DIR = './data'
MODEL_DIR = './models'
RESULTS_DIR = './results'

# CNN Hyperparameters
CNN_LEARNING_RATE = 0.001
CNN_NUM_EPOCHS = 20
CNN_OPT_WEIGHT_DECAY = 0.005
CNN_MOMENTUM = 0.9

# GAN Hyperparameters
GAN_LEARNING_RATE = 0.0001
GAN_NUM_EPOCHS = 50
GAN_BETA1 = 0.5
GAN_BETA2 = 0.999
LATENT_DIM = 128
EMBEDDING_DIM = 100

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

#############################
# Model Definition: CNN
#############################
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.max_pool1 = nn.MaxPool2d(2, 2)

        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv_layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.max_pool2 = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(1600, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = self.max_pool1(out)
        out = self.conv_layer3(out)
        out = self.conv_layer4(out)
        out = self.max_pool2(out)
        out = out.reshape(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out

#############################
# Evaluator Class
#############################
class Evaluator:
    def __init__(self, model, device=DEVICE):
        self.model = model.to(device)
        self.device = device

    def evaluate(self, test_loader, dataset_type):
        self.model.eval()
        y_true = []
        y_pred = []
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
        cm = confusion_matrix(y_true, y_pred)

        # Plot confusion matrix
        self.plot_confusion_matrix(cm, RESULTS_DIR, f'{dataset_type}_confusion_matrix.png')
        # Save classification report
        self.save_classification_report(report, RESULTS_DIR, f'{dataset_type}_classification_report.csv')
        return {'accuracy': accuracy, 'report': report}

    @staticmethod
    def plot_confusion_matrix(cm, save_dir, filename):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=range(NUM_CLASSES),
                    yticklabels=range(NUM_CLASSES))
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig(os.path.join(save_dir, filename))
        plt.close()

    @staticmethod
    def save_classification_report(report, save_dir, filename):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        df = pd.DataFrame(report).transpose()
        df.to_csv(os.path.join(save_dir, filename))

#############################
# Dataset Handling
#############################
class CIFAR10Dataset:
    def __init__(self, root=DATA_DIR, transform=None):
        # Added data augmentation to improve GAN input diversity
        self.transform = transform or transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])
        self.dataset = CIFAR10(root=root, train=True, download=True, transform=self.transform)

        # Test set uses standard normalization since we do not want to augment test data.
        self.test_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])
        self.test_dataset = CIFAR10(root=root, train=False, download=True, transform=self.test_transform)

    def get_test_dataset(self):
        return self.test_dataset

    def get_dataloader(self, dataset, batch_size=BATCH_SIZE, shuffle=True):
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

#############################
# Synthetic Dataset
#############################
class SyntheticDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

def visualize_batch(images, labels, num_samples=10):
    if len(images) > num_samples:
        indices = np.random.choice(len(images), num_samples, replace=False)
        images = images[indices]
        labels = [labels[i] for i in indices]

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    for idx, (ax, img, label) in enumerate(zip(axes.flat, images, labels)):
        img = (img * 0.5) + 0.5  # reverse normalization
        img = img.permute(1, 2, 0).numpy()
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f'{class_names[label]}')

    plt.tight_layout()

def visualize_each_class(synthetic_dataset, samples_per_class=5):
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    fig, axes = plt.subplots(NUM_CLASSES, samples_per_class, figsize=(samples_per_class*3, NUM_CLASSES*3))
    for c in range(NUM_CLASSES):
        class_indices = [i for i, lbl in enumerate(synthetic_dataset.labels) if lbl == c]
        sample_indices = class_indices[:samples_per_class]
        sample_images = synthetic_dataset.images[sample_indices]
        sample_labels = [synthetic_dataset.labels[i] for i in sample_indices]

        for j, (img, lbl) in enumerate(zip(sample_images, sample_labels)):
            ax = axes[c, j]
            img = (img * 0.5) + 0.5
            img = img.permute(1, 2, 0).numpy()
            ax.imshow(img)
            ax.axis('off')
            if j == 0:
                ax.set_ylabel(class_names[c], rotation=90, size='large')

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'generated_images_per_class.png'))
    plt.close()

def generate_synthetic_data(generator, num_samples, device=DEVICE, latent_dim=LATENT_DIM, show_samples=False):
    generator.eval()
    synthetic_images = []
    synthetic_labels = []
    num_samples_per_class = num_samples // NUM_CLASSES

    with torch.no_grad():
        for class_label in range(NUM_CLASSES):
            for _ in range(num_samples_per_class // BATCH_SIZE):
                z = torch.randn(BATCH_SIZE, latent_dim, device=device)
                labels_batch = torch.full((BATCH_SIZE,), class_label, dtype=torch.long, device=device)
                fake_images, _ = generator(z, labels_batch)
                synthetic_images.append(fake_images.cpu())
                synthetic_labels.extend([class_label] * BATCH_SIZE)

    synthetic_images = torch.cat(synthetic_images)
    synthetic_dataset = SyntheticDataset(synthetic_images, synthetic_labels)

    if show_samples:
        visualize_batch(synthetic_images, synthetic_labels)
        plt.show()

    return synthetic_dataset

#############################
# ACGAN Models
#############################
def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)

class ACGANGenerator(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, latent_dim=LATENT_DIM, embed_dim=EMBEDDING_DIM):
        super(ACGANGenerator, self).__init__()
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim

        self.label_emb = nn.Embedding(num_classes, embed_dim)

        self.init_size = 4
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + embed_dim, 256 * self.init_size * self.init_size),
            nn.BatchNorm1d(256 * self.init_size * self.init_size),
            nn.ReLU(True)
        )

        self.main = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False), # 8x8
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),  # 16x16
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),     # 32x32
            nn.Tanh()
        )

    def forward(self, z, labels):
        c = self.label_emb(labels)        # [B, embed_dim]
        x = torch.cat([z, c], 1)          # [B, latent_dim+embed_dim]
        out = self.fc(x)                  # [B, 256*4*4]
        out = out.view(out.size(0), 256, self.init_size, self.init_size)
        img = self.main(out)              # [B, 3, 32,32]
        return img, labels

class ACGANDiscriminator(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, embed_dim=EMBEDDING_DIM):
        super(ACGANDiscriminator, self).__init__()
        self.num_classes = num_classes
        self.img_size = 32

        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),  
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # After main, feature map is 4x4, 256 channels.
        self.feature_size = 256*4*4

        # Real/Fake head
        self.adv_head = nn.Linear(self.feature_size, 1)

        # Class head
        self.aux_head = nn.Linear(self.feature_size, num_classes)

    def forward(self, img):
        features = self.main(img)
        features = features.view(features.size(0), -1)
        validity = self.adv_head(features)
        class_logits = self.aux_head(features)
        return validity, class_logits

#############################
# ACGAN Trainer
#############################
class ACGANTrainer:
    def __init__(self, device=DEVICE,
                 learning_rate=GAN_LEARNING_RATE, beta1=GAN_BETA1, beta2=GAN_BETA2, num_epochs=GAN_NUM_EPOCHS):
        self.device = device
        self.num_epochs = num_epochs
        self.netG = ACGANGenerator(num_classes=NUM_CLASSES, latent_dim=LATENT_DIM, embed_dim=EMBEDDING_DIM).to(self.device)
        self.netD = ACGANDiscriminator(num_classes=NUM_CLASSES, embed_dim=EMBEDDING_DIM).to(self.device)
        self.netG.apply(weights_init)
        self.netD.apply(weights_init)

        self.criterion_gan = nn.BCEWithLogitsLoss()
        self.criterion_aux = nn.CrossEntropyLoss()

        self.optimizerD = optim.Adam(self.netD.parameters(), lr=learning_rate, betas=(beta1, beta2))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=learning_rate, betas=(beta1, beta2))

    def train(self, train_loader):
        G_losses = []
        D_losses = []

        print("Starting ACGAN Training...")
        for epoch in range(self.num_epochs):
            for i, (real_images, labels) in enumerate(train_loader):
                real_images, labels = real_images.to(self.device), labels.to(self.device)
                b_size = real_images.size(0)

                # Create real/fake labels for adversarial loss
                valid = torch.full((b_size,), 1.0, device=self.device)
                fake = torch.full((b_size,), 0.0, device=self.device)

                # (1) Train Discriminator
                self.netD.zero_grad()

                # Forward pass real images
                validity_real, class_logits_real = self.netD(real_images)
                validity_real = validity_real.view(-1)
                d_real_loss = self.criterion_gan(validity_real, valid)
                d_real_class_loss = self.criterion_aux(class_logits_real, labels)

                # Generate fake images
                z = torch.randn(b_size, LATENT_DIM, device=self.device)
                gen_labels = torch.randint(0, NUM_CLASSES, (b_size,), device=self.device)
                fake_images, _ = self.netG(z, gen_labels)

                # Forward pass fake images
                validity_fake, class_logits_fake = self.netD(fake_images.detach())
                validity_fake = validity_fake.view(-1)
                d_fake_loss = self.criterion_gan(validity_fake, fake)

                d_loss = d_real_loss + d_fake_loss + d_real_class_loss
                d_loss.backward()
                self.optimizerD.step()

                # (2) Train Generator
                self.netG.zero_grad()
                validity_fake_for_g, class_logits_fake_for_g = self.netD(fake_images)
                validity_fake_for_g = validity_fake_for_g.view(-1)

                g_adv_loss = self.criterion_gan(validity_fake_for_g, valid)
                g_class_loss = self.criterion_aux(class_logits_fake_for_g, gen_labels)
                g_loss = g_adv_loss + g_class_loss
                g_loss.backward()
                self.optimizerG.step()

                G_losses.append(g_loss.item())
                D_losses.append(d_loss.item())

                if i % 100 == 0:
                    print(f'[{epoch+1}/{self.num_epochs}][{i}/{len(train_loader)}] Loss_D: {d_loss:.4f} Loss_G: {g_loss:.4f}')

        torch.save(self.netG.state_dict(), os.path.join(MODEL_DIR, 'acgan_generator.pt'))
        return G_losses, D_losses

    def get_generator(self):
        return self.netG

    def plot_gan_losses(self, G_losses, D_losses):
        plt.figure(figsize=(10,5))
        plt.title("ACGAN Training Losses")
        plt.plot(G_losses, label="G Loss")
        plt.plot(D_losses, label="D Loss")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(RESULTS_DIR, 'acgan_loss_plot.png'))
        plt.close()

#############################
# CNN Trainer
#############################
class CNNTrainer:
    def __init__(self, num_classes=NUM_CLASSES, learning_rate=CNN_LEARNING_RATE,
                 num_epochs=CNN_NUM_EPOCHS, device=DEVICE):
        self.device = device
        self.num_epochs = num_epochs
        self.model = CNN(num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate,
                                   weight_decay=CNN_OPT_WEIGHT_DECAY, momentum=CNN_MOMENTUM)
        self.train_losses = []

    def train(self, train_loader):
        self.train_losses = []
        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0
            for i, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            epoch_loss = running_loss / len(train_loader)
            self.train_losses.append(epoch_loss)
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, self.num_epochs, epoch_loss))

    def evaluate(self, test_loader, dataset_type):
        evaluator = Evaluator(self.model, self.device)
        metrics = evaluator.evaluate(test_loader, dataset_type)
        return metrics

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

#############################
# Main Training Workflow
#############################
def main():
    print("Loading CIFAR-10 dataset...")
    data_handler = CIFAR10Dataset()
    train_dataset = data_handler.dataset
    test_dataset = data_handler.get_test_dataset()

    # Number of samples per class for the 30k subset:
    samples_per_class = 3000
    total_classes = NUM_CLASSES  # 10 for CIFAR-10

    # Extract targets to identify classes
    train_targets = torch.tensor(train_dataset.targets)
    balanced_indices = []

    for class_id in range(total_classes):
        # Get all indices for this class
        class_indices = (train_targets == class_id).nonzero(as_tuple=True)[0].tolist()
        # Shuffle them
        np.random.shuffle(class_indices)
        # Pick the first samples_per_class indices
        selected_indices = class_indices[:samples_per_class]
        balanced_indices.extend(selected_indices)

    # Shuffle final indices to avoid any class-based ordering in the final dataset
    np.random.shuffle(balanced_indices)
    train_30k_dataset = torch.utils.data.Subset(train_dataset, balanced_indices)

    train_30k_loader = data_handler.get_dataloader(train_30k_dataset)
    test_loader = data_handler.get_dataloader(test_dataset, shuffle=False)
    train_full_loader = data_handler.get_dataloader(train_dataset)

    # Train ACGAN on 30k real images (with augmentation and updated hyperparameters)
    print("Training ACGAN on 30k real images with improvements...")
    acgan_trainer = ACGANTrainer()
    G_losses, D_losses = acgan_trainer.train(train_30k_loader)
    acgan_trainer.plot_gan_losses(G_losses, D_losses)

    # Generate synthetic data using the ACGAN
    print("Generating 20k synthetic images using the improved ACGAN...")
    generator = acgan_trainer.get_generator()
    synthetic_dataset = generate_synthetic_data(generator, 20000, device=DEVICE, show_samples=False)

    # Visualize a few generated images
    print("Visualizing generated images per class...")
    visualize_each_class(synthetic_dataset, samples_per_class=5)

    # Combine real and synthetic datasets
    print("Combining real and synthetic datasets...")
    combined_dataset = ConcatDataset([train_30k_dataset, synthetic_dataset])
    combined_loader = data_handler.get_dataloader(combined_dataset)

    # Train CNN on 30k real images
    print("Training CNN on 30k real images...")
    cnn_trainer_30k_real = CNNTrainer()
    cnn_trainer_30k_real.train(train_30k_loader)
    cnn_trainer_30k_real.save_model(os.path.join(MODEL_DIR, 'cnn_30k_real.pkl'))

    # Train CNN on combined data
    print("Training CNN on combined 30k real + 20k synthetic data...")
    cnn_trainer_combined = CNNTrainer()
    cnn_trainer_combined.train(combined_loader)
    cnn_trainer_combined.save_model(os.path.join(MODEL_DIR, 'cnn_combined.pkl'))

    # Train CNN on full 50k real data
    print("Training CNN on full 50k real images...")
    cnn_trainer_full_real = CNNTrainer()
    cnn_trainer_full_real.train(train_full_loader)
    cnn_trainer_full_real.save_model(os.path.join(MODEL_DIR, 'cnn_full_real.pkl'))

    # Plot training losses for all three CNNs
    print("Plotting CNN training losses for all three conditions...")
    plt.figure(figsize=(10,5))
    plt.title("CNN Training Losses")
    plt.plot(cnn_trainer_30k_real.train_losses, label="30k Real")
    plt.plot(cnn_trainer_combined.train_losses, label="30k Real + 20k Synthetic")
    plt.plot(cnn_trainer_full_real.train_losses, label="Full 50k Real")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, 'cnn_all_losses_plot.png'))
    plt.close()

if __name__ == "__main__":
    main()