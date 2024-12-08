import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split, Subset, ConcatDataset
from torchvision.datasets import CIFAR10
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pandas as pd
import seaborn as sns

#############################
# Configuration and Hyperparameters
#############################
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps")
NUM_CLASSES = 10
BATCH_SIZE = 100
DATA_DIR = './data'
MODEL_DIR = './models'
RESULTS_DIR = './results'

# CNN Hyperparameters
CNN_LEARNING_RATE = 0.001
CNN_NUM_EPOCHS = 1  # Reduced for example; originally it was 30
CNN_OPT_WEIGHT_DECAY = 0.005
CNN_MOMENTUM = 0.9

# GAN Hyperparameters
NOISE_LEVEL = 0.1
GAN_LEARNING_RATE = 0.0002
GAN_NUM_EPOCHS = 1  # Reduced for example; originally it was 10
GAN_BETA1 = 0.5
GAN_BETA2 = 0.999

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
        self.transform = transform or transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.dataset = CIFAR10(root=root, train=True, download=True, transform=self.transform)
        self.test_dataset = CIFAR10(root=root, train=False, download=True, transform=self.transform)

    def get_train_val_split(self, train_size=0.8):
        train_length = int(len(self.dataset) * train_size)
        val_length = len(self.dataset) - train_length
        return random_split(self.dataset, [train_length, val_length])

    def get_test_dataset(self):
        return self.test_dataset

    def get_dataloader(self, dataset, batch_size=BATCH_SIZE, shuffle=True):
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

class NoisyCIFAR10(Dataset):
    def __init__(self, dataset, noise_level=NOISE_LEVEL):
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

class DataSplitter:
    def __init__(self, dataset):
        self.dataset = dataset

    def get_subset(self, indices):
        return Subset(self.dataset, indices)

    def split_indices(self, total_size, split_sizes):
        indices = list(range(total_size))
        np.random.shuffle(indices)
        splits = []
        start = 0
        for size in split_sizes:
            end = start + size
            splits.append(indices[start:end])
            start = end
        return splits

#############################
# Synthetic Data Generation (from GAN)
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
        img = (img + 1) / 2.0
        img = img.permute(1, 2, 0).numpy()
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f'{class_names[label]}')

    plt.tight_layout()
    plt.show()

def generate_synthetic_data(generator, num_samples, show_samples=False):
    generator.eval()
    synthetic_images = []
    synthetic_labels = []
    with torch.no_grad():
        for class_label in range(NUM_CLASSES):
            num_samples_per_class = num_samples // NUM_CLASSES
            for _ in range(num_samples_per_class // BATCH_SIZE):
                noisy_batch = torch.randn(BATCH_SIZE, 3, 32, 32).to(DEVICE)
                labels_batch = torch.full((BATCH_SIZE,), class_label, dtype=torch.long).to(DEVICE)
                fake_images = generator(noisy_batch, labels_batch)
                synthetic_images.append(fake_images.cpu())
                synthetic_labels.extend([class_label] * BATCH_SIZE)

    synthetic_images = torch.cat(synthetic_images)
    synthetic_dataset = SyntheticDataset(synthetic_images, synthetic_labels)

    if show_samples:
        visualize_batch(synthetic_images, synthetic_labels)
    return synthetic_dataset

#############################
# GAN Models
#############################
embedding_dim = 50

class Generator(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, nz=3):
        super(Generator, self).__init__()
        self.nz = nz
        self.label_emb = nn.Embedding(num_classes, embedding_dim)
        self.main = nn.Sequential(
            nn.Conv2d(self.nz + embedding_dim, 64, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x, labels):
        label_emb = self.label_emb(labels)
        label_emb = label_emb.unsqueeze(2).unsqueeze(3)
        label_emb = label_emb.expand(-1, -1, x.size(2), x.size(3))
        x = torch.cat([x, label_emb], 1)
        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(Discriminator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, embedding_dim)
        self.main = nn.Sequential(
            nn.Conv2d(3 + embedding_dim, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        label_emb = self.label_emb(labels)
        label_emb = label_emb.unsqueeze(2).unsqueeze(3)
        label_emb = label_emb.expand(-1, -1, x.size(2), x.size(3))
        x = torch.cat([x, label_emb], 1)
        return self.main(x)

def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)

#############################
# GAN Trainer
#############################
class GANTrainer:
    def __init__(self, noise_level=NOISE_LEVEL, device=DEVICE,
                 learning_rate=GAN_LEARNING_RATE, beta1=GAN_BETA1, beta2=GAN_BETA2, num_epochs=GAN_NUM_EPOCHS):
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
                # Update D
                self.netD.zero_grad()
                noisy_images, labels, real_images = data
                noisy_images = noisy_images.to(self.device)
                real_images = real_images.to(self.device)
                labels = labels.to(self.device)
                b_size = real_images.size(0)
                label_real = torch.full((b_size,), self.real_label, dtype=torch.float, device=self.device)
                label_fake = torch.full((b_size,), self.fake_label, dtype=torch.float, device=self.device)

                output_real = self.netD(real_images, labels).view(-1)
                lossD_real = self.criterion(output_real, label_real)
                lossD_real.backward()

                fake_images = self.netG(noisy_images, labels)
                output_fake = self.netD(fake_images.detach(), labels).view(-1)
                lossD_fake = self.criterion(output_fake, label_fake)
                lossD_fake.backward()
                self.optimizerD.step()

                # Update G
                self.netG.zero_grad()
                fake_images = self.netG(noisy_images, labels)
                output_fake_for_G = self.netD(fake_images, labels).view(-1)
                lossG_adv = self.criterion(output_fake_for_G, label_real)
                lossG_L1 = nn.L1Loss()(fake_images, real_images) * 100
                lossG = lossG_adv + lossG_L1
                lossG.backward()
                self.optimizerG.step()

                G_losses.append(lossG.item())
                D_losses.append((lossD_real + lossD_fake).item())

                if i % 100 == 0:
                    print(f'[{epoch}/{self.num_epochs}][{i}/{len(train_loader)}] '
                          f'Loss_D: {(lossD_real + lossD_fake):.4f} Loss_G: {lossG:.4f}')

        # Save the generator model
        torch.save(self.netG.state_dict(), os.path.join(MODEL_DIR, 'generator.pt'))
        return G_losses, D_losses

    def get_generator(self):
        return self.netG

    def plot_gan_losses(self, G_losses, D_losses):
        plt.figure()
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(G_losses, label="G")
        plt.plot(D_losses, label="D")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"{RESULTS_DIR}/gan_training_losses.pdf")

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

    def train(self, train_loader):
        for epoch in range(self.num_epochs):
            self.model.train()
            for i, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, self.num_epochs, loss.item()))

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
    # Load datasets
    print("Loading CIFAR-10 dataset...")
    data_handler = CIFAR10Dataset()
    train_dataset = data_handler.dataset
    test_dataset = data_handler.get_test_dataset()

    print("Splitting dataset into 30k train and 20k unused...")
    data_splitter = DataSplitter(train_dataset)
    truncated_dataset_indices, unused_dataset_indices = data_splitter.split_indices(len(train_dataset), [30000, 20000])
    train_30k_dataset = data_splitter.get_subset(truncated_dataset_indices)
    train_30k_loader = data_handler.get_dataloader(train_30k_dataset)
    train_full_loader = data_handler.get_dataloader(train_dataset)
    test_loader = data_handler.get_dataloader(test_dataset, shuffle=False)

    # Train GAN
    print("Training GAN on 30k real images...")
    noisy_train_dataset = NoisyCIFAR10(train_30k_dataset)
    gan_trainer = GANTrainer()
    G_losses, D_losses = gan_trainer.train(data_handler.get_dataloader(noisy_train_dataset))
    gan_trainer.plot_gan_losses(G_losses, D_losses)

    # Generate synthetic data
    print("Generating 20k synthetic images using the trained GAN...")
    generator = gan_trainer.get_generator()
    synthetic_dataset = generate_synthetic_data(generator, 20000, show_samples=False)

    # Combine real and synthetic datasets
    print("Combining real and synthetic datasets...")
    combined_dataset = ConcatDataset([train_30k_dataset, synthetic_dataset])
    combined_loader = data_handler.get_dataloader(combined_dataset)

    # Train CNN on 30k real images
    print("Training CNN on 30k real images...")
    cnn_trainer_30k_real = CNNTrainer()
    cnn_trainer_30k_real.train(train_30k_loader)
    torch.save(cnn_trainer_30k_real.model.state_dict(), os.path.join(MODEL_DIR, 'cnn_30k_real.pkl'))

    # Train CNN on combined data
    print("Training CNN on combined 30k real and 20k synthetic data...")
    cnn_trainer_combined = CNNTrainer()
    cnn_trainer_combined.train(combined_loader)
    torch.save(cnn_trainer_combined.model.state_dict(), os.path.join(MODEL_DIR, 'cnn_combined.pkl'))

    # Train CNN on full 50k real data
    print("Training CNN on 50k real images...")
    cnn_trainer_full_real = CNNTrainer()
    cnn_trainer_full_real.train(train_full_loader)
    torch.save(cnn_trainer_full_real.model.state_dict(), os.path.join(MODEL_DIR, 'cnn_full_real.pkl'))

if __name__ == "__main__":
    main()
