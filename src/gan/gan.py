import torch
import torch.nn as nn
import config

embedding_dim = 50  # Adjust as needed


class Generator(nn.Module):
    def __init__(self, num_classes=config.NUM_CLASSES, nz=3):
        super(Generator, self).__init__()
        self.nz = nz  # Number of channels in input images
        self.label_emb = nn.Embedding(num_classes, embedding_dim)
        self.main = nn.Sequential(
            # Input channels = nz + embedding_dim
            nn.Conv2d(self.nz + embedding_dim, 64, 4, 2, 1),  # Output: (64, 16, 16)
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),                     # Output: (128, 8, 8)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),                    # Output: (256, 4, 4)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),           # Output: (128, 8, 8)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),            # Output: (64, 16, 16)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),              # Output: (3, 32, 32)
            nn.Tanh()  # Output values between -1 and 1
        )

    def forward(self, x, labels):
        # Embed labels
        label_emb = self.label_emb(labels)
        # Reshape label embedding to match image dimensions
        label_emb = label_emb.unsqueeze(2).unsqueeze(3)
        label_emb = label_emb.expand(-1, -1, x.size(2), x.size(3))
        # Concatenate image and label embedding
        x = torch.cat([x, label_emb], 1)
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self, num_classes=config.NUM_CLASSES):
        super(Discriminator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, embedding_dim)
        self.main = nn.Sequential(
            # Input channels = 3 + embedding_dim
            nn.Conv2d(3 + embedding_dim, 64, 4, 2, 1),       # Output: (64, 16, 16)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),                     # Output: (128, 8, 8)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),                    # Output: (256, 4, 4)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1),
            nn.Sigmoid()  # Output probability between 0 and 1
        )

    def forward(self, x, labels):
        # Embed labels
        label_emb = self.label_emb(labels)
        # Reshape label embedding to match image dimensions
        label_emb = label_emb.unsqueeze(2).unsqueeze(3)
        label_emb = label_emb.expand(-1, -1, x.size(2), x.size(3))
        # Concatenate image and label embedding
        x = torch.cat([x, label_emb], 1)
        return self.main(x)
