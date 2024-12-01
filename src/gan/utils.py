import torch
import numpy as np
import matplotlib.pyplot as plt

def weights_init(m):
    if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.Linear)):
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)

def imshow(img, title=None):
    npimg = img.numpy()
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    if title:
        plt.title(title)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
