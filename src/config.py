import torch

NUM_CLASSES=10
BATCH_SIZE=64
DEVICE='cuda' if torch.cuda.is_available() else 'mps'