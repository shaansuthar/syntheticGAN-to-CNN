import torch

# General Configurations
DEVICE = torch.device('mps' if torch.mps.is_available() else 'cpu')
NUM_CLASSES = 10
BATCH_SIZE = 100

# Paths
DATA_DIR = '../data'
MODEL_DIR = '../models'
RESULTS_DIR = '../results'

# Hyperparameters
CNN_LEARNING_RATE = 0.001
CNN_NUM_EPOCHS = 30
CNN_OPT_WEIGHT_DECAY = 0.005
CNN_MOMENTUM = 0.9
NOISE_LEVEL = 0.1
GAN_LEARNING_RATE = 0.0002
GAN_NUM_EPOCHS = 10
GAN_BETA1 = 0.5
GAN_BETA2 = 0.999
