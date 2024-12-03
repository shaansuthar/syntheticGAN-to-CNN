LEARNING_RATE = 0.001
NUM_EPOCHS = 20
CIFAR_10_CLASSES = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
CIFAR_10_NUM_CLASSES = len(CIFAR_10_CLASSES)
MODEL_PATH = lambda x: f"../../models/{x}"
BATCH_SIZE = 64
CONFUSION_MATRIX_PATH = lambda x: f"../../results/{x}"

