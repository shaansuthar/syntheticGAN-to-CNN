import os

from scipy.__config__ import show
import test
import config
from data_processor.data_handler import CIFAR10Dataset, NoisyCIFAR10
from data_processor.data_splitter import DataSplitter
from data_processor.synthetic_data import generate_synthetic_data
from cnn.cnn_trainer import CNNTrainer
from gan.gan_trainer import GANTrainer
from torch.utils.data import ConcatDataset
from evaluators.evaluator import Evaluator

def main():
    print("Loading CIFAR-10 dataset...")
    # Load CIFAR-10 dataset
    data_handler = CIFAR10Dataset()
    train_dataset = data_handler.dataset
    test_dataset = data_handler.get_test_dataset()

    print("Splitting dataset into 30k train and 20k unused...")
    # Split the dataset into 30k and 20k subsets
    data_splitter = DataSplitter(train_dataset)
    truncated_dataset, _ = data_splitter.split_indices(len(train_dataset), [30000, 20000])

    # Create datasets
    train_30k_dataset = data_splitter.get_subset(truncated_dataset)
    train_30k_loader = data_handler.get_dataloader(train_30k_dataset)
    train_full_loader = data_handler.get_dataloader(train_dataset)
    test_loader = data_handler.get_dataloader(test_dataset, shuffle=False)

    print("Training GAN on 30k real images...")
    # Train the GAN
    noisy_train_dataset = NoisyCIFAR10(train_30k_dataset)
    gan_trainer = GANTrainer()
    G_losses, D_losses = gan_trainer.train(data_handler.get_dataloader(noisy_train_dataset))
    gan_trainer.plot_gan_losses(G_losses, D_losses)

    print("Generating 20k synthetic images using the trained GAN...")
    # Generate synthetic data
    generator = gan_trainer.get_generator()
    synthetic_dataset = generate_synthetic_data(generator, 20000, show_samples=True)

    print("Combining real and synthetic datasets...")
    # Combine real and synthetic datasets
    combined_dataset = ConcatDataset([train_30k_dataset, synthetic_dataset])
    combined_loader = data_handler.get_dataloader(combined_dataset)

    print("Training CNN on 30k real images...")
    # Train CNN on real data
    cnn_trainer_30k_real = CNNTrainer()
    cnn_trainer_30k_real.train(train_30k_loader)
    print("Evaluating 30k real data CNN on test set...")
    metrics_30k_real = cnn_trainer_30k_real.evaluate(test_loader, 'cifar-30k-real')
    cnn_trainer_30k_real.save_model(os.path.join(config.MODEL_DIR, 'cnn_30k_real.pth'))

    print("Training CNN on combined 30k real and 20k synthetic data...")
    # Train CNN on combined data
    cnn_trainer_combined = CNNTrainer()
    cnn_trainer_combined.train(combined_loader)
    print("Evaluating 30k real + 20k synthetic CNN on test set...")
    metrics_combined = cnn_trainer_combined.evaluate(test_loader, 'cifar-30k-real-20k-fake')
    cnn_trainer_combined.save_model(os.path.join(config.MODEL_DIR, 'cnn_combined.pth'))

    print("Training CNN on 50k real images...")
    # Train CNN on synthetic data
    cnn_trainer_full_real = CNNTrainer()
    cnn_trainer_full_real.train(train_full_loader)
    print("Evaluating 50k real data CNN on test set...")
    metrics_full_real = cnn_trainer_full_real.evaluate(test_loader, 'cifar-50k-real')
    cnn_trainer_full_real.save_model(os.path.join(config.MODEL_DIR, 'cnn_full_real.pth'))

    # Output evaluation metrics
    print("Evaluation on 30k real data CNN:")
    print(metrics_30k_real['accuracy'])
    print("Evaluation on 30k real + 20k synthetic data CNN:")
    print(metrics_combined['accuracy'])
    print("Evaluation on 50k full real data CNN:")
    print(metrics_full_real['accuracy'])

if __name__ == "__main__":
    main()
