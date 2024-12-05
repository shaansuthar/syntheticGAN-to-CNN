import os
import config
from data_processor.data_handler import CIFAR10Dataset, NoisyCIFAR10
from data_processor.data_splitter import DataSplitter
from data_processor.synthetic_data import generate_synthetic_data
from cnn.cnn_trainer import CNNTrainer
from gan.gan_trainer import GANTrainer
from torch.utils.data import ConcatDataset

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

    print("Training GAN on 30k real images...")
    # Train the GAN
    noisy_train_dataset = NoisyCIFAR10(train_30k_dataset)
    gan_trainer = GANTrainer()
    G_losses, D_losses = gan_trainer.train(data_handler.get_dataloader(noisy_train_dataset)) # COME BACK AFTER LOOKING AT EVALUATOR

    print("Generating 20k synthetic images using the trained GAN...")
    # Generate synthetic data
    generator = gan_trainer.get_generator()
    synthetic_dataset = generate_synthetic_data(generator, 20000)
    synthetic_loader = data_handler.get_dataloader(synthetic_dataset)

    print("Combining real and synthetic datasets...")
    # Combine real and synthetic datasets
    combined_dataset = ConcatDataset([train_30k_dataset, synthetic_dataset])
    combined_loader = data_handler.get_dataloader(combined_dataset)

    print("Training CNN on 30k real images...")
    # Train CNN on real data
    cnn_trainer_real = CNNTrainer()
    cnn_trainer_real.train(train_30k_loader)
    print("Evaluating CNN trained on real data...")
    metrics_real = cnn_trainer_real.evaluate(data_handler.get_dataloader(test_dataset, shuffle=False))
    cnn_trainer_real.save_model(os.path.join(config.MODEL_DIR, 'cnn_real.pth'))

    print("Training CNN on 20k synthetic images...")
    # Train CNN on synthetic data
    cnn_trainer_synth = CNNTrainer()
    cnn_trainer_synth.train(synthetic_loader)
    print("Evaluating CNN trained on synthetic data...")
    metrics_synth = cnn_trainer_synth.evaluate(data_handler.get_dataloader(test_dataset, shuffle=False))
    cnn_trainer_synth.save_model(os.path.join(config.MODEL_DIR, 'cnn_synth.pth'))

    print("Training CNN on combined real and synthetic data...")
    # Train CNN on combined data
    cnn_trainer_combined = CNNTrainer()
    cnn_trainer_combined.train(combined_loader)
    print("Evaluating CNN trained on combined data...")
    metrics_combined = cnn_trainer_combined.evaluate(data_handler.get_dataloader(test_dataset, shuffle=False))
    cnn_trainer_combined.save_model(os.path.join(config.MODEL_DIR, 'cnn_combined.pth'))

    # Output evaluation metrics
    print("Evaluation on real data:")
    print(metrics_real)
    print("Evaluation on synthetic data:")
    print(metrics_synth)
    print("Evaluation on combined data:")
    print(metrics_combined)

if __name__ == "__main__":
    main()
