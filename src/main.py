import torch
import config
import subprocess


def main():

    # Train the GAN
    subprocess.run(["python", "./gan/train.py"], check=True) # Note: generator model params saved as './models/generator.pt'

    # GAN generates x # of images for each class (maybe experiment with different values of x)

    # Create two datasets: CIFAR-10 and Noisy CIFAR-10 (with generated images)

    # Train the CNN on only CIFAR-10 dataset sup
    subprocess.run(["python", "./cnn/train.py", "--dataset", "cifar10"], check=True)

    # Train the CNN on the Noisy CIFAR-10 dataset, TODO: might need to refactor the datasets logic in cnn to handle this

    # Perform evaluation on both models and compare the results

    pass




'''
Using the trained generator to synthesize denoised images
'''
def generate_images(netG):
    print("Generating synthetic images...")
    netG.eval()
    synthetic_images = []
    synthetic_labels = []
    with torch.no_grad():
        for class_label in range(config.NUM_CLASSES):
            print(f"Generating images for class {class_label}")
            num_samples = 5000  # Number of images to generate per class
            for _ in range(num_samples // config.NUM_CLASSES):
                # Generate noisy images
                noisy_batch = torch.randn(config.BATCH_SIZE, 3, 32, 32).to(config.DEVICE)
                labels_batch = torch.full((config.BATCH_SIZE,), class_label, dtype=torch.long).to(config.DEVICE)
                fake_images = netG(noisy_batch, labels_batch)
                synthetic_images.append(fake_images.cpu())
                synthetic_labels.extend([class_label] * config.BATCH_SIZE)

    # Stack synthetic images and labels
    synthetic_images = torch.cat(synthetic_images)
    synthetic_labels = torch.tensor(synthetic_labels)    

if __name__ == "__main__":
    main()