import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from training import CNN, NUM_CLASSES, DEVICE, RESULTS_DIR

def main():
    # load test data
    test_loader = load_test_data()

    # compare and contrast each model
    model_files = [f for f in os.listdir('./models') if f.endswith('.pkl')]
    for model_file in model_files:
        model_path = os.path.join('./models', model_file)
        model = CNN(num_classes=NUM_CLASSES)
        model.load_state_dict(torch.load(model_path, weights_only=True, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        print(f"Evaluating model: {model_file[:-4].upper()}")
        evaluate(model, test_loader, model_file[:-4])

def load_test_data():
    # Three-channel normalization
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

    return test_loader

def evaluate(model, test_loader, model_name):
    # Evaluate on test set
    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print("Test Accuracy:", accuracy)
    print("Classification Report:\n", report)
    
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(NUM_CLASSES),
                yticklabels=range(NUM_CLASSES))
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(os.path.join(RESULTS_DIR, f'{model_name}_confusion_matrix.png'))
    plt.close()

if __name__ == "__main__":
    main()