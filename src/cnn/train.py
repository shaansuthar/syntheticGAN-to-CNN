from model import ConvolutionNeuralNet as CNN
import dataset_dispatcher
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import config
import argparse


device = torch.device('cuda' if torch.cuda.is_available() else 'mps')

def train_cnn(dataset):
    run(dataset)

def run(dataset):

    # load data
    train_loader, test_loader = dataset_dispatcher.processing[dataset]

    model = CNN(num_classes=config.CIFAR_10_NUM_CLASSES).to(device)

    # cross entropy loss
    criterion = nn.CrossEntropyLoss()

    # Stochastic Gradient Descent
    optimizer = torch.optim.SGD(model.parameters(), lr=config.LEARNING_RATE, weight_decay = 0.005, momentum = 0.9)  

    for epoch in range(config.NUM_EPOCHS):
    # Load in the data in batches using the train_loader object
        for i, (images, labels) in enumerate(train_loader):  
            
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, config.NUM_EPOCHS, loss.item()))

    # Evaluate the model
    correct, total = evaluate(model, test_loader, dataset)
    print('Accuracy of the network on the {} test images: {} %'.format(50000, 100 * correct / total))

    torch.save(model.state_dict(), config.MODEL_PATH(f"trained_with_{dataset}.pt"))

'''
Evaluates model on test dataset, and generates a confusion matrix
'''
def evaluate(model, test_loader, dataset):
    with torch.no_grad():
        correct = 0
        total = 0
        y_pred = []
        y_true = []
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        cf_matrix = confusion_matrix(y_true, y_pred)
        df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in config.CIFAR_10_CLASSES],
                            columns = [i for i in config.CIFAR_10_CLASSES])
        plt.figure(figsize = (12,7))
        sn.heatmap(df_cm, annot=True)
        plt.savefig(config.CONFUSION_MATRIX_PATH(f'trained_with_{dataset}.png'))        
        
        return correct, total
