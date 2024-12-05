import torch
import torch.nn as nn
import torch.optim as optim
from cnn.cnn import CNN
from evaluators.evaluator import Evaluator
import config

class CNNTrainer:
    def __init__(self, num_classes=config.NUM_CLASSES, learning_rate=config.CNN_LEARNING_RATE,
                 num_epochs=config.CNN_NUM_EPOCHS, device=config.DEVICE):
        self.device = device
        self.num_epochs = num_epochs
        self.model = CNN(num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate,
                                   weight_decay=0.005, momentum=0.9)

    def train(self, train_loader):
        self.model.train()
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            for i, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward + backward + optimize
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

    def evaluate(self, test_loader):
        evaluator = Evaluator(self.model, self.device)
        metrics = evaluator.evaluate(test_loader)
        return metrics

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
