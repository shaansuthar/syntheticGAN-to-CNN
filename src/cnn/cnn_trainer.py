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
                                   weight_decay=config.CNN_OPT_WEIGHT_DECAY, momentum=config.CNN_MOMENTUM)

    def train(self, train_loader):
        for epoch in range(config.CNN_NUM_EPOCHS):
        # Load in the data in batches using the train_loader object
            for i, (images, labels) in enumerate(train_loader):  
                
                # Move tensors to the configured device
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, config.CNN_NUM_EPOCHS, loss.item()))

    def evaluate(self, test_loader, dataset_type):
        evaluator = Evaluator(self.model, self.device)
        metrics = evaluator.evaluate(test_loader, dataset_type)
        return metrics

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
