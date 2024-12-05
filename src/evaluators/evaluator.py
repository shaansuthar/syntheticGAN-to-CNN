import torch
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import config
import os


class Evaluator:
    def __init__(self, model, device=config.DEVICE):
        self.model = model.to(device)
        self.device = device

    def evaluate(self, test_loader, dataset_type):
        self.model.eval()
        y_true = []
        y_pred = []
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
        cm = confusion_matrix(y_true, y_pred)

        # Plot confusion matrix
        self.plot_confusion_matrix(cm, config.RESULTS_DIR, f'{dataset_type}_confusion_matrix.png')

        # Save classification report
        self.save_classification_report(report, config.RESULTS_DIR, f'{dataset_type}_classification_report.csv')

        return {'accuracy': accuracy, 'report': report}

    @staticmethod
    def plot_confusion_matrix(cm, save_dir, filename):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=range(config.NUM_CLASSES),
                    yticklabels=range(config.NUM_CLASSES))
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig(os.path.join(save_dir, filename))
        plt.close()

    @staticmethod
    def save_classification_report(report, save_dir, filename):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        df = pd.DataFrame(report).transpose()
        df.to_csv(os.path.join(save_dir, filename))    