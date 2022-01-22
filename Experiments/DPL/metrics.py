import numpy as np
from sklearn import metrics
import torch

def compute_metrics(targets, predictions, threshold=0.5):
    predictions = np.where(np.exp(predictions) > threshold, 1, 0)
    recall = metrics.recall_score(targets, predictions)
    precision = metrics.precision_score(targets, predictions)
    f1 = metrics.f1_score(targets, predictions)
    accuracy = metrics.accuracy_score(targets, predictions)
    return recall, precision, f1, accuracy


def print_log_metrics(loss, accuracy, precision, recall, f1):
    print('\nVal set: Average loss: {:.4f}, Accuracy: {:.4f}%, precision: ({:.4f}), recall: ({:.4f}),'
          ' f1: ({:.4f}) \n'.format(loss, accuracy, precision, recall, f1))


def f1_score(predictions, targets):
    f1 = metrics.f1_score(targets.argmax(axis=1).detach().cpu().numpy(), torch.exp(predictions[0]).argmax(axis=1).detach().cpu().numpy())
    return torch.Tensor([f1])


def precision_score(predictions, targets):
    precision = metrics.precision_score(targets.argmax(axis=1).detach().cpu().numpy(), torch.exp(predictions[0]).argmax(axis=1).detach().cpu().numpy())
    return torch.Tensor([precision])


def recall_score(predictions, targets):
    recall = metrics.recall_score(targets.argmax(axis=1).detach().cpu().numpy(), torch.exp(predictions[0]).argmax(axis=1).detach().cpu().numpy())
    return torch.Tensor([recall])


def accuracy_score(predictions, targets):
    accuracy = metrics.accuracy_score(targets.argmax(axis=1).detach().cpu().numpy(), torch.exp(predictions[0]).argmax(axis=1).detach().cpu().numpy())
    return torch.Tensor([accuracy])
