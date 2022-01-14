from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import numpy as np

def compute_metrics(targets, predictions, threshold=0.5):
    predictions = np.where(np.exp(predictions) > threshold, 1, 0)
    recall = recall_score(targets, predictions)
    precision = precision_score(targets, predictions)
    f1 = f1_score(targets, predictions)
    accuracy = accuracy_score(targets, predictions)
    return recall, precision, f1, accuracy


def print_log_metrics(loss, accuracy, precision, recall, f1):
    print('\nVal set: Average loss: {:.4f}, Accuracy: {:.4f}%, precision: ({:.4f}), recall: ({:.4f}),'
          ' f1: ({:.4f}) \n'.format(loss, accuracy, precision, recall, f1))
