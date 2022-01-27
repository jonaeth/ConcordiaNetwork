import torch.nn as nn
import torch
from torch.nn.functional import one_hot
from torch.optim import Adam


class MixtureOfExpertsRegression(nn.Module):
    def __init__(self, input_vector_length):
        super().__init__()
        self.fc = nn.Linear(input_vector_length + 1 + 1, 1, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.optimizer = Adam(self.parameters(), lr=0.001)
        self.alpha = -1

    def forward(self, x, psl_predictions, nn_predictions):
        out = torch.concat([x, psl_predictions[:, 1].unsqueeze(dim=1), nn_predictions[:, 1].unsqueeze(dim=1)], axis=1)
        out = self.fc(out)
        return self.sigmoid(out)

    def fit(self, input_features, teacher_prediction, student_prediction, target_predictions):
        alpha = self.forward(input_features, teacher_prediction, student_prediction)
        loss = self.compute_loss(teacher_prediction, student_prediction, alpha, target_predictions)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.alpha = alpha.mean().item()

    def compute_loss(self, teacher_prediction, student_prediction, alpha, true_targets):
        softmax = torch.nn.Softmax(dim=1)
        loss_fn = torch.nn.CrossEntropyLoss()
        mixture_loss = loss_fn(alpha * softmax(student_prediction) + (1 - alpha) * softmax(teacher_prediction), true_targets)
        return mixture_loss
