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

    def forward(self, x, psl_predictions, nn_predictions):
        out = torch.concat([x, psl_predictions, nn_predictions], axis=1)
        out = self.fc(out)
        return self.sigmoid(out)

    def fit(self, input_features, teacher_prediction, student_prediction, target_predictions):
        alpha = self.forward(input_features, teacher_prediction, student_prediction)
        loss = self.compute_loss(teacher_prediction, student_prediction, alpha, target_predictions)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def compute_loss(self, teacher_prediction, student_prediction, alpha, true_targets):
        softmax = torch.nn.Softmax(dim=1)
        mixture_loss = -torch.sum(
            torch.log(alpha * softmax(student_prediction.detach()) +
                      (1 - alpha) * teacher_prediction * one_hot(true_targets.detach(), num_classes=8)))
        return mixture_loss
