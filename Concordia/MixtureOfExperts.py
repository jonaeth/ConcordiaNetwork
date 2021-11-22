import torch.nn as nn
import torch
from torch.nn.functional import one_hot
from torch.optim import Adam


class MixtureOfExperts(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 3, kernel_size=3, stride=1)
        self.fc = nn.Linear(208, 1)
        self.avgpool_10x10 = nn.AdaptiveAvgPool2d((8, 8))
        self.flatten = nn.Flatten()
        self.w_noise = nn.Linear(in_channels, 1, bias=False)
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()
        self.optimizer = Adam(self.parameters(), lr=0.001)

    def forward(self, x, psl_predictions, nn_predictions):
        out = self.conv1(x)
        out = self.avgpool_10x10(out)
        out = self.flatten(out)
        out = torch.concat([torch.mean(out.reshape(1, *out.shape), axis=1)
                                 .reshape(1, -1)
                                 .repeat(psl_predictions.shape[0], 1), psl_predictions, nn_predictions], axis=1)
        out = self.fc(out)
        return self.sigmoid(out)

    def fit(self, image_feature_space, teacher_prediction, student_prediction, target_predictions):
        alpha = self.forward(image_feature_space, teacher_prediction, student_prediction)
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

