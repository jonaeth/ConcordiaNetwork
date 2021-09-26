import torch.nn as nn
import torchvision.models as models


class MobileNet(nn.Module):
    def __init__(self, pretrained=True):
        super(MobileNet, self).__init__()

        mobileNet = models.mobilenet_v2(pretrained=pretrained)

        self.features = mobileNet.features

    def forward(self, x):
        x = self.features(x)
        return [x]