"""

Created on 2025/7/17 20:34
@author: 18310

"""

import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

class ResNet18_1ch(nn.Module):
    """
    Lightweight frame selector based on ResNet-18 adapted for single-channel input.
    Outputs logits for binary classification: [background vs abdomen-present].
    """
    def __init__(self, pretrained: bool = False):
        super().__init__()
        # Load standard ResNet-18
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.net = models.resnet18(weights=weights)
        # Adapt first conv layer to accept 1-channel (instead of 3)
        self.net.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=self.net.conv1.out_channels,
            kernel_size=self.net.conv1.kernel_size,
            stride=self.net.conv1.stride,
            padding=self.net.conv1.padding,
            bias=self.net.conv1.bias is not None,
        )
        # Replace the final fully connected layer for binary classification
        in_features = self.net.fc.in_features
        self.net.fc = nn.Linear(in_features, 2)

    def forward(self, x):
        """
        Forward pass.
        x shape: (B, 1, H, W)
        returns logits shape: (B, 2)
        """
        return self.net(x)