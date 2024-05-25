import torch.nn as nn

from ResidualBlock import ResidualBlock


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_stack = nn.Sequential(
            # 1*28*28
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, padding=1, stride=1),
            # 10*28*28
            nn.ReLU(),
            # 10*28*28
            ResidualBlock(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 10*14*14
            ResidualBlock(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 10*7*7
            ResidualBlock(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            # 10*6*6
            ResidualBlock(),
            nn.MaxPool2d(kernel_size=1, stride=2),
            # 10*3*3
            nn.Flatten(),
            nn.Linear(in_features=90, out_features=10)
        )

    def forward(self, x):
        return self.layer_stack(x)
