import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels=10, out_channels=10, stride=1):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(10)
        )
        self.active = nn.ReLU()

        self.shortcut = nn.Sequential()

    def forward(self, x):
        out = self.layer_stack(x)
        out += self.shortcut(x)
        out = self.active(out)
        return out
