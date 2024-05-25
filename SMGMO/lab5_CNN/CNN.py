import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, conv_block, num_classes=10, img_size=28):
        super(CNN, self).__init__()
        pool_ker_size = 2
        self.conv1 = conv_block(1, 32, pool_ker_size)
        self.conv2 = conv_block(32, 64, pool_ker_size)
        self.fc = nn.Linear(64 * ((img_size // pool_ker_size) // pool_ker_size) ** 2, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
