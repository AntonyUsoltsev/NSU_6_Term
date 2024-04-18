import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation):
        super(MLP, self).__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_sizes[0] if len(hidden_sizes) != 0 else output_size))
        layers.append(activation())
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(activation())
        layers.append(nn.Linear(hidden_sizes[-1] if len(hidden_sizes) != 0 else input_size, output_size))
        #layers.append(nn.Sigmoid())

        self.stack = nn.Sequential(*layers)

    def forward(self, x):
        x = self.stack(x)

        return x
