import torch
from torch import nn
import numpy as np

class MLP(nn.Module):
    def __init__(self, dim_in, hidden_dim, dim_out, depth):
        super().__init__()
        self.depth = depth
        self.input = nn.Sequential(
            nn.Linear(dim_in, hidden_dim),
            nn.PReLU()
        )
        self.process = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.PReLU()
            ) for i in range(depth)
        ])
        self.output = nn.Linear(hidden_dim, dim_out)

    def forward(self, x):
        x = self.input(x)
        for layer in self.process:
            x = layer(x)
        return self.output(x)


# This class should be rewrite after env would be build
class Numbered_MLP(MLP):
    def __init__(self, dim_in, hidden_dim, dim_out, depth):
        super().__init__(dim_in, hidden_dim, dim_out, depth)

    def forward(self, x):
        x = self.input(x[0])
        for layer in self.process:
            x = layer(x)
        return self.output(x)


# This class should be rewrite after env would be build
class Embedding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.randn(1, 15) * x
