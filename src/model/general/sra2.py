import torch
from torch import nn


class SRA2(nn.Module):
    def __init__(self, in_dim, out_dim, layers=5):
        super().__init__()

        modules = [nn.Linear(in_dim, out_dim)]
        for _ in range(layers - 1):
            modules += [nn.GELU(approximate="tanh"), nn.Linear(out_dim, out_dim)]
        self.ffn = nn.Sequential(*modules)

    def forward(self, x):
        x=self.ffn(x)
        return x

