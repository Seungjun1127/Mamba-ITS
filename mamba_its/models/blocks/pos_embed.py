import torch
import torch.nn as nn

class PositionalEncoding2D(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
    def forward(self, x):
        pass
