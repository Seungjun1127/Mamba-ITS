import torch
import torch.nn as nn

class LocalSSMBlock(nn.Module):
    def __init__(self, dim, config):
        super().__init__()
        self.dim = dim
        self.config = config
    def forward(self, x):
        pass
