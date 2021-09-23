import torch
import pickle as pkl
from torch import nn
from torch.nn.modules.module import Module

class Regressor(Module):
    def __init__(self, config, out_dim) -> None:
        super().__init__()
        assert hasattr(config, 'in_dim')
        self.classifier = nn.Linear(config.in_dim, out_dim)
    
        
    def __call__(self, x):
        return self.classifier(x)
        