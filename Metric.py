import torch
from torch import nn
from torch.nn import functional as F


class Metric(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 32)
        self.fc2 = nn.Linear(32, out_dim)
    
    def forward(self, m):
        m = self.fc1(m)
        m = torch.sigmoid(m)
        m = self.fc2(m)
        
        return m
