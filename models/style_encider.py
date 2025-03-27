import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiLevelStyleEncoder(nn.Module):
    def __init__(self, in_dim, out_dim=768):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, (in_dim // out_dim) * out_dim)
        self.fc2 = nn.Linear((in_dim // out_dim) * out_dim, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x.unsqueeze(1)