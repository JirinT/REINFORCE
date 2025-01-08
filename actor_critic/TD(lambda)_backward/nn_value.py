import torch.nn as nn
import torch.optim as optim

class ValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.linear1 = nn.Linear(in_features=4, out_features=64)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(in_features=64, out_features=1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        value = self.linear2(x)

        return value
