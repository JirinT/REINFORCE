import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class PolicyNet(nn.Module):
    def __init__(self, learning_rate):
        super().__init__()

        self.lr = learning_rate

        self.linear1 = nn.Linear(in_features=4, out_features=128)
        self.relu1 = nn.ReLU()

        self.linear2 = nn.Linear(in_features=128, out_features=128)
        self.relu2 = nn.ReLU()

        self.linear3 = nn.Linear(in_features=128, out_features=2)
        self.softmax = nn.Softmax(dim=0)

        self.optimizer = optim.Adam(params=self.parameters(), lr=self.lr)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)

        probs = self.softmax(x)

        return probs #probs are probabilities (discrete probability distribution) of actions that yield highest rewards. its the softmax, we then sample from this probability distribution