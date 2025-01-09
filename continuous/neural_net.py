import torch.nn as nn
import torch.optim as optim

class PolicyNet(nn.Module):
    def __init__(self, learning_rate):
        super().__init__()

        self.lr = learning_rate

        self.linear1 = nn.Linear(in_features=4, out_features=128)
        self.relu1 = nn.ReLU()

        self.linear2 = nn.Linear(in_features=128, out_features=128)
        self.relu2 = nn.ReLU()

        self.linear3 = nn.Linear(in_features=128, out_features=2)

        self.optimizer = optim.Adam(params=self.parameters(), lr=self.lr)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)

        mean = x[0]
        std = nn.functional.softplus(x[1]) # softplus so the standard deviation is always positive

        return mean, std # the prediction is the mean and standard deviation of a probability density function.