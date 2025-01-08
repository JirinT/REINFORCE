import torch.nn as nn

class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear1 = nn.Linear(in_features=4, out_features=128)
        self.relu1 = nn.ReLU()

        self.linear2 = nn.Linear(in_features=128, out_features=128)
        self.relu2 = nn.ReLU()

        self.linear3 = nn.Linear(in_features=128, out_features=2)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)

        probs = self.softmax(x)

        return probs