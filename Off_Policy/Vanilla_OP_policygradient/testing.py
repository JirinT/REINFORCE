import torch
import numpy as np
from torch.distributions import Categorical

# probs = np.full(2, 1/2)
probs = torch.tensor(np.full(2, 1/2))
pdf = Categorical(probs)
action = pdf.sample()
prob = pdf.log_prob(action).exp()