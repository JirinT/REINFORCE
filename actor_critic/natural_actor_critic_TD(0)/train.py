# THIS IS A FULLY ONLINE ACTOR CRITIC ALGORITHIM USING TWO CRITICS AND ONE ACTOR
# THE CRITIC IS UPDATED BASED ON TD(0) ALGORITHM - BOOTSTRAPPING
import numpy as np
import gymnasium as gym
import torch
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.distributions import Categorical
import torch.nn.functional as F
from nn_policy import PolicyNet
from nn_value import ValueNet
from gymnasium.wrappers import RecordVideo, RecordEpisodeStatistics

def get_action(policy, state):
    probs = policy(state)
    pdf = Categorical(probs)
    action = pdf.sample()
    log_prob = pdf.log_prob(action)
    return action, log_prob

# def update_value_fcn(value_fcn, value, next_value, reward):
#     advantage = reward + GAMMA*next_value - value
#     loss = -advantage*value

#     value_fcn.optimizer.zero_grad()
#     loss.backward()
#     value_fcn.optimizer.step()

def update_value_fcn(value_fcn, value, next_value, reward):
    target = reward + GAMMA*next_value
    loss = F.mse_loss(value, target)

    value_fcn.optimizer.zero_grad()
    loss.backward()
    value_fcn.optimizer.step()

def get_f_vector(policy, log_prob):
    f = []
    policy.optimizer.zero_grad()
    log_prob.backward(retain_graph=True)
    f = torch.cat([p.grad.view(-1) for p in policy.parameters() if p.requires_grad and p.grad is not None])
    return f

def compute_x(policy, x, f):
    # compute x:
    if x == []:
        x = torch.zeros_like(f)

    x += 0.001*(reward + GAMMA*next_value-value-torch.dot(x,f))*f
    # reshape x:
    reshaped_x = []
    start_idx = 0
    for p in policy.parameters():
        if p.requires_grad:
            num_parameters = p.numel()
            layer = x[start_idx:start_idx+num_parameters].view(p.shape)
            reshaped_x.append(layer)
            start_idx += num_parameters
    return x, reshaped_x

# ------- MAIN -------
POLICY_LEARNING_RATE = .0001
CRITIC_LEARNING_RATE = .005
GAMMA = 0.99
NUM_EPISODES = 800
VIDEO_PERIOD = 100

policy = PolicyNet(POLICY_LEARNING_RATE)
value_function = ValueNet(CRITIC_LEARNING_RATE)

env = gym.make("CartPole-v1", render_mode="rgb_array") # the states are x-axis position and velocity of the car
env = RecordVideo(env, video_folder="cartpole-agent", name_prefix="training",
                  episode_trigger=lambda x: x % VIDEO_PERIOD == 0)
env = RecordEpisodeStatistics(env)

epi_stats = {
    "time" : [],
    "total_reward" : [],
    "length" : []
}

x = []
for episode in tqdm(range(NUM_EPISODES)):
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32)
    done = False
    while not done:
        action, log_prob = get_action(policy, state)

        next_state, reward, terminated, truncated, info = env.step(action.item())
        next_state = torch.tensor(next_state, dtype=torch.float32)

        value = value_function(state)

        if terminated:
            next_value = torch.tensor([0.0], dtype=torch.float32)
            # reward = -1
        else:
            next_value = value_function(next_state).detach()

        update_value_fcn(value_function, value, next_value, reward)
        f = get_f_vector(policy, log_prob) # gradients from policy network
        x, matrix_x = compute_x(policy, x, f) # advantage critic update

        # update policy network:
        with torch.no_grad():
            for i, p in enumerate(policy.parameters()):
                if p.requires_grad:
                    new_param = p + POLICY_LEARNING_RATE*matrix_x[i]
                    p.copy_(new_param)
        state = next_state
        done = terminated or truncated

    epi_stats["length"].append(info["episode"]["l"])
    epi_stats["time"].append(info["episode"]["t"])
    epi_stats["total_reward"].append(info["episode"]["r"])

env.close()

plt.figure(1)
plt.plot(epi_stats["total_reward"])
plt.title("Training statistics")
plt.ylabel("Total rewards")
plt.xlabel("Episode number")
plt.grid(True)

plt.show()