# Create an environment

# Define a neural network = the policy

# Do the train loop :D
## function for computing the reward in each time step - the network will be updated after each episode t-times.
## function for computing the custom gaussian loss -> gaussian_pdf_value -> logarithm of the pdf value -> reward times the log of the pdf.
## optimizer and updating of the network
import numpy as np
import gymnasium as gym
import torch
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.distributions import Normal
from neural_net import PolicyNet
from gymnasium.wrappers import RecordVideo, RecordEpisodeStatistics

import os
import sys
from cartpole_environment import ContinuousCartPoleEnv
env_path = os.path.join(os.path.abspath(os.getcwd()), '..\\Environments\\ContinuousCartPole')
sys.path.append(env_path)

def custom_loss(action, reward, pdf):
    log_prob = pdf.log_prob(action) # log_prob(action) = log(gaussian(action, mean, std))
    loss = -log_prob*reward
    return loss

def get_action(mean, std): #the actions are picked from gaussian distribution of actions
    pdf = Normal(mean, std)
    action = pdf.sample()
    return action, pdf

def cummulative_reward(rewards, gamma):
    R = torch.zeros(len(rewards), dtype=torch.float32, requires_grad=False)
    running_sum = 0.0
    for t in reversed(range(len(rewards))):
        running_sum = rewards[t] + gamma*running_sum
        R[t] = running_sum

    # R = (R-R.mean())/R.std() # whitening discounted cummulative rewards
    return R

def update_policy(policy_net, trajectory):
    trajectory_length = len(trajectory["states"])
    loss = []
    cumm_rewards = cummulative_reward(trajectory["rewards"], gamma=0.99)

    for step in range(trajectory_length):
        loss.append(custom_loss(trajectory["actions"][step], cumm_rewards[step], trajectory["pdfs"][step]))

    total_loss = sum(loss)
    episode_loss.append(total_loss)

    # Zero gradients, perform backward pass, and update policy
    policy_net.optimizer.zero_grad()
    total_loss.backward()
    policy_net.optimizer.step()

# ------------MAIN------------:
NUM_EPISODES = 500
LEARNING_RATE = .001
VIDEO_PERIOD = 2

env = ContinuousCartPoleEnv()
# env = gym.make("MountainCarContinuous-v0", render_mode="rgb_array") # the states are x-axis position and velocity of the car

policy = PolicyNet(learning_rate=LEARNING_RATE)

trajectory = {
    "states": [],
    "actions": [],
    "rewards": [],
    "pdfs": []
}

predictions = {
    "mean": [],
    "std": []
}

episode_loss = []
episode_score = []

torch.autograd.set_detect_anomaly(True)

for _ in tqdm(range(NUM_EPISODES)):
    done = False
    state = env.reset()
    trajectory = {key: [] for key in trajectory} # empty all the lists in trajectory before new episode
    score = 0
    while not done:
        state = torch.tensor(state, dtype=torch.float32)
        mean, std = policy(state)
        action, pdf = get_action(mean, std) # pdf is the probability density function

        next_state, reward, done, _ = env.step([action.item()])

        trajectory["states"].append(state)
        trajectory["actions"].append(action)
        trajectory["rewards"].append(reward)
        trajectory["pdfs"].append(pdf)

        predictions["mean"].append(mean.detach().numpy())
        predictions["std"].append(std.detach().numpy())
        score += reward
        state = next_state

    episode_score.append(score)
    update_policy(policy, trajectory)

env.close()

detached_loss = [s.detach().numpy() for s in episode_loss]

plt.figure(1)
plt.plot(detached_loss)
plt.title("Loss over episodes")
plt.ylabel("Loss")
plt.xlabel("Episodes")
plt.grid(True)

plt.figure(2)
plt.plot(episode_score)
plt.title("Rewards over episodes")
plt.ylabel("Rewards")
plt.xlabel("Episodes")
plt.grid(True)

plt.figure(3)
plt.subplot(2,1,1)
plt.plot(predictions["mean"])
plt.ylabel("MEAN")
plt.grid(True)

plt.subplot(2,1,2)
plt.plot(predictions["std"])
plt.ylabel("Standard deviation")
plt.xlabel("Steps")
plt.grid(True)
plt.show()