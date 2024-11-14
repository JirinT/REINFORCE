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
    return R

def update_policy(policy_net, trajectory):
    trajectory_length = len(trajectory["states"])
    loss = []
    cumm_rewards = cummulative_reward(trajectory["rewards"], gamma=0.99)

    for step in range(trajectory_length):
        loss.append(custom_loss(trajectory["actions"][step], cumm_rewards[step], trajectory["pdfs"][step]))

    total_loss = sum(loss)
    episode_score.append(total_loss)

    # Zero gradients, perform backward pass, and update policy
    policy_net.optimizer.zero_grad()
    total_loss.backward()
    policy_net.optimizer.step()

# ------------MAIN------------:
NUM_EPISODES = 20
LEARNING_RATE = .001
VIDEO_PERIOD = 2

env = gym.make("MountainCarContinuous-v0", render_mode="rgb_array") # the states are x-axis position and velocity of the car
env = RecordVideo(env, video_folder="cartpole-agent", name_prefix="training",
                  episode_trigger=lambda x: x % VIDEO_PERIOD == 0)
env = RecordEpisodeStatistics(env)

policy = PolicyNet(learning_rate=LEARNING_RATE)
 
trajectory = {
    "states": [],
    "actions": [],
    "rewards": [],
    "pdfs": []
}

epi_stats = {
    "time" : [],
    "total_reward" : [],
    "length" :[]
}

predictions = {
    "mean": [],
    "std": []
}

episode_score = []

torch.autograd.set_detect_anomaly(True)

for _ in tqdm(range(NUM_EPISODES)):
    done = False
    state, _ = env.reset()
    trajectory = {key: [] for key in trajectory} # empty all the lists in trajectory before new episode
    while not done:
        state = torch.tensor(state, dtype=torch.float32)
        mean, std = policy(state)
        action, pdf = get_action(mean, std) # pdf is the probability density function

        next_state, reward, terminated, truncated, info = env.step([action.item()])

        trajectory["states"].append(state)
        trajectory["actions"].append(action)
        trajectory["rewards"].append(reward)
        trajectory["pdfs"].append(pdf)

        predictions["mean"].append(mean.detach().numpy())
        predictions["std"].append(std.detach().numpy())

        state = next_state
        done = terminated or truncated

    epi_stats["length"].append(info["episode"]["l"])
    epi_stats["time"].append(info["episode"]["t"])
    epi_stats["total_reward"].append(info["episode"]["r"])
    
    update_policy(policy, trajectory)

env.close()

detached_score = [s.detach().numpy() for s in episode_score]

plt.figure(1)
plt.plot(detached_score)
plt.title("Loss")
plt.grid(True)

plt.figure(2)
plt.plot(epi_stats["total_reward"])
plt.title("Training statistics")
plt.ylabel("Total rewards")
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