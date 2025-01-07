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
    
def custom_loss(action, reward, pdf, state_value, next_state_value, gamma):
    delta = reward + gamma*next_state_value - state_value # thanks to this we can do online update, because we dont need the whole cummulative reward, instead we take reward plus its prediction in the next state 
    log_prob = pdf.log_prob(action) # log_prob(action) = log(gaussian(action, mean, std))
    loss = -log_prob*delta
    return loss

def get_action(action_probs): #the actions are picked from gaussian distribution of actions
    pdf = Categorical(action_probs)
    action = pdf.sample()
    return action, pdf

def update_policy(policy_net, reward, state_value, next_state_value, action, pdf, I):
    loss = custom_loss(action, reward, pdf, state_value, next_state_value, GAMMA)
    loss *= I

    # Zero gradients, perform backward pass, and update policy
    policy_net.optimizer.zero_grad()
    loss.backward(retain_graph=True)
    policy_net.optimizer.step()

def update_value(value_net, reward, state_value, next_state_value, I):
    target = reward+GAMMA*next_state_value
    prediction = state_value
    loss = F.mse_loss(prediction, target)
    loss *= I

    value_net.optimizer.zero_grad()
    loss.backward()
    value_net.optimizer.step()

# ------------MAIN------------:
NUM_EPISODES = 500
LEARNING_RATE = .001
VIDEO_PERIOD = 100
GAMMA = 0.98

env = gym.make("CartPole-v1", render_mode="rgb_array") # the states are x-axis position and velocity of the car
# env = RecordVideo(env, video_folder="cartpole-agent", name_prefix="training",
                #   episode_trigger=lambda x: x % VIDEO_PERIOD == 0)
env = RecordEpisodeStatistics(env)

policy = PolicyNet(learning_rate=LEARNING_RATE)
value = ValueNet(learning_rate=LEARNING_RATE)

epi_stats = {
    "time" : [],
    "total_reward" : [],
    "length" : []
}

torch.autograd.set_detect_anomaly(True)

for _ in tqdm(range(NUM_EPISODES)):
    done = False
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32)
    I = 1
    while not done:    
        action_probs = policy(state) # its a list of probabilities, sum(action_probs)=1
        state_value = value(state)

        action, pdf = get_action(action_probs) # pdf is the probability density function

        next_state, reward, terminated, truncated, info = env.step(action.item())
        done = terminated or truncated
        
        next_state = torch.tensor(next_state, dtype=torch.float32)
        next_state_value = value(next_state)

        if terminated:
            next_state_value = torch.tensor([0.0], dtype=torch.float32)

        update_policy(policy, reward, state_value, next_state_value, action, pdf, I)
        update_value(value, reward, state_value, next_state_value, I)
        
        I *= GAMMA
        state = next_state

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