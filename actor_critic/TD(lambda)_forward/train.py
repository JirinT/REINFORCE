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
    
def custom_loss(action, lam_reward, pdf, state_value):
    delta = lam_reward-state_value
    log_prob = pdf.log_prob(action) # log_prob(action) = log(gaussian(action, mean, std))
    loss = -log_prob*delta
    return loss

def get_action(action_probs): #the actions are picked from gaussian distribution of actions
    pdf = Categorical(action_probs)
    action = pdf.sample()
    return action, pdf

def lambda_reward(rewards, values, gamma_factor, lambda_factor):
    
    rew_len = len(rewards)
    lam_rewards = np.zeros(rew_len)
    
    for t in range(rew_len):
        # For every time step in trajectory
        lambda_return = 0
        weighting = 1
        for n in range(t, rew_len):
            # for every n-step cummulative return in current time step t
            Gt_tn = sum(gamma_factor**(i-t)*rewards[i] for i in range(t, n+1)) # now n is maximal rew_len-1 so n+1 is rew_len
            if n != rew_len-1:
                Gt_tn += (gamma_factor**(n+1-t)) * values[n+1]

            lambda_return += weighting * Gt_tn
            weighting *= lambda_factor

        lam_rewards[t] = (1-lambda_factor)*lambda_return
    
    return lam_rewards

def update_policy(policy_net, trajectory, lam_rewards, state_values):
    trajectory_length = len(trajectory["states"])
    loss = []

    for step in range(trajectory_length):
        loss.append(custom_loss(trajectory["actions"][step], lam_rewards[step], trajectory["pdfs"][step], state_values[step]))

    total_loss = sum(loss)

    # Zero gradients, perform backward pass, and update policy
    policy_net.optimizer.zero_grad()
    total_loss.backward()
    policy_net.optimizer.step()

def update_value(value_net, lam_rewards, state_values):
    loss = F.mse_loss(state_values, lam_rewards)

    value_net.optimizer.zero_grad()
    loss.backward()
    value_net.optimizer.step()

# ------------MAIN------------:
NUM_EPISODES = 500
LEARNING_RATE = .001
VIDEO_PERIOD = 100
GAMMA_FACTOR = .99
LAMBDA_FACTOR = .8

env = gym.make("CartPole-v1", render_mode="rgb_array") # the states are x-axis position and velocity of the car
# env = RecordVideo(env, video_folder="cartpole-agent", name_prefix="training",
                #   episode_trigger=lambda x: x % VIDEO_PERIOD == 0)
env = RecordEpisodeStatistics(env)

policy = PolicyNet(learning_rate=LEARNING_RATE)
value = ValueNet(learning_rate=LEARNING_RATE)

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

torch.autograd.set_detect_anomaly(True)

for _ in tqdm(range(NUM_EPISODES)):
    done = False
    state, _ = env.reset()
    trajectory = {key: [] for key in trajectory} # empty all the lists in trajectory before new episode
    state_values = []
    while not done:
        state = torch.tensor(state, dtype=torch.float32, requires_grad=False)
        
        action_probs = policy(state) # its a list of probabilities, sum(action_probs)=1
        state_value = value(state)

        action, pdf = get_action(action_probs) # pdf is the probability density function

        next_state, reward, terminated, truncated, info = env.step(action.item())

        trajectory["states"].append(state)
        trajectory["actions"].append(action)
        trajectory["rewards"].append(reward)
        trajectory["pdfs"].append(pdf)
        state_values.append(state_value)
        
        state = next_state
        done = terminated or truncated

    epi_stats["length"].append(info["episode"]["l"])
    epi_stats["time"].append(info["episode"]["t"])
    epi_stats["total_reward"].append(info["episode"]["r"])
    
    lam_rewards = lambda_reward(rewards=trajectory["rewards"], values=state_values, gamma_factor=GAMMA_FACTOR, lambda_factor=LAMBDA_FACTOR)
    lam_rewards = torch.tensor(lam_rewards, dtype=torch.float32)
    state_values = torch.stack(state_values).squeeze()

    update_value(value, lam_rewards, state_values)
    update_policy(policy, trajectory, lam_rewards, [v.detach() for v in state_values])

env.close()

plt.figure(1)
plt.plot(epi_stats["total_reward"])
plt.title("Training statistics")
plt.ylabel("Total rewards")
plt.xlabel("Episode number")
plt.grid(True)

plt.show()