import numpy as np
import gymnasium as gym
import torch
import matplotlib.pyplot as plt

from tqdm import tqdm
from copy import deepcopy
from policy_nn import PolicyNet
from torch.distributions import Categorical
from gymnasium.wrappers import RecordVideo, RecordEpisodeStatistics

def custom_loss(reward, p_log_prob, b_prob):
    """"action is action sampled from behavioral policy and we compute probability of picking this action in the target policy"""
    ro = p_log_prob.exp()/b_prob
    loss = -p_log_prob*reward.item()*ro.item()
    return loss

def get_action(action_probs): #the actions are picked from gaussian distribution of actions
    pdf = Categorical(action_probs) # creates probability density function
    action = pdf.sample() # samples from
    return action, pdf

def cummulative_reward(rewards, gamma):
    R = torch.zeros(len(rewards), dtype=torch.float32, requires_grad=False)
    running_sum = 0.0
    for t in reversed(range(len(rewards))):
        running_sum = rewards[t] + gamma*running_sum
        R[t] = running_sum

    R = (R-R.mean())/R.std() # this keeps rewards in similar scale for all episodes, so the model does not get confused.
    return R

def update_policy(policy_net, b_trajectory, p_log_probs):
    episode_length = len(b_trajectory["states"])
    loss = []
    cumm_rewards = cummulative_reward(b_trajectory["rewards"], gamma=0.99)

    for step in range(episode_length):
        loss.append(custom_loss(cumm_rewards[step], p_log_probs[step], b_trajectory["probs"][step]))

    total_loss = sum(loss)
    
    # Zero gradients, perform backward pass, and update policy
    policy_net.optimizer.zero_grad()
    total_loss.backward()
    policy_net.optimizer.step()

def sample_from_behavior_policy(action_space):
    """Samples action from a uniform distribution of actions - the behavior policy"""
    probs = torch.tensor(np.full(action_space, 1/action_space))
    pdf = Categorical(probs)
    action = pdf.sample()
    probability = pdf.log_prob(action).exp()
    return action, probability

# ------------MAIN------------:
NUM_EPISODES = 1001
LEARNING_RATE = .001
VIDEO_PERIOD = 100

env_training = gym.make("CartPole-v1", render_mode="rgb_array")  # Behavioral policy
env_evaluation = gym.make("CartPole-v1", render_mode="rgb_array")  # Evaluation policy

env_evaluation = RecordVideo(env_evaluation, video_folder="cartpole-agent", name_prefix="training",
                  episode_trigger=lambda x: x % VIDEO_PERIOD == 0)
env_training = RecordEpisodeStatistics(env_training)
env_evaluation = RecordEpisodeStatistics(env_evaluation)

policy = PolicyNet(learning_rate=LEARNING_RATE)
 
b_trajectory = {
    "states": [],
    "actions": [],
    "rewards": [],
    "probs": []
}

b_epi_stats = {
    "time": [],
    "total_reward": [],
    "length": []
}

p_epi_stats = {
    "time": [],
    "total_reward": [],
    "length": []
}

torch.autograd.set_detect_anomaly(True)

for _ in tqdm(range(NUM_EPISODES)):
    done = False
    b_state, _ = env_training.reset()
    p_state, _ = env_evaluation.reset()

    b_trajectory = {key: [] for key in b_trajectory} # empty all the lists in trajectory before new episode
    p_log_probs = []

    while not done:
        b_state = torch.tensor(b_state, dtype=torch.float32)
        b_action, b_prob = sample_from_behavior_policy(env_training.action_space.n)

        # get the target policy probability of picking this action, picked by behavioral policy:
        p_action_probs = policy(b_state)
        _, p_pdf = get_action(p_action_probs)
        p_log_prob = p_pdf.log_prob(b_action)
        p_log_probs.append(p_log_prob)
        
        b_next_state, b_reward, b_terminated, b_truncated, b_info = env_training.step(b_action.item())

        b_trajectory["states"].append(b_state)
        b_trajectory["actions"].append(b_action)
        b_trajectory["rewards"].append(b_reward)
        b_trajectory["probs"].append(b_prob)

        b_state = b_next_state
        done = b_terminated or b_truncated

    done = False
    while not done:
        p_state = torch.tensor(p_state, dtype=torch.float32)
        p_action_probs = policy(p_state)
        p_action, _ = get_action(p_action_probs)
        p_next_state, _, p_terminated, p_truncated, p_info = env_evaluation.step(p_action.item())

        p_state = p_next_state
        done = p_terminated or p_truncated
    
    # log episode statistics from both behavioral and actual policy:
    b_epi_stats["length"].append(b_info["episode"]["l"])
    b_epi_stats["time"].append(b_info["episode"]["t"])
    b_epi_stats["total_reward"].append(b_info["episode"]["r"])
    p_epi_stats["length"].append(p_info["episode"]["l"])
    p_epi_stats["time"].append(p_info["episode"]["t"])
    p_epi_stats["total_reward"].append(p_info["episode"]["r"])
    
    update_policy(policy, b_trajectory, p_log_probs)

env_training.close()
env_evaluation.close()

plt.figure(1)
plt.plot(b_epi_stats["total_reward"], label="Behavioral policy")
plt.plot(p_epi_stats["total_reward"], label="Target policy")
plt.title("Training statistics")
plt.ylabel("Total rewards")
plt.xlabel("Episode number")
plt.legend()
plt.grid(True)

plt.show()