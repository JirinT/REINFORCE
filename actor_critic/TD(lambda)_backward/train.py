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
    
def custom_loss(action, pdf):
    log_prob = pdf.log_prob(action) # log_prob(action) = log(gaussian(action, mean, std))
    return log_prob

def get_action(action_probs): #the actions are picked from gaussian distribution of actions
    pdf = Categorical(action_probs)
    action = pdf.sample()
    return action, pdf

def update_policy(policy_net, action, pdf, z, delta, I):
    # update actor
    loss = custom_loss(action, pdf)
    policy_net.zero_grad() 
    loss.backward(retain_graph=True) # this computes gradients wrt the loss - logaritmic probability

    # now compute the z-trace for all parameters and save it in a list
    with torch.no_grad():
        if not z:
            # for the first step - initilaize z
            for p in policy_net.parameters():
                trace = I*p.grad # the gradients in p.grad are computed in the loss.backward - here we just pick them
                z.append(trace)
        else:
            # in next steps we update the z-traces for each parameter again - thats how the algorithm remembers previous steps
            for i, p in enumerate(policy_net.parameters()):
                z[i] = GAMMA*LAMBDA_POLICY*z[i]+I*p.grad

    # now update the network parameters with the z-traces and delta
    with torch.no_grad():
        for i, p in enumerate(policy_net.parameters()):
            updated_p = p + LEARNING_RATE_POLICY*delta*z[i]
            p.copy_(updated_p)
    return z

def update_value(value_net, state_value, z, delta, I):
    # update critic
    value_net.zero_grad()
    state_value.backward(retain_graph=True) # computes gradients in the net wrt state_value - the predicted value

    # now compute the z-trace for all parameters and save it in a list
    with torch.no_grad():
        if not z:
            # for the first step - initilaize z
            for p in value_net.parameters():
                trace = I*p.grad # the gradients in p.grad are computed in the state_value.backward - here we just pick them
                z.append(trace)
        else:
            # in next steps we update the z-traces for each parameter again - thats how the algorithm remembers previous steps
            for i, p in enumerate(value_net.parameters()):
                z[i] = GAMMA*LAMBDA_VALUE*z[i] + I*p.grad

    # now update the network parameters with the z-traces and delta
    with torch.no_grad():
        for i, p in enumerate(value_net.parameters()):
            updated_p = p + LEARNING_RATE_VALUE*delta*z[i]
            p.copy_(updated_p)
    return z

# ------------MAIN------------:
NUM_EPISODES = 500
VIDEO_PERIOD = 100
GAMMA = 0.99
LEARNING_RATE_POLICY = .001
LEARNING_RATE_VALUE = .001
LAMBDA_POLICY = 0.8
LAMBDA_VALUE = 0.8

env = gym.make("CartPole-v1", render_mode="rgb_array") # the states are x-axis position and velocity of the car
env = RecordEpisodeStatistics(env)

policy = PolicyNet()
value = ValueNet()

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
    z_policy = []
    z_value = []
    while not done:    
        action_probs = policy(state) # its a list of probabilities, sum(action_probs)=1
        action_probs = torch.clamp(action_probs, min=1e-8, max=1.0)
        state_value = value(state)

        action, pdf = get_action(action_probs) # pdf is the probability density function

        next_state, reward, terminated, truncated, info = env.step(action.item())
        done = terminated or truncated
        
        next_state = torch.tensor(next_state, dtype=torch.float32)
        next_state_value = value(next_state)

        if terminated:
            next_state_value = torch.tensor([0.0], dtype=torch.float32)

        delta = reward + GAMMA*next_state_value - state_value

        z_policy = update_policy(policy, action, pdf, z_policy, delta, I)
        z_value = update_value(value, state_value, z_value, delta, I)
        
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