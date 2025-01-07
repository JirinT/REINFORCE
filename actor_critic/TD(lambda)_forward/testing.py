import numpy as np

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


rewards = np.random.randint(0, 10, size=5)
values = np.random.randint(0, 10, size=5)
gamma_factor = .99
lambda_factor = .5

lambda_rews = lambda_reward(rewards, values, gamma_factor, lambda_factor)
print(lambda_rews)