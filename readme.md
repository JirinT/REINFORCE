# REINFORCE - continuous
This project implemets the vanilla REINFORCE algorithm for continuous actions space. It uses the gaussian probability density function as the policy functions, so the actions are sampled from this pdf. The policy is parametrized by neural network.

I have implemented it with Mountain car continuous. The problem is that the rewards are $-.1*action^2$, meaning it penalizes the agent for high energy consumption. Since the agent does not achieve the flag, and so the reward of 200, and it updates the policy based on the rewards it gets, we get stuck in the local minimum of this problem and that is the car is stuck down in the valley and is not moving at all. This leads to zero rewards, which is always better than any negative reward ofc, but we do not get to the top by doing nothing.

So the vanilla reinforce algorithm can not work with the mountain car continuous problem.

# REINFORCE - discrete
The discrete reinforce works kinda well with the discrete cartpole environment, but is very noisy = has high variance, as can be seen in the plot of rewards over episodes.

# REINFORCE - discrete with baseline
The overall performance is much better than without the baseline, it has still variance but not that much. We can see that in the training history we achieved the max score many times and than forget and had to relearned.. = catastrophic forgetting.

# Actor critic algorithms
They all differ only in computing the target value - the reward
- **MC** - computes the the target as the cummulative reward from each time step to end of episode - accurate and works quiet well
- **TD(0)** - computes the target as the current reward plus discounted predicted value of the reward in the next state - $R + gammma*V(s_{t+1})$ - this approach is called bootstraping and updates the networks in each time step = online learning
- **TD(lambda)** - computes the target as the cummulative reward of lamba steps ahead, but it does so for all the possible lambdas in each time step and computing wighted average from it. It is very slow because it neeeds to compute every lambda in each iteration
- **TD(backward lambda)** - Also uses bootstraping as TD(0) and performs online update (=in every time step). It computes z-traces, which is parametrized by lambda = a compromise between frequency and recency of events. The z-traces are restarted for each episode and are updated in every time step, thanks to this the algorithm remembers what it learned and is less prone to catastrophic forgetting

# OFF policy:
- **Vanilla off policy policy gradients** - In an off-policy algorithm, the agent learns from a different policy than it is actually executing. The agent is executing the behavioral policy. In this case the behavioral policy is just uniform distribution of actions and we sample from this. The **target policy** is then updated based on the gained reward from the behavioral policy and based on ratio of $Probability of picking action a with target policy / probability of picking action a with behavioral policy$, where action a is sampled from behavioral policy. The final update of the target policy is: $theta= theta + R*ro*grad(logprob(action|target policy))$, where ro is the ratio and R are the rewards gained from behavioral policy and log prob is logaritmic probability of taking action a in target policy.