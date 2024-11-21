# REINFORCE - continuous
This project implemets the vanilla REINFORCE algorithm for continuous actions space. It uses the gaussian probability density function as the policy functions, so the actions are sampled from this pdf. The policy is parametrized by neural network.

I have implemented it with Mountain car continuous. The problem is that the rewards are $-.1*action^2$, meaning it penalizes the agent for high energy consumption. Since the agent does not achieve the flag, and so the reward of 200, and it updates the policy based on the rewards it gets, we get stuck in the local minimum of this problem and that is the car is stuck down in the valley and is not moving at all. This leads to zero rewards, which is always better than any negative reward ofc, but we do not get to the top by doing nothing.

So the vanilla reinforce algorithm can not work with the mountain car continuous problem.

# REINFORCE - discrete
The discrete reinforce works kinda well with the discrete cartpole environment, but is very noisy = has high variance, as can be seen in the plot of rewards over episodes.

# REINFORCE - discrete with baseline
The overall performance is much better than without the baseline, it has still variance but not that much. We can see that in the training history we achieved the max score many times and than forget and had to relearned.. = catastrophic forgetting.