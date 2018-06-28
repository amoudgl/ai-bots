# simulated annealing algorithm 
# generate a random configuration of the parameters, add small amount of noise to the parameters and evaluate them
# if new params are better than the old ones, keep the new ones
# else keep the new params with some probability and decrease this probability with time
import gym
from gym import wrappers
import numpy as np
from math import exp

def eps_rew(env, observation, params):
    t = 0
    net_reward = 0
    while True:
        if (np.inner(observation, params) < 0):
            action = 0
        else:
            action = 1
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode %d finished after %d timesteps, reward = %d"%(episode, t + 1, net_reward + 1))
            break
        net_reward += reward
        t += 1
    return net_reward

env = gym.make('CartPole-v0')
max_reward = 0
gamma = 0.3
params = 2 * np.random.rand(4) - 1
temp = 1
temp_min = 0.001
temp_decay = 0.9
REWARD_LIMIT = 200

for episode in range(10000):

    # generate new random parameters
    random_noise = 2 * np.random.rand(4) - 1
    noisy_params = params + gamma * random_noise

    # get episode reward
    observation = env.reset()
    net_reward = eps_rew(env, observation, noisy_params)

    # simulated annealing step
    anneal = exp((net_reward-max_reward)/(REWARD_LIMIT*temp))
    if anneal > np.random.rand():
        max_reward = net_reward
        params = noisy_params
    temp = max(temp_min, temp_decay*temp)