# random guessing algorithm 
# generate 10000 random configurations of the model's parameters and pick the one that achieves the best cumulative reward. 

import gym
from gym import wrappers
import numpy as np

env = gym.make('CartPole-v0')
env = wrappers.Monitor(env, '/tmp/cartpole-random-guessing', force=True)
max_reward = 0
optimal_params = np.zeros(4)

for episode in range(1000):
    observation = env.reset()
    params = 2 * np.random.rand(4) - 1
    net_reward = 0
    t = 0
    while(1):
        env.render()
        if (np.inner(observation, params) < 0):
            action = 0
        else:
            action = 1
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode %d finished after %d timesteps, reward = %d"%(episode, t + 1, net_reward + 1))
            if (net_reward > max_reward):
                max_reward = net_reward
                optimal_params = params
            break
        net_reward += reward
        t = t + 1
