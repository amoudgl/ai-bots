# hill climbing algorithm 
# generate a random configuration of the parameters, add small amount of noise to the parameters and evaluate the new parameter configuration
# if new configuration is better than old one, discard the old one and accept the new one

# returns the net episode reward 
def get_episode_reward(env, observation, params):
    t = 0
    net_reward = 0
    while (t < 1000):
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

# imports and initializations
import gym
from gym import wrappers
import numpy as np
env = gym.make('CartPole-v0')
env = wrappers.Monitor(env, '/tmp/cartpole-hill-climbing', force=True)
max_reward = 0
gamma = 0.3
params = 2 * np.random.rand(4) - 1

for episode in range(10000):

    # get new random parameter tuple
    random_noise = 2 * np.random.rand(4) - 1
    noisy_params = params + gamma * random_noise

    # reset environment and check if noisy parameters performs better than current ones
    observation = env.reset()
    net_reward = get_episode_reward(env, observation, noisy_params)
    if (net_reward >= max_reward):
        max_reward = net_reward
        params = noisy_params
