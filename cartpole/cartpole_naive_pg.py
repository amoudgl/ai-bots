''' PyTorch Naive Policy Gradient Implementation '''

import numpy as np
import argparse
import copy
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import gym
from gym import wrappers
from collections import deque
import random

# hyperparams as arguments
args = None
parser = argparse.ArgumentParser(description='Naive Policy Gradient')
parser.add_argument('-n', '--train-episodes', default=10000, type=int, help='number of training episodes')
parser.add_argument('-gamma', '--discount-factor', default=0.95, type=float, help='discount factor')
parser.add_argument('-e', '--epsilon', default=1.0, type=float, help='initial epsilon greedy factor')
parser.add_argument('-ed', '--epsilon-decay', default=0.995, type=float, help='epsilon decay factor')
parser.add_argument('-lr', '--learning-rate', default=1e-3, type=float, help='learning rate')

# converts a python object to torch variable
def convert_to_variable(x, grad=True):
    return Variable(torch.FloatTensor(x), requires_grad=grad)

# pick an action using epsilon greedy policy
def epsilon_greedy(q_values, epsilon):
    q_values = q_values.data.numpy()
    action = 0
    if np.random.rand() <= epsilon:
        action = np.random.randint(0, 2)
    else:
        action = np.argmax(q_values)
    return action

# xavier uniform weight initialization
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

# print gradients of network (for debugging)
def print_gradients(net):
    for param in net.parameters():
        print(param.grad)

# given rewards of an episode, get discounted returns
def get_discounted_returns(eps_rew):
    returns = []
    running_ret = 0
    for rew in reversed(eps_rew):
        ret = rew + args.discount_factor * running_ret
        returns.append(ret)
        running_ret = ret 
    return convert_to_variable(list(reversed(returns)), False)

# naive policy gradient update
def pg_update(net, optimizer, eps_rew, eps_probs, eps_targets):
    discounted_returns = get_discounted_returns(eps_rew)
    eps_targets = convert_to_variable(eps_targets, False)
    grads = discounted_returns[:,None] * (eps_probs - eps_targets)
    optimizer.zero_grad()
    eps_probs.backward(grads)
    optimizer.step()

# simple policy network
# returns probability of each action
class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
        self.main = nn.Sequential(
                nn.Linear(4, 30),
                nn.ReLU(True),
                nn.Linear(30, 25),
                nn.ReLU(True),
                nn.Linear(25, 2),
                nn.Softmax()
            )

    def forward(self, X):
        o = self.main(X)
        return o

# train agent
def train(net, env, optimizer):
    global args
    epsilon = args.epsilon
    memory = deque(maxlen=2000)
    for episode in range(args.train_episodes):
        # reset environment for new episode
        s = env.reset()
        done = False
        t = 0
        eps_rew = []
        eps_probs = None
        eps_targets = None

        # while episode is not finished
        while not done:

            # pick an action using epsilon greedy policy
            probs = net(convert_to_variable(s))
            probs = probs[None, :]
            a = epsilon_greedy(probs, epsilon)

            # get next state and reward from env
            # env.render()
            next_s, rew, done, _ = env.step(a)
            rew = rew if not done else -10

            # build target for policy gradient
            target = torch.zeros(probs.shape)
            target[0, a] = 1

            # store episode target, actions probabilities and targets
            eps_targets = torch.cat((eps_targets, target), 0) if eps_targets is not None else target
            eps_probs = torch.cat((eps_probs, probs), 0) if eps_probs is not None else probs
            eps_rew.append(rew)
            
            # move to next state
            s = next_s
            t = t + 1

        # update network after each episode with naive policy gradient
        pg_update(net, optimizer, eps_rew, eps_probs, eps_targets)

        # decrease exploration with time
        if epsilon > 0.01: # keep 0.01 as minimum epsilon value
            epsilon = epsilon * args.epsilon_decay
        print("Episode %d finished after %d timesteps, epsilon = %f" % (episode+1, t, epsilon))
    return net

def main():
    global args
    args = parser.parse_args()

    # initialization
    torch.manual_seed(2) # optional
    env = gym.make('CartPole-v0')
    net = PolicyNet()
    net.apply(init_weights)
    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)

    # train naive pg agent
    net = train(net, env, optimizer)
    return net

if __name__ == '__main__':
    net = main()
    torch.save(net.state_dict(), 'checkpoint.pth.tar')
