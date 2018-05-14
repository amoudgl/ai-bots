''' PyTorch Deep Q-learning Implementation with Experience Replay '''

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
parser = argparse.ArgumentParser(description='Deep Q-learning')
parser.add_argument('-n', '--train-episodes', default=10000, type=int, help='number of training episodes')
parser.add_argument('-gamma', '--discount-factor', default=0.95, type=float, help='discount factor')
parser.add_argument('-e', '--epsilon', default=1.0, type=float, help='initial epsilon greedy factor')
parser.add_argument('-ed', '--epsilon-decay', default=0.995, type=float, help='epsilon decay factor')
parser.add_argument('-lr', '--learning-rate', default=0.001, type=float, help='learning rate')
parser.add_argument('-b', '--batch-size', default=50, type=int, help='batch size for experience replay')

# converts a python object to torch variable
def convert_to_variable(x, grad=True):
    return Variable(torch.FloatTensor(x), requires_grad=grad)

# returns max value from a torch variable
def torch_max(var):
    return torch.max(var.data)

# pick an action using epsilon greedy policy
def epsilon_greedy(q_values, epsilon):
    q_values = q_values.data.numpy()
    if np.random.rand() <= epsilon:
        return np.random.randint(0, 2)
    return np.argmax(q_values)

# xavier uniform weight initialization
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

# experience replay implementation
# pick a random batch from memory and update the network with Q-learning
# q-learning magic happens right here
def exp_replay(net, memory, loss, optimizer, batch_size):
    minibatch = random.sample(memory, batch_size)
    for s, a, rew, next_s, done in minibatch:
        q_values = net(convert_to_variable(s))
        targetq = torch.Tensor()
        targetq = q_values.data.clone()
        if not done:
            next_q_values = net(convert_to_variable(next_s))
            maxq = torch_max(next_q_values)
            targetq[a] = rew + args.discount_factor * maxq
        else:
            targetq[a] = rew
        optimizer.zero_grad()
        l = loss(q_values, targetq)
        l.backward()
        optimizer.step()

# simple q-network
class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        self.main = nn.Sequential(
                nn.Linear(4, 30),
                nn.ReLU(True),
                nn.Linear(30, 25),
                nn.ReLU(True),
                nn.Linear(25, 2)
            )

    def forward(self, X):
        o = self.main(X)
        return o

# train q-agent
def train(net, env, loss, optimizer):
    global args
    epsilon = args.epsilon
    memory = deque(maxlen=2000)
    for episode in range(args.train_episodes):
        # reset environment for new episode
        s = env.reset()
        done = False
        t = 0

        # while episode is not finished
        while not done:

            # pick an action using epsilon greedy policy
            q_values = net(convert_to_variable(s))
            a = epsilon_greedy(q_values, epsilon)

            # store state, next state, action, reward in memory
            next_s, rew, done, _ = env.step(a)
            rew = rew if not done else -10
            memory.append((s, a, rew, next_s, done))

            # move to next state
            s = next_s
            t = t + 1

        # when episode is finished, update the network with experience replay
        if len(memory) > args.batch_size:
            exp_replay(net, memory, loss, optimizer, args.batch_size)

            # decrease exploration with time
            if epsilon > 0.01: # keep 0.01 as minimum epsilon value
                epsilon = epsilon * args.epsilon_decay
        print("Episode %d finished after %d timesteps, epsilon = %f" % (episode+1, t, epsilon))

    return net

def main():
    global args
    args = parser.parse_args()

    # initialization
    env = gym.make('CartPole-v0')
    net = QNet()
    net.apply(init_weights)
    loss = torch.nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)

    # train q-agent
    net = train(net, env, loss, optimizer)
    return net

if __name__ == '__main__':
    net = main()
    torch.save(net.state_dict(), 'checkpoint.pth.tar')
