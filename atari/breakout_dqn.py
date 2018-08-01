'''
PyTorch Deep Q-learning Implementation with Experience Replay for Atari
Default hyperparams are chosen from the paper: https://arxiv.org/abs/1312.5602

Usage [python3 recommended]:

For testing model from trained checkpoint, do:

python3 breakout_dqn.py -ckpt /path/to/checkpoint.pth.tar --test

For training Q-agent with default hyperparams, do:

python3 breakout_dqn.py

This code uses GPU by default, if your system has one otherwise it uses CPU.
To avoid using any GPU, add '--no-gpu' argument as follows:

python3 breakout_dqn.py --no-gpu

'''

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
import time, datetime
from tensorboardX import SummaryWriter
import os, sys
from PIL import Image

# hyperparams as arguments
exp_name = 'breakout_dqn' # experiment name for saving model checkpoints and tensorboard plot
args = None
save_path = None
use_gpu = torch.cuda.is_available()
writer = SummaryWriter('runs/' + exp_name)
parser = argparse.ArgumentParser(description='Atari DQN')
parser.add_argument('--no-gpu', help='do not use gpu for training', action='store_true')
parser.add_argument('-n', '--train-episodes', default=100000, type=int, help='number of training episodes')
parser.add_argument('-gamma', '--discount-factor', default=0.95, type=float, help='discount factor')
parser.add_argument('-e', '--epsilon', default=1.0, type=float, help='initial epsilon greedy factor')
parser.add_argument('-lr', '--learning-rate', default=0.0002, type=float, help='learning rate')
parser.add_argument('-b', '--batch-size', default=32, type=int, help='batch size for experience replay')
parser.add_argument('-rss', '--replay-start-size', default=100, type=int, help='populate replay buffer for these many steps with random actions')
parser.add_argument('-size', '--replay-buffer-size', default=1000000, type=int, help='size of replay buffer')
parser.add_argument('-f', '--save-frequency', default=50000, type=int, help='save model checkpoint every these many steps')
parser.add_argument('--test', help='test on breakout game using trained checkpoint', action='store_true')
parser.add_argument('-ckpt', '--test-checkpoint', default='', type=str, help='path to trained checkpoint')

# deep q-network
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=8, stride=4, padding=0),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0),
            nn.ReLU())
        self.fc = nn.Sequential(
                nn.Linear(9*9*32, 256),
                nn.ReLU(True),
                nn.Linear(256, 3)
            )

    def forward(self, X):
        o = self.layer1(X)
        o = self.layer2(o)
        o = o.reshape(o.size(0), -1)
        o = self.fc(o)
        return o

# takes (210,160,3) input and returns (84,84) image
def preprocess(I):
    I = Image.fromarray(np.uint8(I))
    I = I.resize((84,110))
    I = np.array(I)
    I = np.mean(I, axis=2).astype(np.uint8)
    I = I[26:,:]
    I = I[:,:,None]
    return I

# returns +1/-1 reward
def clip_reward(reward):
    return np.sign(reward)

# schedules exploration rate
def decay(f, t):
    if (t >= int(1e6)):
        return 0.1
    else:
        return f(t)

# converts a python object to torch variable
def convert_to_variable(x, grad=True):
    x = x/255.
    x = x.transpose((2,0,1))
    x = x[None,:,:,:]
    if use_gpu:
        return Variable(torch.FloatTensor(x).cuda(), requires_grad=grad)
    return Variable(torch.FloatTensor(x), requires_grad=grad)

# returns max value from a torch variable
def torch_max(var):
    if use_gpu:
        return torch.max(var.data.cpu())
    return torch.max(var.data)

# pick an action using epsilon greedy policy
def act(net, history, epsilon, global_t):
    if global_t <= args.replay_start_size or np.random.rand() <= epsilon:
        return np.random.randint(0, 3)
    else:
        history = convert_to_variable(history)
        q_values = net(history)
        if use_gpu:
            q_values = q_values.data.cpu().numpy()
        else:
            q_values = q_values.data.numpy()
        return np.argmax(q_values)

# for debugging
def print_gradients(net):
    for i in net.parameters():
        print(i.grad)

# experience replay implementation
# pick a random batch from memory and update the network with Q-learning
def exp_replay(net, memory, loss, optimizer, batch_size, global_t):
    minibatch = random.sample(memory, batch_size)
    batch_qvals = torch.Tensor()
    batch_targets = torch.Tensor()
    if use_gpu:
        batch_qvals = torch.Tensor().cuda()
        batch_targets = torch.Tensor().cuda()

    for history, a, rew, next_history, done in minibatch:
        history = convert_to_variable(history)
        q_values = net(history)
        targetq = torch.Tensor()
        if use_gpu:
            targetq = targetq.cuda()
        targetq = q_values.data.clone()
        if not done:
            next_history = convert_to_variable(next_history)
            next_q_values = net(next_history)
            maxq = torch_max(next_q_values)
            targetq[0][a] = rew + args.discount_factor * maxq
        else:
            targetq[0][a] = rew
        batch_qvals = torch.cat((batch_qvals, q_values))
        batch_targets = torch.cat((batch_targets, targetq))
    optimizer.zero_grad()
    l = loss(batch_qvals, batch_targets)
    l.backward()
    # print_gradients(net)
    optimizer.step()
    writer.add_scalar('perf/loss_vs_steps', l.item(), global_t+1)

# train q-agent
def train(net, env, loss, optimizer):
    global args
    epsilon = args.epsilon

    # fit a linear scheduler
    coff = np.polyfit([1,1000000],[epsilon,0.1],1)
    f = np.poly1d(coff)
    global_t = 0
    global_time = 0

    memory = deque(maxlen=args.replay_buffer_size)
    for episode in range(args.train_episodes):
        # reset environment for new
        start_time = time.time()
        s = env.reset()
        eps_rew = 0
        done = False
        t = 0
        s = preprocess(s)
        history = np.concatenate((s,s,s,s), axis=2) # copy states since we have no proceeding states

        # while episode is not finished
        while not done:
            # pick an action using epsilon greedy policy
            a = act(net, history, epsilon, global_t)

            # store state, next state, action, reward in memory
            next_s, rew, done, _ = env.step(a+1)
            rew = clip_reward(rew)
            next_s = preprocess(next_s)
            next_history = np.concatenate((next_s, history[:,:,:3]), axis=2)
            memory.append((history, a, rew, next_history, done))

            # move to next state
            s = next_s
            history = next_history
            t += 1
            global_t += 1
            eps_rew += rew

            # update the network with experience replay
            if len(memory) > args.replay_start_size:
                exp_replay(net, memory, loss, optimizer, args.batch_size, global_t)

            # decrease exploration with time
            epsilon = epsilon if global_t < args.replay_start_size else decay(f, global_t-args.replay_start_size)

            if global_t % args.save_frequency == 0:
                torch.save(net.state_dict(), save_path + '/checkpoint_' + str(global_t) + '.pth.tar')
            writer.add_scalar('data/epsilon_vs_steps', epsilon, global_t)
            writer.add_scalar('data/reward_vs_steps', rew, global_t)

        end_time = time.time()
        global_time += (end_time-start_time)

        # log everything
        print("[total steps = %d, time = %s] episode %d, steps = %d, reward = %d, time = %f seconds" % (global_t,
                                                                                            str(datetime.timedelta(seconds=int(global_time))),
                                                                                            episode+1,
                                                                                            t, eps_rew,
                                                                                            end_time-start_time))
        sys.stdout.flush()
        writer.add_scalar('perf/score_vs_episode', eps_rew, episode+1)
        writer.add_scalar('data/timesteps_vs_episode', t, episode+1)
    return net

# test trained q-agent
def test(net, env):
    test_episodes = 10
    eps = 0.1
    net.load_state_dict(torch.load(args.test_checkpoint))
    print('Loaded %s for testing' % (args.test_checkpoint))
    if use_gpu:
        net = net.cuda()
    for episode in range(test_episodes):
        s = env.reset()
        s = preprocess(s)
        score = 0
        done = False
        history = np.concatenate((s,s,s,s), axis=2)
        t = 0
        while not done:
            a = act(net, history, eps, 1000)
            env.render()
            time.sleep(0.01)
            next_s, rew, done, _ = env.step(a+1)
            next_s = preprocess(next_s)
            next_history = np.concatenate((next_s, history[:,:,:3]), axis=2)
            s = next_s
            history = next_history
            score += rew
            t += 1
        print("[test] episode %d, reward = %d steps = %d" % (episode+1, score, t))

def main():
    global args, save_path, use_gpu
    args = parser.parse_args()
    env = gym.make('BreakoutDeterministic-v4')
    net = DQN()

    if args.test:
        test(net, env)
    else:
        # initialization
        if args.no_gpu:
            use_gpu = False
        loss = torch.nn.MSELoss()
        optimizer = optim.RMSprop(net.parameters(), lr=args.learning_rate, alpha=0.99, eps=1e-6)
        save_path = '../data/' + exp_name
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            print('Created directory ' + save_path + ' to save model checkpoints')
        if use_gpu:
            net = net.cuda()
            loss = loss.cuda()

        # train q-agent
        net = train(net, env, loss, optimizer)
    env.close()
    return net

if __name__ == '__main__':
    net = main()
    if not args.test:
        torch.save(net.state_dict(), save_path + '/best_checkpoint.pth.tar')