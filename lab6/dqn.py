'''DLP DQN Lab'''
__author__ = 'chengscott'
__copyright__ = 'Copyright 2020, NCTU CGI Lab'
import argparse 
from collections import deque
import itertools
import random
import time

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as op
from torch.utils.tensorboard import SummaryWriter


class ReplayMemory:
    __slots__ = ['buffer']

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, *transition):
        # (state, action, reward, next_state, done)
        self.buffer.append(tuple(map(tuple, transition)))

    def sample(self, batch_size, device):
        '''sample a batch of transition tensors'''
        transitions = random.sample(self.buffer, batch_size)
        return (torch.tensor(x, dtype=torch.float, device=device)
                for x in zip(*transitions))


class Net(nn.Module):
    def __init__(self, state_dim=8, action_dim=4, hidden_dim=(256, 512)):
        super().__init__()
        ## TODO ##
        # raise NotImplementedError
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.first_layer = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim[0]),
            nn.ReLU(inplace = True)
        )
        self.second_layer = nn.Sequential(
            nn.Linear(self.hidden_dim[0], self.hidden_dim[1]),
            nn.ReLU(inplace = True)
        )
        self.third_layer = nn.Linear(self.hidden_dim[1], self.action_dim)

    def forward(self, x):
        ## TODO ##
        # raise NotImplementedError
        y = self.first_layer(x)
        z = self.second_layer(y)
        return self.third_layer(z)


class DQN:
    def __init__(self, args):
        self._behavior_net = Net().to(args.device)
        self._target_net = Net().to(args.device)
        # initialize target network
        self._target_net.load_state_dict(self._behavior_net.state_dict())
        ## TODO ##
        self._optimizer = op.Adam(self._behavior_net.parameters(), lr=args.lr)
        # raise NotImplementedError
        # memory
        self._memory = ReplayMemory(capacity=args.capacity)

        ## config ##
        self.device = args.device
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.freq = args.freq
        self.target_freq = args.target_freq

    def select_action(self, state, epsilon, action_space):
        '''epsilon-greedy based on behavior network'''
        # print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n')
        #  ## TODO ##
        # raise NotImplementedError
        if random.random() <= epsilon:
            ## Exploration
            action = action_space.sample()
        else:
            ## Exploitation
            with torch.no_grad():
                input_state = torch.from_numpy(state).to(self.device)
                actions_vec = self._behavior_net(input_state)
                action = torch.argmax(actions_vec).item()
        return action

    def append(self, state, action, reward, next_state, done):
        self._memory.append(state, [action], [reward / 10], next_state,
                            [int(done)])

    def update(self, total_steps):
        if total_steps % self.freq == 0:
            self._update_behavior_network(self.gamma)
        if total_steps % self.target_freq == 0:
            self._update_target_network()

    def _update_behavior_network(self, gamma):
        # sample a minibatch of transitions
        state, action, reward, next_state, done = self._memory.sample(
            self.batch_size, self.device)

        ## TODO ##
        q_value = torch.gather(self._behavior_net(state), dim=1, index=action.long())
        with torch.no_grad():
            q_next = torch.max(self._target_net(next_state), dim=1)[0].unsqueeze(dim=1)
            q_target = reward + gamma * q_next * (1 - done)
        criterion = nn.MSELoss()
        loss = criterion(q_value, q_target)
        # raise NotImplementedError
        # optimize
        self._optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self._behavior_net.parameters(), 5)
        self._optimizer.step()

    def _update_target_network(self):
        '''update target network by copying from behavior network'''
        ## TODO ##
        self._target_net.load_state_dict(self._behavior_net.state_dict())

    def save(self, model_path, checkpoint=False):
        if checkpoint:
            torch.save(
                {
                    'behavior_net': self._behavior_net.state_dict(),
                    'target_net': self._target_net.state_dict(),
                    'optimizer': self._optimizer.state_dict(),
                }, model_path)
        else:
            torch.save({
                'behavior_net': self._behavior_net.state_dict(),
            }, model_path)

    def load(self, model_path, checkpoint=False):
        model = torch.load(model_path)
        self._behavior_net.load_state_dict(model['behavior_net'])
        if checkpoint:
            self._target_net.load_state_dict(model['target_net'])
            self._optimizer.load_state_dict(model['optimizer'])


def train(args, env, agent, writer):
    print('Start Training')
    action_space = env.action_space
    total_steps, epsilon = 0, 1.
    ewma_reward = 0
    for episode in range(args.episode):
        total_reward = 0
        state = env.reset()
        for t in itertools.count(start=1):
            # select action
            if total_steps < args.warmup:
                action = action_space.sample()
                # print('aaaaaaaaaaaaaaaaaaaaaaa\n')
            else:
                action = agent.select_action(state, epsilon, action_space)
                epsilon = max(epsilon * args.eps_decay, args.eps_min)
            # execute action
            next_state, reward, done, _ = env.step(action)
            # store transition
            agent.append(state, action, reward, next_state, done)
            if total_steps >= args.warmup:
                # print('ccccccccccccccccccccccc\n')
                agent.update(total_steps)

            state = next_state
            total_reward += reward
            total_steps += 1
            if done:
                ewma_reward = 0.05 * total_reward + (1 - 0.05) * ewma_reward
                writer.add_scalar('Train/Episode Reward', total_reward,
                                  total_steps)
                writer.add_scalar('Train/Ewma Reward', ewma_reward,
                                  total_steps)
                print(
                    'Step: {}\tEpisode: {}\tLength: {:3d}\tTotal reward: {:.2f}\tEwma reward: {:.2f}\tEpsilon: {:.3f}'
                    .format(total_steps, episode, t, total_reward, ewma_reward,
                            epsilon))
                break
    env.close()


def test(args, env, agent, writer):
    print('Start Testing')
    action_space = env.action_space
    epsilon = 0.001
    seeds = (args.seed + i for i in range(10))
    rewards = []
    for n_episode, seed in enumerate(seeds):
        total_reward = 0
        env.seed(seed)
        state = env.reset()
        ## TODO ##
        # ...
        for t in itertools.count(start=1):
            if args.render:
                env.render()

            ## Select action
            action = agent.select_action(state, epsilon, action_space)
            
            ## Execute action
            next_state, reward, done, _= env.step(action)

            state = next_state
            total_reward += reward
            if done:
                writer.add_scalar("Test/Episode Reward", total_reward, n_episode)
                print("Episode: {}\tTotal reward: {:.2f}".format(n_episode, total_reward))
                rewards.append(total_reward)
                break
    print('Average Reward', np.mean(rewards))
    env.close()


def main():
    ## arguments ##
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-d', '--device', default='cuda')
    parser.add_argument('-m', '--model', default='dqn.pth')
    parser.add_argument('--logdir', default='log/dqn')
    # train
    parser.add_argument('--warmup', default=10000, type=int)
    parser.add_argument('--episode', default=1200, type=int)
    parser.add_argument('--capacity', default=10000, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=.0005, type=float)
    parser.add_argument('--eps_decay', default=.995, type=float)
    parser.add_argument('--eps_min', default=.01, type=float)
    parser.add_argument('--gamma', default=.99, type=float)
    parser.add_argument('--freq', default=4, type=int)
    parser.add_argument('--target_freq', default=100, type=int)
    # test
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--seed', default=20200519, type=int)
    parser.add_argument('--  ', default=.001, type=float)
    args = parser.parse_args()

    ## main ##
    env = gym.make('LunarLander-v2')
    agent = DQN(args)
    writer = SummaryWriter(args.logdir)
    if not args.test_only:
        train(args, env, agent, writer)
        agent.save(args.model)
    agent.load(args.model)
    test(args, env, agent, writer)


if __name__ == '__main__':
    main()
