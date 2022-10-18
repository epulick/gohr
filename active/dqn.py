# Adapted from work by Shubham Bharti and Yiding Chen

import pandas as pd
import numpy as np
import copy
import random, math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.distributions import Categorical

from collections import namedtuple, deque

np.set_printoptions(precision=2)

verbose = 0

class QNet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(QNet,self).__init__()
        self.network = nn.Linear(in_dim,out_dim)

    def forward(self,X):
        return self.network(X)

class ReplayMemory(object):
    def __init__(self, capacity, Transition):
        self.memory = deque([], maxlen=capacity)
        self.transitions = Transition

    def push(self, *args):
        # Append a transition to the queue
        self.memory.append(self.transitions(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN():
    def __init__(self,env,args):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.env = env
        self.in_dim, self.out_dim = self.env.in_dim, self.env.out_dim
        self.net = QNet(self.in_dim,self.out_dim).to(self.device)
        self.target_net = copy.deepcopy(self.net)
        self.epsilon = args["EPS_START"]
        self.train_horizon = args['TRAIN_HORIZON']
        self.n_episodes = args['TRAIN_EPISODES']

        # Set up dataframe for recording the results
        # to do - consider adding return code
        self.all_data_df = pd.DataFrame(columns=['episode', 'time', 'action_type', 'action', 'reward', 'done', 'other'])

        # Set up replay memory
        self.transitions = namedtuple('Transition', ('state','action','next_state','reward'))
        self.replay_memory = ReplayMemory(args['REPLAY_BUFFER_SIZE'], self.transitions)

    def train(self):
        for episode in range(self.n_episodes):
            done = 0
            episode_reward = 0
            state = self.env.reset()
            
            # --------------
            # FOR REFERENCE
            # --------------
            # print("torch.from_numpy(state): ", torch.from_numpy(state))
            # # Creates a tensor

            # print("torch.from_numpy(state).float(): ", torch.from_numpy(state).float())
            # # Converts to float (not positive why this is needed yet)

            # print("torch.from_numpy(state).float().unsqueeze(0): ", torch.from_numpy(state).float().unsqueeze(0))
            # # Adds a dimension at dim 0

            # print("torch.from_numpy(state).float().size: ", torch.from_numpy(state).float().size())
            # # Check dimension of original tensor (should just be [input space])

            # print("torch.from_numpy(state).float().unsqueeze(0).size: ", torch.from_numpy(state).float().unsqueeze(0).size())
            # # Check dimension of new tensor (now should be [1, input space])
            
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            for t in range(self.train_horizon):

                # Step the environment forward with an action
                action, action_type = self.select_action(state)
                next_state, reward, done, _ = self.env.step(int(action))

                # Process next_state into a tensor
                next_state = torch.from_numpy(next_state).float().unsqueeze(0).to(self.device)

                episode_reward+=reward

                current_df = pd.DataFrame({'episode':episode, 'time':t, 'action_type':action_type, 'action':int(action), 'reward':int(reward), 'done':done, 'other':'none'},index=[0])
                self.all_data_df=pd.concat([self.all_data_df, current_df],ignore_index=True)
                #self.all_data_df = self.all_data_df.append({'episode':episode, 'time':t, 'action_type':action_type, 'action':int(action), 'reward':int(reward), 'done':done, 'other':'none'}, ignore_index=True)
                
                # TO DO - double check on behavior at end of episode
                state=next_state

                # Truncate episode early if the board is clear
                if done:
                    break

            #print(str(episode), episode_reward)
            # Reset the episode reward
            episode_reward=0

    def select_action(self,state):
        if np.random.rand()<self.epsilon: 
            # Exploration action
            action = self.env.action_space.sample()
            action_type = 'random'
        else:
            # Greedy action
            with torch.no_grad():
                action = self.net(state).max(1)[1].to(self.device)

                # --------------
                # FOR REFERENCE
                # --------------
                # print("self.net(state): ",self.net(state))
                # # returns tensor of q values

                # print("self.net(state).max(1): ",self.net(state).max(1))
                # # returns max q value and corresponding index

                # print("self.net(state).max(1)[1]: ",self.net(state).max(1)[1])
                # # returns the index (i.e. the relevant action)
                
            action_type = 'greedy'
        return action, action_type
        