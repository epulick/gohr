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
    def __init__(self, in_dim, out_dim, hidden_sizes):
        super(QNet,self).__init__()
        activation = nn.ReLU
        sizes = [in_dim] + hidden_sizes + [out_dim]
        layers = []
        for i in range(len(sizes)-2):
            layers+=[nn.Linear(sizes[i], sizes[i+1]),activation()]
        layers += [nn.Linear(sizes[-2],sizes[-1]),nn.Identity()]
        self.network = nn.Sequential(*layers)

    def forward(self,X):
        return self.network(X)

class ReplayMemory(object):
    def __init__(self, capacity, Transition):
        self.memory = deque([], maxlen=capacity)
        self.transitions = Transition

    def push(self, *args):
        self.memory.append(self.transitions(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN():
    def __init__(self,env,args):
        # Put onto correct hardware
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Pull in the game environment
        self.env = env
        
        # Misc. parameter import
        self.eps_start, self.eps_end, self.eps_decay = args["EPS_START"], args['EPS_END'], args['EPS_DECAY']
        self.train_horizon = args['TRAIN_HORIZON']
        self.n_episodes = args['TRAIN_EPISODES']
        self.batch_size = args['GRAD_BATCH_SIZE']
        self.gamma = args['GAMMA']
        self.sync = args ['TARGET_UPDATE_FREQ']
        self.reward_map = args['REWARD_MAPPING']
        self.steps = 0

        # Build the neural network (net and target_net)
        self.in_dim, self.out_dim = self.env.in_dim, self.env.out_dim
        self.net = QNet(self.in_dim,self.out_dim,args['HIDDEN_SIZES']).to(self.device)
        self.target_net = copy.deepcopy(self.net)

        # TO-DO
        # Consider other optimizers via parameter input
        self.optimizer=torch.optim.Adam(self.net.parameters(),args['LR'])
        self.loss = nn.SmoothL1Loss()

        # Set up dataframe for recording the results
        # to do - consider adding return code (some pieces are more informative than others)
        self.all_data_df = pd.DataFrame(columns=['episode', 'time', 'action_type', 'action', 'reward', 'done','epsilon', 'other'])

        # Set up replay memory
        self.transitions = namedtuple('Transition', ('state','action','next_state','reward'))
        self.replay_memory = ReplayMemory(args['REPLAY_BUFFER_SIZE'], self.transitions)

    def reward_mapping(self, old_reward):
        if old_reward == -1:
            return 0
        else:
            return 1

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
            
            state = torch.from_numpy(state).float().to(self.device)
            for t in range(self.train_horizon):

                # Step the environment forward with an action
                action, action_type, eps = self.select_action(state)
                next_state, reward, done, _ = self.env.step(int(action))
                #if self.reward_map:
                #    reward = self.reward_mapping(reward)

                # Process next_state into a tensor
                next_state = torch.from_numpy(next_state).float().to(self.device)
                
                # Add step reward to episode reward
                episode_reward+=reward

                # Write current step's data to a dataframe and concat with the main dataframe
                current_df = pd.DataFrame({'episode':episode, 'time':t, 'action_type':action_type, 'action':int(action), 'reward':int(reward), 'done':done, 'epsilon':eps, 'other':'none'},index=[0])
                self.all_data_df=pd.concat([self.all_data_df, current_df],ignore_index=True)
                
                # Append this step to the replay buffer
                action, reward = torch.IntTensor([action]).to(self.device), torch.FloatTensor([reward]).to(self.device)
                self.replay_memory.push(state, action, next_state, reward)

                # TO DO - double check on behavior at end of episode
                state=next_state
                
                # Optimize the model
                self.learn()
                self.steps+=1

                # If at an update interval, copy policy net to target net
                if (self.steps % self.sync == 0):
                    print("Synchronized networks!")
                    print("Current step: ", self.steps)

                # Truncate episode early if the board is clear
                if done:
                    break

            # Reset the episode reward before the next iteration
            episode_reward=0



    def select_action(self,state):
        # epsilon greedy policy - epsilon decays exponentially with time
        eps_threshold = self.eps_end + (self.eps_start-self.eps_end)*math.exp(-1*self.steps/self.eps_decay)

        # Random exploration action
        if np.random.rand()<eps_threshold: 
            action = self.env.action_space.sample()
            action_type = 'random'
        # Greedy action per policy
        else:
            with torch.no_grad():
                action = self.net(state).max(0)[1].to(self.device)

                # --------------
                # FOR REFERENCE
                # --------------
                #print("self.net(state): ",self.net(state))
                # # returns tensor of q values

                #print("self.net(state).max(1): ",self.net(state).max(0))
                # # returns max q value and corresponding index

                #print("self.net(state).max(1)[1]: ",self.net(state).max(0)[1])
                # # returns the index (i.e. the relevant action)
                
            action_type = 'greedy'
        return action, action_type, eps_threshold
        
    def learn(self):
        
        # Can only optimize if the buffer filled above the batch threshold
        if len(self.replay_memory) < self.batch_size:
            return

        # Get a batch
        batch = self.replay_memory.sample(self.batch_size)
        
        state, action, next_state, reward = map(torch.stack, zip(*batch))
        # # Check the content/shapes
        # print("state: ", state)
        # print("state size: ", state.size())
        # print("action: ", action)
        # print("action size: ", action.size())
        # print("reward: ", reward)
        # print("reward size: ", reward.size())

        q = self.net(state)
        #print("q values: ", q)
        #print()
        #print("q size: ", q.size())
        self.t_state = state
        self.t_action = action
        self.t_reward = reward
        self.q = q
        td_estimate = q[np.arange(0,self.batch_size),action.to(torch.int64)]
        #print("td_estimate: ",td_estimate)
        #print("td_estimate.size(): ",td_estimate.size())
        with torch.no_grad():
            net_next_q = self.net(next_state)
            target_next_q = self.target_net(next_state)
            net_action = torch.argmax(net_next_q,axis=1)
            target_net_q = target_next_q[np.arange(0,self.batch_size),net_action]
        
        # TO-DO modify this to consider end-of-episode behavior correctly (1-done?)
        td_target = (reward + self.gamma *target_net_q).float()
        #print("td_target: ",td_target)
        #print("td_target.size(): ", td_target.size())

        # Calculate the loss
        loss = self.loss(td_estimate,td_target)

        # Zero the gradient and calculate the gradient
        self.optimizer.zero_grad()
        loss.backward()

        # TO-DO
        # Consider clamping gradients
        for param in self.net.parameters():
            param.grad.data.clamp_(-1,1)
        # print("q values: ", q)
        # print("q size: ", q.size())
        # print("td_estimate: ",td_estimate)
        # print("td_estimate.size(): ",td_estimate.size())
        # print("td_target: ",td_target)
        # print("td_target.size(): ", td_target.size())
        # print("loss: ",loss)
        # print("loss.size(): ", loss.size())
        # breakpoint()
        # Run optimization step
        self.optimizer.step()
        return
    
    # Synchronize the policy net (net) and the target net (target_net)
    def sync_net(self):
        self.target_net.load_state_dict(self.net.state_dict())