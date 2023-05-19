# Adapted from work by Shubham Bharti and Yiding Chen

import pandas as pd
import numpy as np
import copy
import random, math

import torch
import torch.nn as nn
#import torch.nn.functional as F
#import torch.optim as optim
from tqdm import tqdm
from collections import namedtuple, deque

np.set_printoptions(precision=2)

verbose = 0

class QNet(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_sizes, act):
        super().__init__()
        if act == "ReLU":
            activation = nn.ReLU
        elif act == "LeakyReLU":
            activation = nn.LeakyReLU
        else:
            breakpoint()
        # Make list of sizes from the user input and the defined inner/outer dimensions of the problem
        sizes = [in_dim] + hidden_sizes + [out_dim]
        
        # Generate a list of the layers and activations to zip up at the end
        layers = []
        for i in range(len(sizes)-2):
            layers+=[nn.Linear(sizes[i], sizes[i+1]),activation()]
        layers += [nn.Linear(sizes[-2],sizes[-1]),nn.Identity()]
        # Create network from the list
        self.network= nn.Sequential(*layers)

    def forward(self,X):
        return self.network(X)

class ReplayMemory(object):
    def __init__(self, capacity, Transition):
        self.memory = deque([], maxlen=capacity)
        self.transitions = Transition

    # Add new experience to replay memory
    def push(self, *args):
        self.memory.append(self.transitions(*args))

    # Get a sample of the requested batch size
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN():
    def __init__(self,env,args,log_paths):
        # Put onto correct hardware
        #self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")

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
        self.move_path,self.ep_path = log_paths[0],log_paths[1]
        if args["RUN_TYPE"]=='cluster':
            self.tqdm=False
        else:
            self.tqdm=True

        # Build the neural network (net and target_net)
        self.in_dim, self.out_dim = self.env.in_dim, self.env.out_dim
        self.net = QNet(self.in_dim,self.out_dim,args['HIDDEN_SIZES'],args["ACTIVATION"]).to(self.device)
        #self.net = self.make_nn(self.in_dim, self.out_dim, args['HIDDEN_SIZES'])#.to(self.device)
        self.target_net = copy.deepcopy(self.net)
        self.clamp = args["CLAMP"]

        # Q_target parameters are frozen.
        for p in self.target_net.parameters():
            p.requires_grad = False

        # TO-DO
        # Consider other optimizers or losses via parameter input
        if args['OPTIMIZER'] == 'ADAM':
            self.optimizer=torch.optim.Adam(self.net.parameters(),args['LR'])
        elif args['OPTIMIZER'] == 'RMSprop':
            self.optimizer=torch.optim.RMSprop(self.net.parameters(),args['LR'])
        elif args['OPTIMIZER'] == 'SGD':
            self.optimizer=torch.optim.SGD(self.net.parameters(),args['LR'])
        else:
            breakpoint()
        self.loss = nn.SmoothL1Loss()

        # Set up dataframe for recording the results
        # to do - consider adding return code (some pieces are more informative than others)
        #self.all_data_df = pd.DataFrame(columns=['episode', 'time', 'action_type', 'action', 'reward', 'done','epsilon','board','valid','debug_q','zero_ind_action_tuple', 'other'])
        self.all_data_df = pd.DataFrame(columns=['episode', 'time', 'action_type', 'action', 'reward', 'done','epsilon', 'board'])
        self.all_data_df.to_csv(self.move_path,mode='a',index=False)
        self.loss_df = pd.DataFrame(columns= ['loss'])
        self.episode_df = pd.DataFrame(columns=['episode','reward'])
        self.episode_df.to_csv(self.ep_path,mode='a',index=False)

        # Set up replay memory
        self.transitions = namedtuple('Transition', ('state','action','next_state','reward','done'))
        self.replay_memory = ReplayMemory(args['REPLAY_BUFFER_SIZE'], self.transitions)

    # Synchronize the policy net (net) and the target net (target_net)
    def sync_net(self):
        self.target_net.load_state_dict(self.net.state_dict())

    def train(self):
        for episode in tqdm(range(self.n_episodes)) if self.tqdm else range(self.n_episodes):
            # Set initial variables for looping
            all_data_df_list = []
            self.debug = 0
            done = 0
            episode_reward = 0
            state_dict = self.env.reset()
            state = state_dict['features']
            mask = state_dict['mask']
            valid = state_dict['valid']
            # --------------
            # DEBUGGING BLOCK
            # --------------
            # print("torch.from_numpy(state): ", torch.from_numpy(state))
            # print("torch.from_numpy(state).float(): ", torch.from_numpy(state).float())
            # print("torch.from_numpy(state).float().unsqueeze(0): ", torch.from_numpy(state).float().unsqueeze(0))
            # print("torch.from_numpy(state).float().size: ", torch.from_numpy(state).float().size()
            # print("torch.from_numpy(state).float().unsqueeze(0).size: ", torch.from_numpy(state).float().unsqueeze(0).size())
            
            # Get the initial state value to start the looping
            state = torch.from_numpy(state).float().to(self.device)
            # Loop over the per-episode training horizon
            for t in range(self.train_horizon):
                # Step the environment forward with an action
                action, action_type, eps, debug_q = self.select_action(state,mask,valid,episode,t)
                next_state_dict, reward, done, move_result = self.env.step(int(action))
                next_state = next_state_dict['features']
                next_mask = next_state_dict['mask']
                next_valid = next_state_dict['valid']

                # Process next_state into a tensor
                next_state = torch.from_numpy(next_state).float().to(self.device)
                
                # Add step reward to episode reward
                episode_reward+=move_result #reward

                # Write current step's data to a dataframe and concat with the main dataframe
                #breakpoint()
                #current_df = pd.DataFrame({'episode':episode, 'time':t, 'action_type':action_type, 'action':int(action), 'reward':int(reward), 'done':done, 'epsilon':eps,'board':[self.env.board],'valid':[valid],'debug_q':[debug_q], 'zero_ind_action_tuple':[self.env.action_index_to_tuple(action)], 'other':'none'},index=[0])
                #breakpoint()
                if action_type=="random":
                    log_action = "r"
                else:
                    log_action = "g"
                #current_df = pd.DataFrame({'episode':episode, 'time':t, 'action_type':log_action, 'action':int(action), 'reward':int(move_result), 'done':done, 'epsilon':"{0:.4f}".format(eps),'board':[self.env.board]},index=[0])
                #all_data_df_list.append(current_df)
                
                #if done:
                #    self.all_data_df.to_csv(self.move_path,header=False,mode='a',index=False)
                #    self.all_data_df = pd.DataFrame(columns=['episode', 'time', 'action_type', 'action', 'reward', 'done','epsilon', 'board'])
                # Append this step to the replay buffer
                action, reward, done = torch.IntTensor([action]).to(self.device), torch.FloatTensor([reward]).to(self.device), torch.FloatTensor([done])
                #action, reward = torch.IntTensor([action]), torch.FloatTensor([reward])
                self.replay_memory.push(state, action, next_state, reward, done)

                # TO DO - double check on behavior at end of episode
                state=next_state
                mask=next_mask
                valid=next_valid
                
                # Optimize the model
                loss = self.learn(episode,t)
                current_loss = pd.DataFrame({'loss':loss},index=[0])
                #self.loss_df = pd.concat([self.loss_df, current_loss],ignore_index=True)
                self.steps+=1

                # If at an update interval, copy policy net to target net
                if (self.steps % self.sync == 0):
                    self.sync_net()

                # Truncate episode early if the board is clear
                if done:
                    break

            # Reset the episode reward before the next iteration
            episode_df = pd.DataFrame({'episode':episode, 'reward':episode_reward},index=[0])
            episode_df.to_csv(self.ep_path,header=False,mode='a',index=False)
            #all_data_df=pd.concat(all_data_df_list,ignore_index=True)
            #all_data_df.to_csv(self.move_path,header=False,mode='a',index=False)
            #self.episode_df=pd.concat([self.episode_df, episode_df],ignore_index=True)
            episode_reward=0

    def select_action(self,state,mask,valid,ep,t):
        # epsilon greedy policy - epsilon decays exponentially with time
        eps_threshold = self.eps_end + (self.eps_start-self.eps_end)*math.exp(-1*self.steps/self.eps_decay)
        with torch.no_grad():
            debug_q = self.net(state)[valid]
        # Random exploration action
        if np.random.rand()<eps_threshold:
            # Choose an action from the provided list of valid actions
            action = random.choice(valid)
            # Alternative implementation which doesn't mask out 'invalid actions'
            #action = self.env.action_space.sample()

            action_type = 'random'

        # Greedy action per policy
        else:
            with torch.no_grad():
                #breakpoint()
                q_val = self.net(state)
                min_val = q_val.min(0)[0]-1
                boolmask = torch.BoolTensor(mask)
                masked_q = q_val.masked_fill(boolmask,min_val)
                action = masked_q.max(0)[1].to(self.device)
                # --------------
                # DEBUGGING BLOCK
                # --------------
                #print("self.net(state): ",self.net(state))
                #print("self.net(state).max(1): ",self.net(state).max(0))
                #print("self.net(state).max(1)[1]: ",self.net(state).max(0)[1])
                
            action_type = 'greedy'
        return action, action_type, eps_threshold, debug_q
        
    def learn(self,ep, t):
        
        # Can only optimize if the buffer filled above the batch threshold
        if len(self.replay_memory) < self.batch_size:
            return
        
        batch =self.replay_memory.sample(self.batch_size)
        
        state, action, next_state, reward, done = map(torch.stack, zip(*batch))
        
        # --------------
        # DEBUGGING BLOCK
        # --------------
        # print("state: ", state)
        # print("state size: ", state.size())
        # print("action: ", action)
        # print("action size: ", action.size())
        # print("reward: ", reward)
        # print("reward size: ", reward.size())
        
        # Get the Q values from the online Q network
        q = self.net(state)
        # TD-estimate is the Q values from the actions taken by the learner
        td_estimate = self.net(state).gather(1,action.to(torch.int64))
       
        # --------------
        # DEBUGGING BLOCK
        # --------------
        #print("q values: ", q)
        #print()
        #print("q size: ", q.size())
        #self.t_state = state
        #self.t_action = action
        #self.t_reward = reward
        #self.q = q
        #td_estimate = q[np.arange(0,self.batch_size),action.to(torch.int64)]
        #print("td_estimate: ",td_estimate)
        #print("td_estimate.size(): ",td_estimate.size())

        # No need for gradient in gathering the input information for the TD-target
        with torch.no_grad():
            # Q values from the online network in the next state
            net_next_q = self.net(next_state)
            # Q values from the offline network in the next stat
            target_next_q = self.target_net(next_state)
            # Choose the action based on the online network
            net_action = torch.argmax(net_next_q,axis=1).unsqueeze(1)
            # Evaluate the action based on the offline network
            target_net_q = target_next_q.gather(1,net_action)
        
        td_target = (reward + (1-done)*self.gamma *target_net_q).float()

        # --------------
        # DEBUGGING BLOCK
        # --------------
        #print("td_target: ",td_target)
        #print("td_target.size(): ", td_target.size())
        
        # Calculate the loss
        loss = self.loss(td_estimate,td_target)
        if (ep%200)==0 and not(self.debug):
            #breakpoint()
            self.debug=1

        # Zero the gradient and calculate the gradient
        self.optimizer.zero_grad()
        loss.backward()

        if self.clamp:
            for param in self.net.parameters():
                param.grad.data.clamp_(-1,1)

        # --------------
        # DEBUGGING BLOCK
        # --------------
        # print("q values: ", q)
        # print("q size: ", q.size())
        # print("td_estimate: ",td_estimate)
        # print("td_estimate.size(): ",td_estimate.size())
        # print("td_target: ",td_target)
        # print("td_target.size(): ", td_target.size())
        # print("loss: ",loss)
        # print("loss.size(): ", loss.size())

        # Run optimization step
        self.optimizer.step()
        return loss.item()