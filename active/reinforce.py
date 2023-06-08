
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

#np.set_printoptions(precision=2)

class MLP(nn.Module):
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

class REINFORCE():
    def __init__(self,env,args,log_paths):
        # Put onto correct hardware
        #self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")

        # Pull in the game environment
        self.env = env
        
        # Misc. parameter import
        self.train_horizon = args['TRAIN_HORIZON']
        self.n_episodes = args['TRAIN_EPISODES']
        self.gamma = args['GAMMA']
        self.move_path,self.ep_path = log_paths[0],log_paths[1]
        if args["RUN_TYPE"]=='cluster':
            self.tqdm=False
        else:
            self.tqdm=True

        # Build the neural network (net and target_net)
        self.in_dim, self.out_dim = self.env.in_dim, self.env.out_dim
        self.net = MLP(self.in_dim,self.out_dim,args['HIDDEN_SIZES'],args["ACTIVATION"]).to(self.device)
        #print(self.net)
        # Consider other optimizers or losses via parameter input
        if args['OPTIMIZER'] == 'ADAM':
            self.optimizer=torch.optim.Adam(self.net.parameters(),args['LR'])
        elif args['OPTIMIZER'] == 'RMSprop':
            self.optimizer=torch.optim.RMSprop(self.net.parameters(),args['LR'])
        elif args['OPTIMIZER'] == 'SGD':
            self.optimizer=torch.optim.SGD(self.net.parameters(),args['LR'])
        else:
            breakpoint()

        # Set up dataframe for recording the results
        self.all_data_df = pd.DataFrame(columns=['episode', 'time', 'action', 'reward', 'done', 'board'])
        self.all_data_df.to_csv(self.move_path,mode='a',index=False)
        #self.loss_df = pd.DataFrame(columns= ['loss'])
        self.episode_df = pd.DataFrame(columns=['episode','reward'])
        self.episode_df.to_csv(self.ep_path,mode='a',index=False)

    def transform_rewards(self,rewards, g):
        steps = len(rewards)
        traj_returns = np.empty(steps)
        future_return = 0

        for i in reversed(range(steps)):
            future_return = rewards[i]+g*future_return
            traj_returns[i]=future_return
        return traj_returns-traj_returns.mean()

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

            log_probs = []
            rewards = []
    
            # Loop over the per-episode training horizon
            for t in range(self.train_horizon):
                # Step the environment forward with an action
                t_state = torch.from_numpy(state).float().to(self.device)
                action,log_prob = self.select_action(t_state,mask,valid)
                next_state_dict, reward, done, move_result = self.env.step(int(action))
                next_state = next_state_dict['features']
                next_mask = next_state_dict['mask']
                next_valid = next_state_dict['valid']
                
                # Add step reward to episode reward
                episode_reward+=move_result

                # Write current step's data to a dataframe and concat with the main dataframe
                current_df = pd.DataFrame({'episode':episode, 'time':t, 'action':int(action), 'reward':int(move_result), 'done':done, 'board':[self.env.board]},index=[0])
                # Comment out to stop logging move data if file sizes are too large
                all_data_df_list.append(current_df)

                # Update values to prepare for next iteration
                rewards.append(reward)
                log_probs.append(log_prob)

                state=next_state
                mask=next_mask
                valid=next_valid

                # Truncate episode early if the board is clear
                if done:
                    break

            # Reset the episode reward before the next iteration
            episode_df = pd.DataFrame({'episode':episode, 'reward':episode_reward},index=[0])
            episode_df.to_csv(self.ep_path,header=False,mode='a',index=False)
            # Comment out to stop logging move data if file sizes are too large
            all_data_df=pd.concat(all_data_df_list,ignore_index=True)
            all_data_df.to_csv(self.move_path,header=False,mode='a',index=False)
            episode_reward=0

            # Pass off episode information for training
            t_rewards = torch.FloatTensor(self.transform_rewards(rewards,self.gamma))
            t_log_probs = torch.stack(log_probs)
            #print(rewards)
            loss = self.learn(t_rewards,t_log_probs)

    def select_action(self,state,mask,valid):
        orig_logits = self.net(state)
        mask_value = torch.finfo(orig_logits.dtype).min
        boolmask = torch.BoolTensor(mask,device=self.device)
        masked_logits = orig_logits.masked_fill(boolmask,mask_value)
        if torch.isnan(masked_logits).any().item():
            print("orig_logits",orig_logits)
            print("state",state)
            print("mask",mask)
            print("valid",valid)
            print("masked_logits",masked_logits)
            breakpoint()
        action_dist = torch.distributions.categorical.Categorical(logits=masked_logits)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action.item(),log_prob
        
    def learn(self,rewards,log_probs):
        
        self.optimizer.zero_grad()
        loss_vals = -1*rewards*log_probs
        loss=torch.mean(loss_vals)
        loss.backward()
        self.optimizer.step()

        return loss.item()