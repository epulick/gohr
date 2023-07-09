# Adapted from work by Shubham Bharti and Yiding Chen

import pandas as pd
import numpy as np
import copy
import random, math

from tqdm import tqdm

np.set_printoptions(precision=2)

class meta_bandit():
    def __init__(self,env,args,log_paths):
        # Pull in the game environment
        self.env = env
        
        # Misc. parameter import
        self.train_horizon = args['TRAIN_HORIZON']
        self.n_episodes = args['TRAIN_EPISODES']
        self.steps = 0
        self.init_q_value = -.5
        
        self.record_moves = False
        self.move_path,self.ep_path = log_paths[0],log_paths[1]
        if args["RUN_TYPE"]=='cluster':
            self.tqdm=False
        else:
            self.tqdm=True
        
        self.in_dim, self.out_dim = self.env.in_dim, self.env.out_dim
        self.q_values = np.full((self.in_dim,self.out_dim),self.init_q_value)

        # Set up dataframe for recording the results
        # Switch commented lines as part of removing move-by-move results
        if self.record_moves:
            self.all_data_df = pd.DataFrame(columns=['episode', 'time', 'action_type', 'action', 'reward', 'done', 'board'])
            #self.all_data_df = pd.DataFrame({'episode':0, 'time':0, 'action_type':0, 'action':0, 'reward':0, 'done':0, 'epsilon':0,'board':0},index=[0])
            self.all_data_df.to_csv(self.move_path,mode='a',index=False)
        
        # Always record episodes
        self.episode_df = pd.DataFrame(columns=['episode','reward'])
        self.episode_df.to_csv(self.ep_path,mode='a',index=False)

    def train(self):
        for episode in tqdm(range(self.n_episodes)) if self.tqdm else range(self.n_episodes):
            # Set initial variables for looping
            if self.record_moves:
                all_data_df_list = []
            
            done = 0
            episode_reward = 0
            state_dict = self.env.reset()
            state = state_dict['features']
            row = state_dict['row']
            col = state_dict['col']

            # Loop over the per-episode training horizon
            for t in range(self.train_horizon):
                # Step the environment forward with an action
                action = self.select_action(state)
                next_state_dict, reward, done, move_result = self.env.step(int(action),row,col,4)
                next_state = next_state_dict['features']
                next_row = next_state_dict['row']
                next_col = next_state_dict['col']
                
                # Add step reward to episode reward
                episode_reward+=move_result 

                # Write current step's data to a dataframe and concat with the main dataframe
                log_action = "g"
                if self.record_moves:
                    current_df = pd.DataFrame({'episode':episode, 'time':t, 'action_type':log_action, 'action':int(action), 'reward':int(move_result), 'done':done,'board':[self.env.board]},index=[0])
                    all_data_df_list.append(current_df)

                # Learn from feedback
                self.learn(state,action,reward)

                # Set up next iteration
                state = next_state
                row = next_row
                col = next_col
                
                self.steps+=1

                # Truncate episode early if the board is clear
                if done:
                    break

            # Reset the episode reward before the next iteration
            episode_df = pd.DataFrame({'episode':episode, 'reward':episode_reward},index=[0])
            episode_df.to_csv(self.ep_path,header=False,mode='a',index=False)
            if self.record_moves:
                all_data_df=pd.concat(all_data_df_list,ignore_index=True)
                all_data_df.to_csv(self.move_path,header=False,mode='a',index=False)
            episode_reward=0
        print(self.credibility)

    def select_action(self,state):
        # Choose action greedily
        q_vals = self.q_values[state,:]
        return np.argmax(q_vals)
        
    def learn(self,state,action,reward):
        #breakpoint()
        old_val = self.q_values[state,action]
        self.q_values[state,action]=reward
        if self.credibility!=-1 and old_val!=self.init_q_value:
            if old_val==reward:
                self.credibility+=1
            else:
                self.credibility=-1
        #breakpoint()