# Adapted from work by Shubham Bharti and Yiding Chen

import pandas as pd
import numpy as np
import copy
import random, math
from itertools import combinations
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
        
        self.record_moves = args['RECORD_MOVES']
        self.move_path,self.ep_path = log_paths[0],log_paths[1]
        if args["RUN_TYPE"]=='cluster':
            self.tqdm=False
        else:
            self.tqdm=True
        
        # Set up bandits
        self.model_features = args['MODEL_FEATURES']
        self.combination_types = args['FEATURE_ARRANGEMENTS']
        self.feature_combinations = []
        for r in self.combination_types:
            self.feature_combinations.extend(combinations(self.model_features,r))
        #breakpoint()
        self.models=[bandit(feat_arr,env.calc_dim(feat_arr)) for feat_arr in self.feature_combinations]

        # Set up dataframe for recording the results
        # Switch commented lines as part of removing move-by-move results
        if self.record_moves:
            self.all_data_df = pd.DataFrame(columns=['episode', 'time', 'action_type', 'action', 'reward', 'done', 'board'])
            self.all_data_df.to_csv(self.move_path,mode='a',index=False)
        
        # Always record episodes
        self.episode_df = pd.DataFrame(columns=['episode','reward'])
        self.episode_df.to_csv(self.ep_path,mode='a',index=False)
        #breakpoint()
    def train(self):
        for episode in tqdm(range(self.n_episodes)) if self.tqdm else range(self.n_episodes):
            # Set initial variables for looping
            if self.record_moves:
                all_data_df_list = []
            
            done = 0
            episode_reward = 0
            state_dict = self.env.reset()
            move_row = state_dict['move_row']
            move_col = state_dict['move_col']

            # Loop over the per-episode training horizon
            for t in range(self.train_horizon):
                # Step the environment forward with an action
                action,control = self.select_action(state_dict)
                next_state_dict, reward, done, move_result = self.env.step(int(action),move_row,move_col,4)
                #next_state = next_state_dict['features']
                #print(next_state_dict)
                next_move_row = next_state_dict['move_row']
                next_move_col = next_state_dict['move_col']
                
                # Add step reward to episode reward
                episode_reward+=move_result 

                # Write current step's data to a dataframe and concat with the main dataframe
                log_action = self.feature_combinations[control]
                if self.record_moves:
                    current_df = pd.DataFrame({'episode':episode, 'time':t, 'action_type':str(log_action), 'action':int(action), 'reward':int(move_result), 'done':done,'board':[self.env.board]},index=[0])
                    all_data_df_list.append(current_df)

                # Learn from feedback
                for model in self.models:
                    model.learn(action,reward)

                # Set up next iteration
                state_dict = next_state_dict
                move_row = next_move_row
                move_col = next_move_col
                
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
        # for model in self.models:
        #     print(model.feats)
        #     print(model.return_credibility())
        #     #print(model.q_values)


    def select_action(self,states):
        candidate_actions = [model.propose_action(states) for model in self.models]
        candidate_credibilities = [model.return_credibility() for model in self.models]
        #print(candidate_actions)
        #print(candidate_credibilities)
        if all(cred == -1 for cred in candidate_credibilities):
            print("all models lost credibility")
            breakpoint()
        best_candidate = np.argmax(candidate_credibilities)
        #print(best_candidate)
        return candidate_actions[best_candidate],best_candidate
    
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
        

class bandit():
    def __init__(self,feats,dims):
        # Initialze bandit q-table and credibility
        self.init_q_value = 0
        self.feats = feats
        self.feat_dims = dims
        self.in_dim, self.out_dim = np.prod(self.feat_dims), 4
        self.q_values = np.full((self.in_dim,self.out_dim),self.init_q_value,dtype=np.int8)
        self.credibility=1

    def propose_action(self,state_dict):
        #print(self.feats)
        self.state=None
        states = tuple(state_dict[feat] for feat in self.feats)
        self.state = np.ravel_multi_index(states,self.feat_dims)
        q_vals = self.q_values[self.state,:]
        #print(self.state)
        #print(q_vals)
        return np.argmax(q_vals)
    
    def return_credibility(self):
        return self.credibility
        
    def learn(self,action,reward):
        if self.state == None:
            print("error!")
            breakpoint()
        old_val = self.q_values[self.state,action]
        self.q_values[self.state,action]=int(reward)
        if self.credibility!=-1 and old_val!=self.init_q_value:
            if old_val==reward:
                self.credibility+=1
            else:
                self.credibility-=3
                # Suspect there is a bug in this causing some runs to be massive failures (when an algorithm is instantly discredited)