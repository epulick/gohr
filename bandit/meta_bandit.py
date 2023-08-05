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
            self.all_data_df = pd.DataFrame(columns=['episode', 'time', 'action_type', 'action','move_row','move_col', 'acting_cred','reward', 'done', 'board','cred'])
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
            state_dict_list = self.env.reset()

            # Loop over the per-episode training horizon
            for t in range(self.train_horizon):
                # Step the environment forward with an action
                (bucket,move_row,move_col,piece_index),control = self.select_action(state_dict_list)
                next_state_dict_list, reward, done, move_result = self.env.step(bucket,move_row,move_col,4)
                
                # Add step reward to episode reward
                episode_reward+=move_result 

                # Write current step's data to a dataframe and concat with the main dataframe
                log_action = self.feature_combinations[control]
                if self.record_moves:
                    cred = self.get_credibilities()
                    acting_q = self.models[control].return_qvals()
                    #print(acting_q)
                    #breakpoint()
                    current_df = pd.DataFrame({'episode':episode, 'time':t, 'action_type':str(log_action), 'action':int(bucket),'move_row':move_row,'move_col':move_col,'acting_cred':[acting_q], 'reward':int(move_result), 'done':done,'board':[self.env.board],'cred':[cred]},index=[0])
                    all_data_df_list.append(current_df)

                # Learn from feedback
                for model in self.models:
                    model.learn(bucket,piece_index,reward)

                # Set up next iteration
                state_dict_list = next_state_dict_list
                
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


    def select_action(self,states):
        # returns a list of tuples (bucket,move_row,move_col) from each model
        candidate_actions = [model.propose_action(states) for model in self.models]
        # list of credibility scores from each model
        candidate_credibilities = [model.return_credibility() for model in self.models]
        # Index of the model with the best credibility
        best_candidate = np.argmax(candidate_credibilities)
        return candidate_actions[best_candidate],best_candidate
    
    def get_credibilities(self):
        credibilities = [(model.feats,model.return_credibility()) for model in self.models]
        #breakpoint()
        return credibilities

class bandit():
    def __init__(self,feats,dims):
        # Initialize bandit q-table and credibility
        self.init_q_value = 0
        self.feats = feats
        self.feat_dims = dims
        self.in_dim, self.out_dim = np.prod(self.feat_dims), 4
        self.q_values = np.full((self.in_dim,self.out_dim),self.init_q_value,dtype=np.int8)
        self.credibility=-1*len(self.feats)

    def propose_action(self,state_dict_list):
        piece_count = len(state_dict_list)
        self.state_list = []
        for i,piece in enumerate(state_dict_list):
            states = tuple(state_dict_list[i][feat] for feat in self.feats)
            state = np.ravel_multi_index(states,self.feat_dims)
            self.state_list.append(state)
        q_vals = self.q_values[self.state_list,:]
        selection = np.random.choice(np.flatnonzero(q_vals==q_vals.max()))
        bucket = selection % 4
        piece_index = selection // 4
        selected_piece = state_dict_list[piece_index]
        return (bucket,selected_piece['move_row'],selected_piece['move_col'],piece_index)
    
    def return_credibility(self):
        return self.credibility
        
    def learn(self,action,piece_index,reward):
        if len(self.state_list)==0:
            print("error!")
            breakpoint()
        #breakpoint()
        state = self.state_list[piece_index]
        old_val = self.q_values[state,action]
        self.q_values[state,action]=int(reward)
        if old_val!=self.init_q_value and self.credibility!=-1:
            if old_val==int(reward):
                self.credibility+=1
            else:
                self.credibility=-1

    def return_qvals(self):
        return np.copy(self.q_values)