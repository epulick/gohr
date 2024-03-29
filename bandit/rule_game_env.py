# Parent class to engine - contains methods for more complicated interaction with CGS (similar to gym env)
# Originally written by Shubham Bharti and Yiding Chen

import numpy as np
import gym, os, sys
from gym import spaces
from collections import deque
from rule_game_engine import *

class RuleGameEnv(gym.Env, RuleGameEngine):

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #
    #    Base environment class, get_features() should be implemented by individual rules.
    #
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def __init__(self, args):
        super(RuleGameEnv, self).__init__(args)
        # define an action space with possible number of distinct actions
        self.action_space = spaces.Discrete(self.board_size*self.board_size*self.bucket_space)
        self.reset_lists()
        # self.last_attributes = []
        # self.last_moves = []
        self.prev_board = None
        # self.prev_attributes = None
        # self.n_steps = args['N_STEPS']
        # self.learner = args['LEARNER']
        # for i in range(self.n_steps):
        #     self.last_boards.append(None)
        #     self.last_moves.append(None)
        #     self.last_attributes.append(None)
        self.error_count = 0

    def reset_lists(self):
        memory = 5
        self.full_move_list = deque([None for x in range(memory)],maxlen=memory)
        self.reduced_move_list = deque([None for x in range(memory)],maxlen=memory)
        self.board_list = deque([None for x in range(memory)],maxlen=memory)
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #
    #   Extract feature vector - to extract the features corresponding to all actions
    #
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def get_feature(self):
        pass
    
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #
    #   Reset the backend to a new episode
    #
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def reset(self):
        self.sample_new_board()
        self.reset_lists()
        self.prev_board=None
        # self.last_attributes = []
        # self.last_boards = []
        # self.last_moves = []
        # for i in range(self.n_steps):
        #     self.last_boards.append(None)
        #     self.last_moves.append(None)
        #     self.last_attributes.append(None)
        #breakpoint()
        return self.get_feature()

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #
    #   Gym step function
    #       - input : agents actions 
    #       - returns : (observation_vector, reward, done, info)
    #
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def step(self, action, action_row_index = None, action_col_index=None, dim=144):
        
        # If the provided action uses the complete action space:
        if dim == 144:
            # Map action from action index back to a row, column, and bucket interpretable to the CGS as a move
            action_row_index, action_col_index, action_bucket_index = self.action_index_to_tuple(action)
            action_row_index+=1
            action_col_index+=1
        else:
            # In the reduced action space, the bucket itself is the index
            action_bucket_index=action

        # Defined in engine, map 0-3 out to the specific (row,col) coordinates of the bucket
        bucket_row, bucket_col = self.bucket_tuple[action_bucket_index]         
        self.prev_board = self.board

        # done : 'True' if the episode ends, response_code : response code of the move - accepted(0) etc.
        done, response_code, reward = self.take_action(action_row_index, action_col_index, bucket_row, bucket_col)          # cgs requires one-indexed board positions
        
        # if response code is 0(action is accepted), update the last successful step information
        # Note that other relevant codes are (as of 10/17/22):
        # 2 (stalemate), 
        # 3 (move rejected because cell is empty), 
        # 4 (move rejected because object not permitted in that bucket), 
        # 7 (move rejected since piece is immovable)

        if(response_code==0):
            # Only update the move/board lists if the action was accepted (i.e., there has been a change in state of the MDP)
            self.full_move_list.append(self.action_tuple_to_index(action_row_index-1,action_col_index-1,action_bucket_index))
            self.reduced_move_list.append(action_bucket_index)
            self.board_list.append(self.prev_board)
        else:
            # if self.learner!="REINFORCE":
            #     self.move_list.append(action)
            self.error_count+=1
        if (response_code==0):
            # Re-using info field to denote a correct or incorrect move for logging purposes (separate from the chosen reward values)
            info=0
        else:
            info=-1
        feature = self.get_feature()

        return feature, reward, done, info 

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #
    #   Some utility functions
    #       - action_tuple_to_index : convert action_tuple to action_index  
    #       - action_index_to_tuple : convert action_index to action_tuple
    #
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def action_tuple_to_index(self, o_row, o_col, b_index):

        return np.ravel_multi_index((o_row,o_col,b_index),(self.board_size,self.board_size,self.bucket_space))

    def action_index_to_tuple(self, action):
        # inputs : zero-index flattened action index
        # output : zero-index action tuple (o_row, o_col, b_index)
        return np.unravel_index(action, (self.board_size, self.board_size, self.bucket_space))         
    
def test_rule_game_env(args):
    # some testing code
    env = RuleGameEnv(args)
    phi = env.get_feature()
    breakpoint()

if __name__ == "__main__":
    print("starting")

    rule_dir_path, rule_name = sys.argv[1], 'rules-05.txt'
    rule_file_path = os.path.join(rule_dir_path, rule_name)

    args = {
        'RULE_FILE_PATH' : rule_file_path,
        'RULE_NAME'  : rule_name,
        'BOARD_SIZE'  : 6,
        'OBJECT_SPACE'  : 16,
        'COLOR_SPACE'  : 4,
        'SHAPE_SPACE'  : 4,
        'BUCKET_SPACE'  : 4,
        'INIT_OBJ_COUNT'  : 3, 
        'R_ACCEPT' : -1,
        'R_REJECT' : -1,
        'TRAIN_HORIZON' : 300,
        'ALPHA' :   1,   
        'TEST_EPISODES' :  100,
        'TEST_FREQ' :   1000,
        'VERBOSE' : 1,
        'LR' : 1e-2,
        'SHAPING' : 0,
        'SEED' : 0,
        'RUN_MODE' : "RULE"
    }

    test_rule_game_env(args)
    
