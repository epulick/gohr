# Parent class to engine - contains methods for more complicated interaction with CGS (similar to gym env)
# Originally written by Shubham Bharti and Yiding Chen

import numpy as np
import gym, os, sys
import traceback
from gym import spaces
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
        self.l_shape, self.l_color, self.l_bucket, self.l_index  = -1,-1,-1,-1        # last successful shape, color, bucket, index
        self.c_shape, self.c_color, self.c_bucket, self.c_index  = -1,-1,-1,-1       # current shape, color, bucket
        self.move_list=[]
        self.last_boards = []
        self.last_attributes = []
        self.last_moves = []
        self.prev_board = None
        self.prev_attributes = None
        self.n_steps = args['N_STEPS']
        for i in range(self.n_steps):
            self.last_boards.append(None)
            self.last_moves.append(None)
            self.last_attributes.append(None)
        self.error_count = 0
        
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
        self.l_shape, self.l_color, self.l_bucket, self.l_index = -1,-1,-1,-1    # reset last successful shape, color, bucket
        self.c_shape, self.c_color, self.c_bucket, self.c_index = -1,-1,-1,-1    # reset current shape, color, bucket
        self.last_attributes = []
        self.last_boards = []
        self.last_moves = []
        for i in range(self.n_steps):
            self.last_boards.append(None)
            self.last_moves.append(None)
            self.last_attributes.append(None)
        #breakpoint()
        return self.get_feature()

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #
    #   Gym step function
    #       - input : agents actions 
    #       - returns : (observation_vector, reward, done, info)
    #
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def step(self, action):
        # Map action from action index back to a row, column, and bucket interpretable to the CGS as a move
        action_row_index, action_col_index, action_bucket_index = self.action_index_to_tuple(action)
        # Defined in engine, map 0-3 out to the specific (row,col) coordinates of the bucket
        bucket_row, bucket_col = self.bucket_tuple[action_bucket_index]  

        o_shape, o_color = -1, -1

        # if the o_row, o_col corresponding to the current action has some object, pick its shape and color
        for object_tuple in self.board:
            tuple_row, tuple_col = object_tuple['y']-1, object_tuple['x']-1
            if(tuple_row == action_row_index and tuple_col == action_col_index):
                o_shape, o_color = self.shape_id[object_tuple['shape']], self.color_id[object_tuple['color']]          

        self.c_shape = o_shape
        self.c_color = o_color
        self.c_bucket = action_bucket_index
        self.c_index = action_row_index*self.board_size+action_col_index                                                    # cgs specifies bucket with (row, column) tuple
        
        # done : 'True' if the episode ends, response_code : response code of the move - accepted(0) etc.
        self.prev_board = self.board
        self.prev_attributes = (o_shape,o_color)
        done, response_code, reward = self.take_action(action_row_index+1, action_col_index+1, bucket_row, bucket_col)          # cgs requires one-indexed board positions
        
        # if response code is 0(action is accepted), update the last successful step information
        # Note that other relevant codes are (as of 10/17/22):
        # 2 (stalemate), 
        # 3 (move rejected because cell is empty), 
        # 4 (move rejected because object not permitted in that bucket), 
        # 7 (move rejected since piece is immovable)

        if(response_code==0):
            #self.l_shape, self.l_color, self.l_bucket, self.l_index = self.c_shape, self.c_color, self.c_bucket, action_row_index*self.board_size+action_col_index
            self.l_shape, self.l_color, self.l_bucket, self.l_index = self.c_shape, self.c_color, self.c_bucket, self.c_index
            self.move_list=[]
            # Record the successful move internally for n_steps of memory
            self.last_moves.pop(0)
            self.last_moves.append(action)
            self.last_boards.pop(0)
            self.last_boards.append(self.prev_board)
            self.last_attributes.pop(0)
            self.last_attributes.append(self.prev_attributes)
        else:
            self.move_list.append(action)
            self.error_count+=1
        info, feature = {}, self.get_feature()

        return feature, reward, done, info 

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #
    #   Some utility functions
    #       - render : to render the board in console
    #       - action_tuple_to_index : convert action_tuple to action_index  
    #       - action_index_to_tuple : convert action_index to action_tuple
    #
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def render(self, mode='console'):
        print("Current board feature")
        print(self.get_feature())

    #def action_tuple_to_index(self, o_row, o_col, b_index):
        '''
            inputs : zero-index action tuple  (o_row, o_col, b_index)
            output : zero-index flattened action index
        '''
    #   return o_row+o_col*self.board_size+b_index*self.board_size*self.board_size
    def action_tuple_to_index(self, o_row, o_col, b_index):

        return np.ravel_multi_index((o_row,o_col,b_index),(self.board_size,self.board_size,self.bucket_space))

    def action_index_to_tuple(self, action):
        '''
            inputs : zero-index flattened action index
            output : zero-index action tuple (o_row, o_col, b_index)
        '''
        return np.unravel_index(action, (self.board_size, self.board_size, self.bucket_space))         # get zero-indexed row, col and bucket 

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
    
