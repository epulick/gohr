# Adapted from code by Shubham Bharti and Yiding Chen

import numpy as np
import os, sys
from gym import spaces
from rule_game_env import *

class NaiveBoard(RuleGameEnv):
   
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #
    #      This class constructs a memoryless, one hot representation of the board state 
    #
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def __init__(self, args):
        super(NaiveBoard, self).__init__(args)
        
        self.in_dim = self.board_size*self.board_size*(self.shape_space+self.color_space)
        self.out_dim = self.board_size*self.board_size*self.bucket_space
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #
    #   Create feature vector with 1's corresponding to objects on the board
    #
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def get_feature(self):
        feature_dict = {}
        mask = np.zeros(self.out_dim)
        inv_mask = np.ones(self.out_dim)
        features = np.zeros((self.board_size, self.board_size, self.shape_space+self.color_space))

        # Loop over the corresponding objects on the board (features are already initialized to zero otherwise)
        for object_tuple in self.board:
            # Extract information associated with current object
            o_row, o_col, o_color, o_shape = object_tuple['y'], object_tuple['x'], self.color_id[object_tuple['color']], self.shape_id[object_tuple['shape']]
            #print(o_row,o_col,o_shape,o_color)
            #breakpoint()
            for i in range(self.bucket_space):
                idx = np.ravel_multi_index((o_row-1,o_col-1,i),(self.board_size,self.board_size,self.bucket_space))
                mask[idx]=1
                inv_mask[idx]=0
            # Write out 1's for the objects shape and color (in the correct row,col position in the feature array)
            features[o_row-1][o_col-1][o_shape]=1
            features[o_row-1][o_col-1][self.shape_space+o_color]=1
        #print(features)
        features = features.flatten()
        for inv in self.move_list:
            mask[inv] = 0
            inv_mask[inv]=1
        feature_dict['features']=features
        feature_dict['mask']=inv_mask
        feature_dict['valid']=np.nonzero(mask)[0]
        return feature_dict

class NaiveBoard_N(RuleGameEnv):
   
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #
    #      This class constructs a one hot representation of the board state with n-steps of memory
    #
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def __init__(self, args):
        super(NaiveBoard_N, self).__init__(args)
        self.board_representation_size = self.board_size*self.board_size*(self.shape_space+self.color_space)
        self.out_dim = self.board_size*self.board_size*self.bucket_space
        self.in_dim = self.n_steps*(self.board_representation_size + self.out_dim) + self.board_representation_size
        #print("in_dim: ", self.in_dim)
        #print("out_dim: ", self.out_dim)
        #print("n_steps: ", self.n_steps)

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #
    #   Create feature vector with 1's corresponding to objects on the board
    #
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def get_feature(self):
        feature_dict = {}
        mask = np.zeros(self.out_dim)
        inv_mask = np.ones(self.out_dim)
        features = np.zeros((self.board_size,self.board_size,self.shape_space+self.color_space))
        step_features = np.zeros((self.board_size,self.board_size,self.shape_space+self.color_space))
        step_move = np.zeros(self.out_dim)

        # Loop over the corresponding objects on the board (features are already initialized to zero otherwise)
        for object_tuple in self.board:
            # Extract information associated with current object
            o_row, o_col, o_color, o_shape = object_tuple['y'], object_tuple['x'], self.color_id[object_tuple['color']], self.shape_id[object_tuple['shape']]
            #print(o_row,o_col,o_shape,o_color)
            #breakpoint()
            for i in range(self.bucket_space):
                idx = np.ravel_multi_index((o_row-1,o_col-1,i),(self.board_size,self.board_size,self.bucket_space))
                mask[idx]=1
                inv_mask[idx]=0
            # Write out 1's for the objects shape and color (in the correct row,col position in the feature array)
            features[o_row-1][o_col-1][o_shape]=1
            features[o_row-1][o_col-1][self.shape_space+o_color]=1
        features = features.flatten()
        for step in range(self.n_steps):
            if self.last_boards[step] is not None:
                for object_tuple in self.last_boards[step]:
                    o_row, o_col, o_color, o_shape = object_tuple['y'], object_tuple['x'], self.color_id[object_tuple['color']], self.shape_id[object_tuple['shape']]
                    step_features[o_row-1][o_col-1][o_shape]=1
                    step_features[o_row-1][o_col-1][self.shape_space+o_color]=1
            step_features = step_features.flatten()
            if self.last_moves[step] is not None:
                #if sum(x is not None for x in self.last_moves)==3:
                #    breakpoint()
                step_move[self.last_moves[step]] = 1
            features = np.concatenate((features,step_features,step_move),axis=0)
            step_features = np.zeros((self.board_size,self.board_size,self.shape_space+self.color_space))
            step_move = np.zeros(self.out_dim)
            #if sum(x is not None for x in self.last_moves)==3:
            #    breakpoint()
        for inv in self.move_list:
            mask[inv] = 0
            inv_mask[inv]=1
        feature_dict['features']=features
        feature_dict['mask']=inv_mask
        feature_dict['valid']=np.nonzero(mask)[0]
        #self.last_board = self.board
        return feature_dict   

class NaiveBoard_N_dense(RuleGameEnv):
   
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #
    #      This class constructs a one hot representation of the board state with n-steps of memory
    #
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def __init__(self, args):
        super(NaiveBoard_N_dense, self).__init__(args)
        #self.board_representation_size = self.board_size*self.board_size*(self.shape_space+self.color_space)
        self.out_dim = self.board_size*self.board_size*self.bucket_space
        self.dense_action_dim = self.board_size+self.board_size+self.bucket_space
        self.in_dim = self.n_steps*(self.color_space+self.shape_space + self.dense_action_dim) + self.board_size*self.board_size*(self.shape_space+self.color_space)
        #print("in_dim: ", self.in_dim)
        #print("out_dim: ", self.out_dim)
        #print("n_steps: ", self.n_steps)

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #
    #   Create feature vector with 1's corresponding to objects on the board
    #
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def get_feature(self):
        feature_dict = {}
        mask = np.zeros(self.out_dim)
        inv_mask = np.ones(self.out_dim)
        features = np.zeros((self.board_size,self.board_size,self.shape_space+self.color_space))
        step_features = np.zeros(self.shape_space+self.color_space)
        step_move = np.zeros(self.dense_action_dim)

        # Loop over the corresponding objects on the board (features are already initialized to zero otherwise)
        for object_tuple in self.board:
            # Extract information associated with current object
            o_row, o_col, o_color, o_shape = object_tuple['y'], object_tuple['x'], self.color_id[object_tuple['color']], self.shape_id[object_tuple['shape']]
            #print(o_row,o_col,o_shape,o_color)
            #breakpoint()
            for i in range(self.bucket_space):
                idx = np.ravel_multi_index((o_row-1,o_col-1,i),(self.board_size,self.board_size,self.bucket_space))
                mask[idx]=1
                inv_mask[idx]=0
            # Write out 1's for the objects shape and color (in the correct row,col position in the feature array)
            features[o_row-1][o_col-1][o_shape]=1
            features[o_row-1][o_col-1][self.shape_space+o_color]=1
        features = features.flatten()
        for step in range(self.n_steps):
            if self.last_attributes[step] is not None:
                #for object_tuple in self.last_boards[step]:
                #    o_row, o_col, o_color, o_shape = object_tuple['y'], object_tuple['x'], self.color_id[object_tuple['color']], self.shape_id[object_tuple['shape']]
                #    step_features[o_row-1][o_col-1][o_shape]=1
                #    step_features[o_row-1][o_col-1][self.shape_space+o_color]=1
                step_features[self.last_attributes[step][0]]=1
                step_features[self.last_attributes[step][1]+self.shape_space]=1
            if self.last_moves[step] is not None:
                #if sum(x is not None for x in self.last_moves)==3:
                #    breakpoint()
                row,col,bucket = self.action_index_to_tuple(self.last_moves[step])
                #row+=1
                #col+=1
                #bucket+=1
                step_move[row] = 1
                step_move[self.board_size+col]=1
                step_move[self.board_size+self.board_size+bucket]=1
            features = np.concatenate((features,step_features,step_move),axis=0)
            step_features = np.zeros(self.shape_space+self.color_space)
            step_move = np.zeros(self.dense_action_dim)
            #if sum(x is not None for x in self.last_moves)==3:
            #    breakpoint()
        for inv in self.move_list:
            mask[inv] = 0
            inv_mask[inv]=1
        feature_dict['features']=features
        feature_dict['mask']=inv_mask
        feature_dict['valid']=np.nonzero(mask)[0]
        #self.last_board = self.board
        return feature_dict 

def test_featurization(args):
    # Testing code for this level of abstraction - this may not be updated reliably
    #env = NaiveBoard(args)
    env = NaiveBoard_N(args)
    phi = env.get_feature()
    breakpoint()

if __name__ == "__main__":
    print("starting")
    #rule_dir_path, record = sys.argv[1], bool(int(sys.argv[2]))                 # directory path of rules is provided by the caller
    rule_dir_path, record = sys.argv[1], 0
    rule_name = 'rules-05.txt'                 # select the rule-name here
    rule_file_path = os.path.join(rule_dir_path, rule_name)

    args = {   
            'FINITE'  : True,                   # run till success, need not converge
            'NORMALIZE' : False,                # mean-variance normalize goto returns
            'RECORD' : record,                  # record data to neptune
            'SHAPING' : False,                  # use potential-shaped rewards
            'INIT_OBJ_COUNT'  : 1,             # initial number of objects on the board
            'R_ACCEPT' : -1,                    # reward for a reject move
            'R_REJECT' : -1,                    # reward for an accept move
            'TRAIN_HORIZON' : 200,              # horizon for each training episode
            'ALPHA' :   1.3,                    # success relaxation threshold parameter
            'TRAIN_EPISODES' :  1000,           # run this many training episodes if 'CONVERGE'==1
            'TEST_EPISODES' :  100,             # total test episode in each test trial, for success algorithm must clear each of these episode wihin
                                                
            'TEST_FREQ' :   1000,               # interval between the training episodes when a test trial is perfomed
            'VERBOSE' : 0,                      # for descriptive output, not implemented properly yet
            'LR' : 1e-2,                        # learning rate of the PG learner
            'GAMMA' : 1,                        # discount factor                 
            'RULE_FILE_PATH' : rule_file_path,  # full rule-file path      
            'RULE_NAME'  : rule_name,           # rule-name
            'BOARD_SIZE'  : 6,                  # for board of size 6x6
            'OBJECT_SPACE'  : 16,               # total distinct types of objects possible
            'COLOR_SPACE'  : 4,                 # total possible colors
            'SHAPE_SPACE'  : 4,                 # total possible shapes
            'BUCKET_SPACE'  : 4,                # total buckets
            'SEED' : -1,
            'RUN_MODE' : "RULE",
            'N_STEPS' : 1
        }

    test_featurization(args)
