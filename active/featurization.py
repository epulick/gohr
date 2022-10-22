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

        # PENDING MODIFICATION
        # defining the feature dimension - circle and bucket-4 indicators removed to avoid over parametrization
        self.action_feature_dim = self.shape_space+self.color_space+self.bucket_space-2    # 10
        
        self.in_dim = self.board_size*self.board_size*(self.shape_space+self.color_space)
        #self.in_dim = self.board_size*self.board_size*self.shape_space
        self.out_dim = self.board_size*self.board_size*self.bucket_space

        # PENDING MODIFICATION
        # define observation space of the model, in our case it corresponds to the feature dimension(which is 10 dimensional in this example) for each action
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.board_size*self.board_size*self.bucket_space,self.action_feature_dim), dtype=int)
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
        #features = np.zeros((self.board_size,self.board_size,self.shape_space))

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
        #return features.flatten()

class NaiveBoard_m1(RuleGameEnv):
   
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #
    #      This class constructs a memoryless, one hot representation of the board state 
    #
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def __init__(self, args):
        super(NaiveBoard_m1, self).__init__(args)

        # PENDING MODIFICATION
        # defining the feature dimension - circle and bucket-4 indicators removed to avoid over parametrization
        self.action_feature_dim = self.shape_space+self.color_space+self.bucket_space-2    # 10
        
        #self.in_dim = self.board_size*self.board_size*self.shape_space
        self.offset = self.board_size*self.board_size*(self.shape_space+self.color_space)
        self.out_dim = self.board_size*self.board_size*self.bucket_space
        self.in_dim = (self.offset)*2 + self.out_dim
    
        # PENDING MODIFICATION
        # define observation space of the model, in our case it corresponds to the feature dimension(which is 10 dimensional in this example) for each action
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.board_size*self.board_size*self.bucket_space,self.action_feature_dim), dtype=int)
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #
    #   Create feature vector with 1's corresponding to objects on the board
    #
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def get_feature(self):
        feature_dict = {}
        mask = np.zeros(self.out_dim)
        inv_mask = np.ones(self.out_dim)
        features = np.zeros((self.board_size,self.board_size,self.shape_space+self.color_space,2))
        #features = np.zeros((self.board_size,self.board_size,self.shape_space))

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
            features[o_row-1][o_col-1][o_shape][0]=1
            features[o_row-1][o_col-1][self.shape_space+o_color][0]=1
        if self.last_board is not None:
            for object_tuple in self.last_board:
                o_row, o_col, o_color, o_shape = object_tuple['y'], object_tuple['x'], self.color_id[object_tuple['color']], self.shape_id[object_tuple['shape']]
                features[o_row-1][o_col-1][o_shape][1]=1
                features[o_row-1][o_col-1][self.shape_space+o_color][1]=1
        #print(features)
        features = features.flatten()
        m1 = np.zeros(self.out_dim)
        if self.m1 is not None:
            m1[self.m1]=1
        #breakpoint()
        features=np.concatenate((features,m1),axis=0)
        for inv in self.move_list:
            mask[inv] = 0
            inv_mask[inv]=1
        feature_dict['features']=features
        feature_dict['mask']=inv_mask
        feature_dict['valid']=np.nonzero(mask)[0]
        self.last_board = self.board
        return feature_dict    
        #return features.flatten()

def test_featurization(args):
    # some testing code
    env = NaiveBoard(args)
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
            'RUN_MODE' : "RULE"
        }

    test_featurization(args)
