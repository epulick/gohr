# Parent class to env - contains code for different featurizations
# Originally written by Shubham Bharti and Yiding Chen

import numpy as np
import os, sys
from gym import spaces
from rule_game_env import *

'''
    Module description : This module contains featurization classes that builds on top of RuleGameEnv class and defines different types of featurizations.
    The set of features have kept on increasing since the beginning of the project to handle more and more complex rules as so are the classes defined in this module.
'''

class RuleUnary(RuleGameEnv):

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #
    #      This class constructs all three types of unary indicator features with no overparameterization    
    #
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def __init__(self, args):
        super(RuleUnary, self).__init__(args)

        # defining the feature dimension - circle and bucket-4 indicators removed to avoid over parametrization
        self.action_feature_dim = self.shape_space+self.color_space+self.bucket_space-2    # 10

        # define observation space of the model, in our case it corresponds to the feature dimension(which is 10 dimensional in this example) for each action
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.board_size*self.board_size*self.bucket_space,self.action_feature_dim), dtype=np.int)

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #
    #   To extract feature vector [shape = {star, square, triangle}, color = {red, blue, black, yellow}, bucket = {bucket-1, bucket-2, bucket-3}] for each active positions on the board
    #
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def get_feature(self):

        features = np.zeros((self.board_size, self.board_size, self.bucket_space, self.action_feature_dim))             # feature matrix for all possible actions
        shape_io, color_io, bucket_io = 0, self.shape_space-1, self.shape_space-1+self.color_space                      # beginning index for different types of features

        # extract features only for active action(ones with some object), equivalent to conjucting the feature with I[there is an object of the position]
        for object_tuple in self.board:
            o_row, o_col, o_color, o_shape = object_tuple['y'], object_tuple['x'], self.color_id[object_tuple['color']], self.shape_id[object_tuple['shape']]

            for bucket in range(self.bucket_space):
                if(o_shape<=2):
                    features[o_row-1][o_col-1][bucket][shape_io+o_shape] = 1
                features[o_row-1][o_col-1][bucket][color_io+o_color] = 1

        for bucket in range(self.bucket_space-1):
            features[:,:,bucket,bucket_io+bucket] = 1
        
        return features.flatten()



class RuleUnaryBinary(RuleGameEnv):

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #
    #      This class constructs all three types of unary features and three types of binary indicator features    
    #
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def __init__(self, args):
        super(RuleUnaryBinary_TEST, self).__init__(args)
        unary_dim = self.shape_space+self.color_space+self.bucket_space
        binary_dim = self.shape_space*self.color_space+self.shape_space*self.bucket_space+ self.color_space*self.bucket_space   
        self.action_feature_dim = unary_dim + binary_dim             # 3 x 4 + 3 x 4 x 4 = 60

        # define observation space of the model, in our case it corresponds to the feature dimension(which is 64 dimensional in this example) for each action
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.board_size*self.board_size*self.bucket_space,self.action_feature_dim), dtype=np.int)
        
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #
    #   To extract feature vector ( shape, color, bucket, shape x color, shape x bucket, color x bucket ) for each action
    #
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def get_feature(self):
        features = np.zeros((self.board_size, self.board_size, self.bucket_space, self.action_feature_dim))
        
        for object_tuple in self.board:
            o_row, o_col = object_tuple['y']-1, object_tuple['x']-1     # cgs index start with (1,1) so subtract 1
            o_color, o_shape = self.color_id[object_tuple['color']], self.shape_id[object_tuple['shape']]

            color_i0 = self.shape_space      # color start index in the feature vector
            bucket_i0 = self.shape_space+self.color_space     # bucket start index in the feature vector
            shape_color_i0 = self.shape_space + self.color_space + self.bucket_space    # (shape, color) start index
            shape_bucket_i0 = shape_color_i0 + self.shape_space*self.color_space        # (shape, bucekt) start index
            color_bucket_i0 = shape_bucket_i0 + self.shape_space*self.bucket_space      # (color, bucket) start index
            
            for bucket in range(self.bucket_space):
                features[o_row][o_col][bucket][o_shape] = 1
                features[o_row][o_col][bucket][color_i0+o_color] = 1
                features[:,:,bucket, bucket_i0+bucket] = 1

                features[o_row][o_col][bucket][shape_color_i0+o_shape*self.shape_space+o_color] = 1
                features[o_row][o_col][bucket][shape_bucket_i0+o_shape*self.shape_space+bucket] = 1
                features[o_row][o_col][bucket][color_bucket_i0+o_color*self.color_space+bucket] = 1

        return features.flatten()

class ColorMatch(RuleGameEnv):

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #
    #      This class constructs all three types of unary features and three types of binary indicator features    
    #
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def __init__(self, args):
        super(ColorMatch, self).__init__(args)
        self.action_feature_dim = self.color_space*self.bucket_space             # 4 x 4 = 16

        # define observation space of the model, in our case it corresponds to the feature dimension(which is 64 dimensional in this example) for each action
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.board_size*self.board_size*self.bucket_space,self.action_feature_dim), dtype=np.int)
        
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #
    #   To extract feature vector ( shape, color, bucket, shape x color, shape x bucket, color x bucket ) for each action
    #
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def get_feature(self):
        features = np.zeros((self.board_size, self.board_size, self.bucket_space, self.action_feature_dim))
        
        for object_tuple in self.board:
            o_row, o_col = object_tuple['y']-1, object_tuple['x']-1     # cgs index start with (1,1) so subtract 1
            o_color, o_shape = self.color_id[object_tuple['color']], self.shape_id[object_tuple['shape']]
            
            for bucket in range(self.bucket_space):
                features[o_row][o_col][bucket][o_color*self.color_space+bucket] = 1

        return features.flatten()

class ShapeMatch(RuleGameEnv):

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #
    #      This class constructs all three types of unary features and three types of binary indicator features    
    #
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def __init__(self, args):
        super(ShapeMatch, self).__init__(args)
        self.action_feature_dim = self.shape_space*self.bucket_space             # 4 x 4 = 16

        # define observation space of the model, in our case it corresponds to the feature dimension(which is 64 dimensional in this example) for each action
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.board_size*self.board_size*self.bucket_space,self.action_feature_dim), dtype=np.int)
        
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #
    #   To extract feature vector ( shape, color, bucket, shape x color, shape x bucket, color x bucket ) for each action
    #
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def get_feature(self):
        features = np.zeros((self.board_size, self.board_size, self.bucket_space, self.action_feature_dim))
        
        for object_tuple in self.board:
            o_row, o_col = object_tuple['y']-1, object_tuple['x']-1     # cgs index start with (1,1) so subtract 1
            o_color, o_shape = self.color_id[object_tuple['color']], self.shape_id[object_tuple['shape']]
            
            for bucket in range(self.bucket_space):
                features[o_row][o_col][bucket][o_shape*self.color_space+bucket] = 1

        return features.flatten()        

class RuleUnaryBinary_TEST(RuleGameEnv):

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #
    #      This class constructs all three types of unary features and three types of binary indicator features    
    #
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def __init__(self, args):
        super(RuleUnaryBinary, self).__init__(args)
        unary_dim = self.shape_space+self.color_space+self.bucket_space
        binary_dim = self.shape_space*self.color_space+self.shape_space*self.bucket_space+ self.color_space*self.bucket_space   
        self.action_feature_dim = unary_dim + binary_dim             # 3 x 4 + 3 x 4 x 4 = 60

        # define observation space of the model, in our case it corresponds to the feature dimension(which is 64 dimensional in this example) for each action
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.board_size*self.board_size*self.bucket_space,self.action_feature_dim), dtype=np.int)
        
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #
    #   To extract feature vector ( shape, color, bucket, shape x color, shape x bucket, color x bucket ) for each action
    #
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def get_feature(self):
        features = np.zeros((self.board_size, self.board_size, self.bucket_space, self.action_feature_dim))
        
        for object_tuple in self.board:
            o_row, o_col = object_tuple['y']-1, object_tuple['x']-1     # cgs index start with (1,1) so subtract 1
            o_color, o_shape = self.color_id[object_tuple['color']], self.shape_id[object_tuple['shape']]

            color_i0 = self.shape_space      # color start index in the feature vector
            bucket_i0 = self.shape_space+self.color_space     # bucket start index in the feature vector
            shape_color_i0 = self.shape_space + self.color_space + self.bucket_space    # (shape, color) start index
            shape_bucket_i0 = shape_color_i0 + self.shape_space*self.color_space        # (shape, bucekt) start index
            color_bucket_i0 = shape_bucket_i0 + self.shape_space*self.bucket_space      # (color, bucket) start index
            
            for bucket in range(self.bucket_space):
                features[o_row][o_col][bucket][o_shape] = 1
                features[o_row][o_col][bucket][color_i0+o_color] = 1
                features[:,:,bucket, bucket_i0+bucket] = 1

                features[o_row][o_col][bucket][shape_color_i0+o_color*self.shape_space+o_shape] = 1
                features[o_row][o_col][bucket][shape_bucket_i0+bucket*self.shape_space+o_shape] = 1
                features[o_row][o_col][bucket][color_bucket_i0+o_color*self.color_space+bucket] = 1

        return features.flatten()



class RuleOneStepUnary(RuleGameEnv):

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #
    #   Initialize backend engine and setup action and observation space
    #   Uses all Unary and Binary features on color, shape and bucket. Runs on all Rule 5/6/7 
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def __init__(self, args):
        super(RuleOneStepUnary, self).__init__(args)
        s, c, b = self.shape_space, self.color_space, self.bucket_space
        unary_dim = s+c+b
        one_step_unary_dim = (s+1)*s + (c+1)*c + (b+1)*b
        self.action_feature_dim = unary_dim + one_step_unary_dim      # 3 x 4 + 3 x 20 = 72
        
        # define observation space of the model, in our case it corresponds to the feature dimension(which is 72 dimensional in this example) for each action
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.board_size*self.board_size*self.bucket_space,self.action_feature_dim), dtype=np.int)
        
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #
    #   Extract feature vector ( shape, color, bucket, shape x color, shape x bucket, color x bucket )
    #
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def get_feature(self):
        features = np.zeros((self.board_size, self.board_size, self.bucket_space, self.action_feature_dim))
        
        # extract features only for active action(ones with some object), equivalent to conjucting the feature with I[there is an object of the position]
        for object_tuple in self.board:
            o_row, o_col = object_tuple['y']-1, object_tuple['x']-1   
            o_color, o_shape = self.color_id[object_tuple['color']], self.shape_id[object_tuple['shape']]

            color_i0 = self.shape_space                                                         # color start index in the feature vector
            bucket_i0 = self.shape_space+self.color_space                                       # bucket start index in the feature vector
            #lc_shape_i0 = bucket_i0 + self.color_space*self.bucket_space                        # (last_shape, current_shape) start index
            lc_shape_i0 = bucket_i0 + self.bucket_space                        # (last_shape, current_shape) start index
            lc_color_i0 = lc_shape_i0 + (self.shape_space+1)*self.shape_space                   # (last_shape, current_shape) start index
            lc_bucket_i0 = lc_color_i0 + (self.color_space+1)*self.color_space                  # (last_shape, current_shape) start index
            
            for bucket in range(self.bucket_space):
                features[o_row][o_col][bucket][o_shape] = 1
                features[o_row][o_col][bucket][color_i0+o_color] = 1
                #features[:,:,bucket, bucket_i0+bucket] = 1
                features[o_row,o_col,bucket, bucket_i0+bucket] = 1

                features[o_row][o_col][bucket][lc_shape_i0+(self.l_shape+1)*(self.shape_space)+o_shape] = 1
                features[o_row][o_col][bucket][lc_color_i0+(self.l_color+1)*(self.color_space)+o_color] = 1
                features[o_row][o_col][bucket][lc_bucket_i0+(self.l_bucket+1)*(self.bucket_space)+bucket] = 1

        return features.flatten()


class RuleOneStepUnaryBinary(RuleGameEnv):

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #
    #      This class constructs 3 types of unary features, 3 types of binary features and their last step and current step composition(cross one step binary features {exp : sc x sb} is not considered)
    #
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def __init__(self, args):
        super(RuleOneStepUnaryBinary, self).__init__(args)
        s, c, b = self.shape_space, self.color_space, self.bucket_space
        unary_dim = s+c+b
        binary_dim = s*c + s*b + c*b   
        one_step_unary_dim = (s+1)*s + (c+1)*c + (b+1)*b
        one_step_binary_dim = (s+1)*(c+1)*s*c + (s+1)*(b+1)*s*b + (c+1)*(b+1)*c*b
        self.action_feature_dim = unary_dim + binary_dim + one_step_unary_dim + one_step_binary_dim        # 3 x 4 + 3 x 16 + 3 x 20 + 3 x 400 = 1320


        # define observation space of the model, in our case it corresponds to the feature dimension(which is 1320 dimensional in this example) for each action
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.board_size*self.board_size*self.bucket_space,self.action_feature_dim), dtype=np.int)
        
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #
    #   Extract feature vector described in the constructor
    #
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def get_feature(self):
        features = np.zeros((self.board_size, self.board_size, self.bucket_space, self.action_feature_dim))
        
        for object_tuple in self.board:
            o_row, o_col = object_tuple['y']-1, object_tuple['x']-1                                            
            o_color, o_shape = self.color_id[object_tuple['color']], self.shape_id[object_tuple['shape']]

            color_i0 = self.shape_space                                                                                                     # color start index in the feature vector
            bucket_i0 = self.shape_space+self.color_space                                                                                   # bucket start index in the feature vector
            
            shape_color_i0 = self.shape_space + self.color_space + self.bucket_space                                                        # (shape, color) start index
            shape_bucket_i0 = shape_color_i0 + self.shape_space*self.color_space                                                            # (shape, bucekt) start index
            color_bucket_i0 = shape_bucket_i0 + self.shape_space*self.bucket_space                                                          # (color, bucket) start index
            
            lc_shape_i0 = color_bucket_i0 + self.color_space*self.bucket_space                                                              # (last_shape, current_shape) start index
            lc_color_i0 = lc_shape_i0 + (self.shape_space+1)*self.shape_space                                                               # (last_shape, current_shape) start index
            lc_bucket_i0 = lc_color_i0 + (self.color_space+1)*self.color_space                                                              # (last_shape, current_shape) start index

            lc_shape_color_i0 = lc_bucket_i0 + (self.bucket_space+1)*self.bucket_space                                                      # (last_shape, current_shape) start index
            lc_shape_bucket_i0 = lc_shape_color_i0 + (self.shape_space+1)*(self.color_space+1)*self.shape_space*self.color_space            # (last_shape, current_shape) start index
            lc_color_bucket_i0 = lc_shape_bucket_i0 + (self.shape_space+1)*(self.bucket_space+1)*self.shape_space*self.bucket_space         # (last_shape, current_shape) start index

            
            for bucket in range(self.bucket_space):
                ## unaries
                features[o_row][o_col][bucket][o_shape] = 1
                features[o_row][o_col][bucket][color_i0+o_color] = 1
                features[:,:,bucket, bucket_i0+bucket] = 1

                ## binaries
                features[o_row][o_col][bucket][shape_color_i0+o_shape*self.color_space+o_color] = 1
                features[o_row][o_col][bucket][shape_bucket_i0+o_shape*self.bucket_space+bucket] = 1
                features[o_row][o_col][bucket][color_bucket_i0+o_color*self.bucket_space+bucket] = 1
                
                ## one_step unaries
                features[o_row][o_col][bucket][lc_shape_i0+(self.l_shape+1)*(self.shape_space)+o_shape] = 1
                features[o_row][o_col][bucket][lc_color_i0+(self.l_color+1)*(self.color_space)+o_color] = 1
                features[o_row][o_col][bucket][lc_bucket_i0+(self.l_bucket+1)*(self.bucket_space)+bucket] = 1

                ## one_step binaries
                features[o_row][o_col][bucket][lc_shape_color_i0+(self.l_shape+1)*(self.color_space+1)*self.shape_space*self.color_space+
                                                                 (self.l_color+1)*self.shape_space*self.color_space+
                                                                 o_shape*self.shape_space+
                                                                 o_color] = 1

                features[o_row][o_col][bucket][lc_shape_bucket_i0+(self.l_shape+1)*(self.bucket_space+1)*self.shape_space*self.bucket_space+
                                                                  (self.l_bucket+1)*self.shape_space*self.bucket_space+
                                                                 o_shape*self.bucket_space+
                                                                 bucket] = 1

                features[o_row][o_col][bucket][lc_color_bucket_i0+(self.l_color+1)*(self.bucket_space+1)*self.color_space*self.bucket_space+
                                                                  (self.l_bucket+1)*self.color_space*self.bucket_space+
                                                                 o_color*self.bucket_space+
                                                                 bucket] = 1

        return features.flatten()


class RuleOneStepUnaryBinaryWithoutIndex(RuleGameEnv):

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #
    #      This class constructs 3 types of unary features, 3 types of binary features and their last step and current step composition(cross one step binary is used)
    #
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    def __init__(self, args):
        super(RuleOneStepUnaryBinaryWithoutIndex, self).__init__(args)
        s, c, b, i = self.shape_space, self.color_space, self.bucket_space, self.index_space
        unary_dim = s+c+b
        binary_dim = s*c + s*b + c*b 
        one_step_unary_dim = (s+1)*s + (c+1)*c + (b+1)*b
        one_step_binary_dim = (s+1)*(c+1)*(s*c + s*b + c*b) + (s+1)*(b+1)*(s*c + s*b + c*b) + (c+1)*(b+1)*(s*c + s*b + c*b)
        self.action_feature_dim = unary_dim + binary_dim + one_step_unary_dim + one_step_binary_dim        # 3 * 4 + 3 * 16 + 3 * 20 + 9 * 400 = 3720

        # define observation space of the model, in our case it corresponds to the feature dimension(which is 3720 dimensional in this example) for each action
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.board_size*self.board_size*self.bucket_space,self.action_feature_dim), dtype=np.int)


    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #
    #   Extract feature vector ( unary, binary, last-current-unary, last-current-unary-binary )
    #
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def get_feature(self):
        features = np.zeros((self.board_size, self.board_size, self.bucket_space, self.action_feature_dim))
        for object_tuple in self.board:
            o_row, o_col = object_tuple['y']-1, object_tuple['x']-1     # cgs index start with (1,1) so subtract 1
            o_color, o_shape = self.color_id[object_tuple['color']], self.shape_id[object_tuple['shape']]

            s_i0 = 0                                                                                                                # start index of current shape
            c_i0 = s_i0 + self.shape_space                                                                                          # start index of current color
            b_i0 = c_i0 + self.color_space                                                                                          # start index of current bucket
            
            sc_i0 = b_i0 + self.bucket_space                                                                                        # start index (current shape, current color) 
            sb_i0 = sc_i0 + self.shape_space*self.color_space                                                                       # start index (current shape, current bucekt)
            cb_i0 = sb_i0 + self.shape_space*self.bucket_space                                                                      # start index (current color, current bucket)
            
            lc_s_i0 = cb_i0 + self.color_space*self.bucket_space                                                                    # start index (last_shape, current_shape) 
            lc_c_i0 = lc_s_i0 + (self.shape_space+1)*self.shape_space                                                               # start index (last_color, current_color)
            lc_b_i0 = lc_c_i0 + (self.color_space+1)*self.color_space                                                               # start index (last_bucket, current_bucket)

            lc_sc_sc_i0 = lc_b_i0 + (self.bucket_space+1)*self.bucket_space                                                         # (last_shape, last_color, current_shape, current_color) sc_sc in the lhs name stands for this 4-tuple
            lc_sc_sb_i0 = lc_sc_sc_i0 + (self.shape_space+1)*(self.color_space+1)*self.shape_space*self.color_space                 # (last_shape, last_color, current_shape, current_bucket) sc_sb in the lhs name stands for this 4-tuple and so on for other features.
            lc_sc_cb_i0 = lc_sc_sb_i0 + (self.shape_space+1)*(self.color_space+1)*self.shape_space*self.bucket_space

            lc_sb_sc_i0 = lc_sc_cb_i0 + (self.shape_space+1)*(self.color_space+1)*self.color_space*self.bucket_space
            lc_sb_sb_i0 = lc_sb_sc_i0 + (self.shape_space+1)*(self.bucket_space+1)*self.shape_space*self.color_space
            lc_sb_cb_i0 = lc_sb_sb_i0 + (self.shape_space+1)*(self.bucket_space+1)*self.shape_space*self.bucket_space

            lc_cb_sc_i0 = lc_sb_cb_i0 + (self.shape_space+1)*(self.bucket_space+1)*self.color_space*self.bucket_space
            lc_cb_sb_i0 = lc_cb_sc_i0 + (self.color_space+1)*(self.bucket_space+1)*self.shape_space*self.color_space
            lc_cb_cb_i0 = lc_cb_sb_i0 + (self.color_space+1)*(self.bucket_space+1)*self.shape_space*self.bucket_space

            
            for o_bucket in range(self.bucket_space):
                ## unaries
                features[o_row][o_col][o_bucket][s_i0+o_shape] = 1
                features[o_row][o_col][o_bucket][c_i0+o_color] = 1
                features[:,:,o_bucket, b_i0+o_bucket] = 1

                ## binaries
                features[o_row][o_col][o_bucket][sc_i0+o_shape*self.color_space+o_color] = 1
                features[o_row][o_col][o_bucket][sb_i0+o_shape*self.bucket_space+o_bucket] = 1
                features[o_row][o_col][o_bucket][cb_i0+o_color*self.bucket_space+o_bucket] = 1
                
                ## one_step unaries
                features[o_row][o_col][o_bucket][lc_s_i0+(self.l_shape+1)*(self.shape_space)+o_shape] = 1
                features[o_row][o_col][o_bucket][lc_c_i0+(self.l_color+1)*(self.color_space)+o_color] = 1
                features[o_row][o_col][o_bucket][lc_b_i0+(self.l_bucket+1)*(self.bucket_space)+o_bucket] = 1

                ## one_step binaries
                features[o_row][o_col][o_bucket][lc_sc_sc_i0+ (self.l_shape+1)*(self.color_space+1)*self.shape_space*self.color_space+
                                                            (self.l_color+1)*self.shape_space*self.color_space+
                                                            o_shape*self.color_space+
                                                            o_color] = 1
                
                features[o_row][o_col][o_bucket][lc_sc_sb_i0+ (self.l_shape+1)*(self.color_space+1)*self.shape_space*self.bucket_space+
                                                            (self.l_color+1)*self.shape_space*self.bucket_space+
                                                            o_shape*self.shape_space+
                                                            o_bucket] = 1

                features[o_row][o_col][o_bucket][lc_sc_cb_i0+ (self.l_shape+1)*(self.color_space+1)*self.color_space*self.bucket_space+
                                                            (self.l_color+1)*self.color_space*self.bucket_space+
                                                            o_color*self.bucket_space+
                                                            o_bucket] = 1
                
                features[o_row][o_col][o_bucket][lc_sb_sc_i0+ (self.l_shape+1)*(self.bucket_space+1)*self.shape_space*self.color_space+
                                                            (self.l_bucket+1)*self.shape_space*self.color_space+
                                                            o_shape*self.color_space+
                                                            o_color] = 1
                
                features[o_row][o_col][o_bucket][lc_sb_sb_i0+ (self.l_shape+1)*(self.bucket_space+1)*self.shape_space*self.bucket_space+
                                                            (self.l_bucket+1)*self.shape_space*self.bucket_space+
                                                            o_shape*self.bucket_space+
                                                            o_bucket] = 1

                features[o_row][o_col][o_bucket][lc_sb_cb_i0+ (self.l_shape+1)*(self.bucket_space+1)*self.color_space*self.bucket_space+
                                                            (self.l_bucket+1)*self.color_space*self.bucket_space+
                                                            o_color*self.bucket_space+
                                                            o_bucket] = 1

                features[o_row][o_col][o_bucket][lc_cb_sc_i0+ (self.l_color+1)*(self.bucket_space+1)*self.shape_space*self.color_space+
                                                            (self.l_bucket+1)*self.shape_space*self.color_space+
                                                            o_shape*self.color_space+
                                                            o_color] = 1
                
                features[o_row][o_col][o_bucket][lc_cb_sb_i0+ (self.l_color+1)*(self.bucket_space+1)*self.shape_space*self.bucket_space+
                                                            (self.l_bucket+1)*self.shape_space*self.bucket_space+
                                                            o_shape*self.bucket_space+
                                                            o_bucket] = 1

                features[o_row][o_col][o_bucket][lc_cb_cb_i0+ (self.l_color+1)*(self.bucket_space+1)*self.color_space*self.bucket_space+
                                                            (self.l_bucket+1)*self.color_space*self.bucket_space+
                                                            o_color*self.bucket_space+
                                                            o_bucket] = 1
        return features.flatten()


class RuleOneStepUnaryBinaryWithIndex(RuleGameEnv):

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #
    #      This class constructs the following feature :
    #          - all unary features : color, shape, bucket, index
    #          - all binary features : shape x color, shape x bucket, shape x index, color x bucket, color x index, bucket x index
    #          - one step unary and one step binary - formed from the cross of only unary features
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    def __init__(self, args):
        super(RuleOneStepUnaryBinaryWithIndex, self).__init__(args)
        s, c, b, i = self.shape_space, self.color_space, self.bucket_space, self.board_size*self.board_size
        unary_dim = s+c+b+i                                                                                                                 # 4+4+4+36 = 48
        binary_dim = s*c + s*b + s*i + c*b  + c*i + b*i                                                                                     # 4*4 + 4*4 + 4*36 + 4*4 + 4*36 + 4*36 =  480
        one_step_unary_dim = (s+1)*s + (c+1)*c + (b+1)*b + (i+1)*i                                                                          #  5*4 + 5*4 + 5*4 + 37*36 = 1392
        one_step_binary_dim = (s+1)*(c+1)*(s*c + s*b + c*b) + (s+1)*(b+1)*(s*c + s*b + c*b) + (c+1)*(b+1)*(s*c + s*b + c*b)                 # 5*5*(4*4+4*4+4*4)*3 = 3600
        one_step_binary_dim += (i+1)*(s+1)*i*s + (i+1)*(c+1)*i*c + (i+1)*(b+1)*i*b                                                          # 37*5*(36*4)*3 = 79920

        self.action_feature_dim = unary_dim + binary_dim + one_step_unary_dim + one_step_binary_dim                                         # total := 85440

        if(self.verbose==0):
            print("Total number of features : ", self.action_feature_dim)
            
        # define observation space of the model, in our case it corresponds to the feature dimension(which is 85440 dimensional in this example) for each action
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.board_size*self.board_size*self.bucket_space,self.action_feature_dim), dtype=np.int)
        
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #
    #   Extract feature vector defined in the constructor above
    #
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def get_feature(self):
        features = np.zeros((self.board_size, self.board_size, self.bucket_space, self.action_feature_dim))
        
        for object_tuple in self.board:
            o_row, o_col = object_tuple['y']-1, object_tuple['x']-1 
            o_color, o_shape, o_index = self.color_id[object_tuple['color']], self.shape_id[object_tuple['shape']], o_row*self.board_size + o_col
            
            s_i0 = 0
            c_i0 = s_i0 + self.shape_space                                                                                  # color start index in the feature vector
            i_i0 = c_i0 + self.index_space                                                                                  # bucket start index in the feature vector
            b_i0 = i_i0 + self.color_space                                                                                  # bucket start index in the feature vector
            

            sc_i0 = b_i0 + self.bucket_space                                                                                # start index (current shape, current color) 
            sb_i0 = sc_i0 + self.shape_space*self.color_space                                                               # start index (current shape, current bucekt)
            si_i0 = sb_i0 + self.shape_space*self.bucket_space
            cb_i0 = si_i0 + self.shape_space*self.index_space                                                               # start index (current color, current bucket)
            ci_i0 = cb_i0 + self.color_space*self.bucket_space
            bi_i0 = ci_i0 + self.color_space*self.index_space


            lc_s_i0 = bi_i0 + self.bucket_space*self.index_space                                                            # start index (last_shape, current_shape) 
            lc_c_i0 = lc_s_i0 + (self.shape_space+1)*self.shape_space                                                       # start index (last_color, current_color)
            lc_b_i0 = lc_c_i0 + (self.color_space+1)*self.color_space                                                       # start index (last_bucket, current_bucket)
            lc_i_i0 = lc_b_i0 + (self.bucket_space+1)*self.bucket_space                                                     # start index (last_bucket, current_bucket)

            lc_sc_sc_i0 = lc_i_i0 + (self.index_space+1)*self.index_space                                                   # (last_shape, last_color, current_shape, current_color) sc_sc in the lhs name stands for this 4-tuple
            lc_sc_sb_i0 = lc_sc_sc_i0 + (self.shape_space+1)*(self.color_space+1)*self.shape_space*self.color_space         # (last_shape, last_color, current_shape, current_bucket) sc_sb in the lhs name stands for this 4-tuple and so on for other features.
            lc_sc_cb_i0 = lc_sc_sb_i0 + (self.shape_space+1)*(self.color_space+1)*self.shape_space*self.bucket_space

            lc_sb_sc_i0 = lc_sc_cb_i0 + (self.shape_space+1)*(self.color_space+1)*self.color_space*self.bucket_space
            lc_sb_sb_i0 = lc_sb_sc_i0 + (self.shape_space+1)*(self.bucket_space+1)*self.shape_space*self.color_space
            lc_sb_cb_i0 = lc_sb_sb_i0 + (self.shape_space+1)*(self.bucket_space+1)*self.shape_space*self.bucket_space

            lc_cb_sc_i0 = lc_sb_cb_i0 + (self.shape_space+1)*(self.bucket_space+1)*self.color_space*self.bucket_space
            lc_cb_sb_i0 = lc_cb_sc_i0 + (self.color_space+1)*(self.bucket_space+1)*self.shape_space*self.color_space
            lc_cb_cb_i0 = lc_cb_sb_i0 + (self.color_space+1)*(self.bucket_space+1)*self.shape_space*self.bucket_space

            lc_is_is_i0 = lc_cb_cb_i0 + (self.color_space+1)*(self.bucket_space+1)*self.color_space*self.bucket_space
            lc_ic_ic_i0 = lc_is_is_i0 + (self.index_space+1)*(self.shape_space+1)*self.index_space*self.shape_space
            lc_ib_ib_i0 = lc_ic_ic_i0 + (self.index_space+1)*(self.color_space+1)*self.index_space*self.color_space

            
            for o_bucket in range(self.bucket_space):

                ## unary features
                features[o_row][o_col][o_bucket][s_i0+o_shape] = 1
                features[o_row][o_col][o_bucket][c_i0+o_color] = 1
                features[o_row][o_col][o_bucket][i_i0+o_index] = 1
                features[:,:,o_bucket, b_i0+o_bucket] = 1

                ## add binary features
                features[o_row][o_col][o_bucket][sc_i0+o_shape*self.color_space+o_color] = 1
                features[o_row][o_col][o_bucket][sb_i0+o_shape*self.bucket_space+o_bucket] = 1
                features[o_row][o_col][o_bucket][si_i0+o_shape*self.index_space+o_index] = 1
                features[o_row][o_col][o_bucket][cb_i0+o_color*self.bucket_space+o_bucket] = 1
                features[o_row][o_col][o_bucket][ci_i0+o_color*self.index_space+o_index] = 1
                features[o_row][o_col][o_bucket][bi_i0+o_bucket*self.index_space+o_index] = 1

                ## one_step unaries
                features[o_row][o_col][o_bucket][lc_s_i0+(self.l_shape+1)*(self.shape_space)+o_shape] = 1
                features[o_row][o_col][o_bucket][lc_c_i0+(self.l_color+1)*(self.color_space)+o_color] = 1
                features[o_row][o_col][o_bucket][lc_b_i0+(self.l_bucket+1)*(self.bucket_space)+o_bucket] = 1
                features[o_row][o_col][o_bucket][lc_i_i0+(self.l_index+1)*(self.index_space)+o_index] = 1

                ## one_step binaries
                features[o_row][o_col][o_bucket][lc_sc_sc_i0+ (self.l_shape+1)*(self.color_space+1)*self.shape_space*self.color_space+
                                                            (self.l_color+1)*self.shape_space*self.color_space+
                                                            o_shape*self.color_space+
                                                            o_color] = 1
                
                features[o_row][o_col][o_bucket][lc_sc_sb_i0+ (self.l_shape+1)*(self.color_space+1)*self.shape_space*self.bucket_space+
                                                            (self.l_color+1)*self.shape_space*self.bucket_space+
                                                            o_shape*self.shape_space+
                                                            o_bucket] = 1

                features[o_row][o_col][o_bucket][lc_sc_cb_i0+ (self.l_shape+1)*(self.color_space+1)*self.color_space*self.bucket_space+
                                                            (self.l_color+1)*self.color_space*self.bucket_space+
                                                            o_color*self.bucket_space+
                                                            o_bucket] = 1
                
                features[o_row][o_col][o_bucket][lc_sb_sc_i0+ (self.l_shape+1)*(self.bucket_space+1)*self.shape_space*self.color_space+
                                                            (self.l_bucket+1)*self.shape_space*self.color_space+
                                                            o_shape*self.color_space+
                                                            o_color] = 1
                
                features[o_row][o_col][o_bucket][lc_sb_sb_i0+ (self.l_shape+1)*(self.bucket_space+1)*self.shape_space*self.bucket_space+
                                                            (self.l_bucket+1)*self.shape_space*self.bucket_space+
                                                            o_shape*self.bucket_space+
                                                            o_bucket] = 1

                features[o_row][o_col][o_bucket][lc_sb_cb_i0+ (self.l_shape+1)*(self.bucket_space+1)*self.color_space*self.bucket_space+
                                                            (self.l_bucket+1)*self.color_space*self.bucket_space+
                                                            o_color*self.bucket_space+
                                                            o_bucket] = 1

                features[o_row][o_col][o_bucket][lc_cb_sc_i0+ (self.l_color+1)*(self.bucket_space+1)*self.shape_space*self.color_space+
                                                            (self.l_bucket+1)*self.shape_space*self.color_space+
                                                            o_shape*self.color_space+
                                                            o_color] = 1
                
                features[o_row][o_col][o_bucket][lc_cb_sb_i0+ (self.l_color+1)*(self.bucket_space+1)*self.shape_space*self.bucket_space+
                                                            (self.l_bucket+1)*self.shape_space*self.bucket_space+
                                                            o_shape*self.bucket_space+
                                                            o_bucket] = 1

                features[o_row][o_col][o_bucket][lc_cb_cb_i0+ (self.l_color+1)*(self.bucket_space+1)*self.color_space*self.bucket_space+
                                                            (self.l_bucket+1)*self.color_space*self.bucket_space+
                                                            o_color*self.bucket_space+
                                                            o_bucket] = 1

                features[o_row][o_col][o_bucket][lc_is_is_i0+ (self.l_index+1)*(self.shape_space+1)*self.index_space*self.shape_space+
                                                            (self.l_shape+1)*self.index_space*self.shape_space+
                                                            o_index*self.shape_space+
                                                            o_shape] = 1
                
                features[o_row][o_col][o_bucket][lc_ic_ic_i0+ (self.l_index+1)*(self.color_space+1)*self.index_space*self.color_space+
                                                            (self.l_color+1)*self.index_space*self.color_space+
                                                            o_index*self.color_space+
                                                            o_color] = 1

                features[o_row][o_col][o_bucket][lc_ib_ib_i0+ (self.l_index+1)*(self.bucket_space+1)*self.index_space*self.bucket_space+
                                                            (self.l_bucket+1)*self.index_space*self.bucket_space+
                                                            o_index*self.bucket_space+
                                                            o_bucket] = 1

        return features.flatten()


class RuleNearestBucket(RuleGameEnv):
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #
    #      This class constructs distance feature of bucket from action index
    #
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    def __init__(self, args):
        super(RuleNearestBucket, self).__init__(args)

        self.action_feature_dim = 2
        if(self.verbose==0):
            print("Total number of features : ", self.action_feature_dim)
            
        # define observation space of the model, in our case it corresponds to the feature dimension(which is 2 dimensional in this example) for each action
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.board_size*self.board_size*self.bucket_space,self.action_feature_dim), dtype=np.int)

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #
    #   Extract feature vector defined in the constructor
    #
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def get_feature(self):
        features = np.zeros((self.board_size, self.board_size, self.bucket_space, self.action_feature_dim))
        features[:,:,:,1] = 1
        
        for object_tuple in self.board:
            o_row, o_col = object_tuple['y']-1, object_tuple['x']-1   
            
            for o_bucket in range(self.bucket_space):
                b_row, b_col = self.bucket_tuple[o_bucket]
                # one step index with closest bucket
                features[o_row][o_col][o_bucket][0] = np.sqrt((b_row-o_row-1)**2+(b_col-o_col-1)**2)
                features[o_row][o_col][o_bucket][1] = 0


        return features.flatten()


class RuleNearestBucketWithFullFeatures(RuleGameEnv):
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #
    #      This class constructs distance feature with all other features
    #
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    def __init__(self, args):
        super(RuleNearestBucketWithFullFeatures, self).__init__(args)

        s, c, b = self.shape_space, self.color_space, self.bucket_space
        unary_dim = s+c+b
        binary_dim = s*c + s*b + c*b 
        one_step_unary_dim = (s+1)*s + (c+1)*c + (b+1)*b
        one_step_binary_dim = (s+1)*(c+1)*(s*c + s*b + c*b) + (s+1)*(b+1)*(s*c + s*b + c*b) + (c+1)*(b+1)*(s*c + s*b + c*b)
        self.action_feature_dim = unary_dim + binary_dim + one_step_unary_dim + one_step_binary_dim + 2       # 3 * 4 + 3 * 16 + 3 * 20 + 9 * 400 = 3722

        # define observation space of the model, in our case it corresponds to the feature dimension(which is 3722 dimensional in this example) for each action
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.board_size*self.board_size*self.bucket_space,self.action_feature_dim), dtype=np.int)

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #
    #   Extract feature vector defined in the constructor
    #
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def get_feature(self):
        features = np.zeros((self.board_size, self.board_size, self.bucket_space, self.action_feature_dim))
        features[:,:,:,1] = 1
        

        features = np.zeros((self.board_size, self.board_size, self.bucket_space, self.action_feature_dim))
        for object_tuple in self.board:
            o_row, o_col = object_tuple['y']-1, object_tuple['x']-1     # cgs index start with (1,1) so subtract 1
            o_color, o_shape = self.color_id[object_tuple['color']], self.shape_id[object_tuple['shape']]

            s_i0 = 0                                                                                                                # start index of current shape
            c_i0 = s_i0 + self.shape_space                                                                                          # start index of current color
            b_i0 = c_i0 + self.color_space                                                                                          # start index of current bucket
            
            sc_i0 = b_i0 + self.bucket_space                                                                                        # start index (current shape, current color) 
            sb_i0 = sc_i0 + self.shape_space*self.color_space                                                                       # start index (current shape, current bucekt)
            cb_i0 = sb_i0 + self.shape_space*self.bucket_space                                                                      # start index (current color, current bucket)
            
            lc_s_i0 = cb_i0 + self.color_space*self.bucket_space                                                                    # start index (last_shape, current_shape) 
            lc_c_i0 = lc_s_i0 + (self.shape_space+1)*self.shape_space                                                               # start index (last_color, current_color)
            lc_b_i0 = lc_c_i0 + (self.color_space+1)*self.color_space                                                               # start index (last_bucket, current_bucket)

            lc_sc_sc_i0 = lc_b_i0 + (self.bucket_space+1)*self.bucket_space                                                         # (last_shape, last_color, current_shape, current_color) sc_sc in the lhs name stands for this 4-tuple
            lc_sc_sb_i0 = lc_sc_sc_i0 + (self.shape_space+1)*(self.color_space+1)*self.shape_space*self.color_space                 # (last_shape, last_color, current_shape, current_bucket) sc_sb in the lhs name stands for this 4-tuple and so on for other features.
            lc_sc_cb_i0 = lc_sc_sb_i0 + (self.shape_space+1)*(self.color_space+1)*self.shape_space*self.bucket_space

            lc_sb_sc_i0 = lc_sc_cb_i0 + (self.shape_space+1)*(self.color_space+1)*self.color_space*self.bucket_space
            lc_sb_sb_i0 = lc_sb_sc_i0 + (self.shape_space+1)*(self.bucket_space+1)*self.shape_space*self.color_space
            lc_sb_cb_i0 = lc_sb_sb_i0 + (self.shape_space+1)*(self.bucket_space+1)*self.shape_space*self.bucket_space

            lc_cb_sc_i0 = lc_sb_cb_i0 + (self.shape_space+1)*(self.bucket_space+1)*self.color_space*self.bucket_space
            lc_cb_sb_i0 = lc_cb_sc_i0 + (self.color_space+1)*(self.bucket_space+1)*self.shape_space*self.color_space
            lc_cb_cb_i0 = lc_cb_sb_i0 + (self.color_space+1)*(self.bucket_space+1)*self.shape_space*self.bucket_space

            dis_i0 = lc_cb_cb_i0 + (self.color_space+1)*(self.bucket_space+1)*self.color_space*self.bucket_space
            
            for o_bucket in range(self.bucket_space):
                b_row, b_col = self.bucket_tuple[o_bucket]

                ## unaries
                features[o_row][o_col][o_bucket][s_i0+o_shape] = 1
                features[o_row][o_col][o_bucket][c_i0+o_color] = 1
                features[:,:,o_bucket, b_i0+o_bucket] = 1

                ## binaries
                features[o_row][o_col][o_bucket][sc_i0+o_shape*self.color_space+o_color] = 1
                features[o_row][o_col][o_bucket][sb_i0+o_shape*self.bucket_space+o_bucket] = 1
                features[o_row][o_col][o_bucket][cb_i0+o_color*self.bucket_space+o_bucket] = 1
                
                ## one_step unaries
                features[o_row][o_col][o_bucket][lc_s_i0+(self.l_shape+1)*(self.shape_space)+o_shape] = 1
                features[o_row][o_col][o_bucket][lc_c_i0+(self.l_color+1)*(self.color_space)+o_color] = 1
                features[o_row][o_col][o_bucket][lc_b_i0+(self.l_bucket+1)*(self.bucket_space)+o_bucket] = 1

                ## one_step binaries
                features[o_row][o_col][o_bucket][lc_sc_sc_i0+ (self.l_shape+1)*(self.color_space+1)*self.shape_space*self.color_space+
                                                            (self.l_color+1)*self.shape_space*self.color_space+
                                                            o_shape*self.color_space+
                                                            o_color] = 1
                
                features[o_row][o_col][o_bucket][lc_sc_sb_i0+ (self.l_shape+1)*(self.color_space+1)*self.shape_space*self.bucket_space+
                                                            (self.l_color+1)*self.shape_space*self.bucket_space+
                                                            o_shape*self.shape_space+
                                                            o_bucket] = 1

                features[o_row][o_col][o_bucket][lc_sc_cb_i0+ (self.l_shape+1)*(self.color_space+1)*self.color_space*self.bucket_space+
                                                            (self.l_color+1)*self.color_space*self.bucket_space+
                                                            o_color*self.bucket_space+
                                                            o_bucket] = 1
                
                features[o_row][o_col][o_bucket][lc_sb_sc_i0+ (self.l_shape+1)*(self.bucket_space+1)*self.shape_space*self.color_space+
                                                            (self.l_bucket+1)*self.shape_space*self.color_space+
                                                            o_shape*self.color_space+
                                                            o_color] = 1
                
                features[o_row][o_col][o_bucket][lc_sb_sb_i0+ (self.l_shape+1)*(self.bucket_space+1)*self.shape_space*self.bucket_space+
                                                            (self.l_bucket+1)*self.shape_space*self.bucket_space+
                                                            o_shape*self.bucket_space+
                                                            o_bucket] = 1

                features[o_row][o_col][o_bucket][lc_sb_cb_i0+ (self.l_shape+1)*(self.bucket_space+1)*self.color_space*self.bucket_space+
                                                            (self.l_bucket+1)*self.color_space*self.bucket_space+
                                                            o_color*self.bucket_space+
                                                            o_bucket] = 1

                features[o_row][o_col][o_bucket][lc_cb_sc_i0+ (self.l_color+1)*(self.bucket_space+1)*self.shape_space*self.color_space+
                                                            (self.l_bucket+1)*self.shape_space*self.color_space+
                                                            o_shape*self.color_space+
                                                            o_color] = 1
                
                features[o_row][o_col][o_bucket][lc_cb_sb_i0+ (self.l_color+1)*(self.bucket_space+1)*self.shape_space*self.bucket_space+
                                                            (self.l_bucket+1)*self.shape_space*self.bucket_space+
                                                            o_shape*self.bucket_space+
                                                            o_bucket] = 1

                features[o_row][o_col][o_bucket][lc_cb_cb_i0+ (self.l_color+1)*(self.bucket_space+1)*self.color_space*self.bucket_space+
                                                            (self.l_bucket+1)*self.color_space*self.bucket_space+
                                                            o_color*self.bucket_space+
                                                            o_bucket] = 1

                features[o_row][o_col][o_bucket][dis_i0] = np.sqrt((b_row-o_row-1)**2+(b_col-o_col-1)**2)
                features[o_row][o_col][o_bucket][dis_i0] = 0

        return features.flatten()

class RuleReadingOrderWithEmptyIndicator(RuleGameEnv):
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #
    #      This class constructs row and column features
    #
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    def __init__(self, args):
        super(RuleReadingOrderWithEmptyIndicator, self).__init__(args)
        self.action_feature_dim =  3 

        # define observation space of the model, in our case it corresponds to the feature dimension(which is 3 dimensional in this example) for each action
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.board_size*self.board_size*self.bucket_space,self.action_feature_dim), dtype=np.int)


    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #
    #   Extract feature vector ( unary, binary, last-current-unary, last-current-unary-binary )
    #
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def get_feature(self):
        features = np.zeros((self.board_size, self.board_size, self.bucket_space, self.action_feature_dim))
        features[:,:,:,-1] = 1      # initialize pick_empty_cell indicator feature for all actions  - fix the corresponding weight at -10**10 and do not learn it

        for object_tuple in self.board:
            o_row, o_col = object_tuple['y']-1, object_tuple['x']-1     # cgs index start with (1,1) so subtract 1
            
            for o_bucket in range(self.bucket_space):
                features[o_row][o_col][o_bucket][0] = o_row+1
                features[o_row][o_col][o_bucket][1] = o_col+1
                features[o_row][o_col][o_bucket][2] = 0

        return features.flatten()

class RuleReadingOrder(RuleGameEnv):
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #
    #      This class constructs row and column features
    #
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    def __init__(self, args):
        super(RuleReadingOrder, self).__init__(args)
        self.action_feature_dim =  2  

        # define observation space of the model, in our case it corresponds to the feature dimension(which is 2 dimensional in this example) for each action
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.board_size*self.board_size*self.bucket_space,self.action_feature_dim), dtype=np.int)


    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #
    #   Extract feature vector ( unary, binary, last-current-unary, last-current-unary-binary )
    #
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def get_feature(self):
        features = np.zeros((self.board_size, self.board_size, self.bucket_space, self.action_feature_dim))
        for object_tuple in self.board:
            o_row, o_col = object_tuple['y']-1, object_tuple['x']-1     # cgs index start with (1,1) so subtract 1
            
            for o_bucket in range(self.bucket_space):
                features[o_row][o_col][o_bucket][0] = o_row+1
                features[o_row][o_col][o_bucket][1] = o_col+1

        return features.flatten()


class RuleReadingOrderWithFullFeatures(RuleGameEnv):
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #
    #      This class constructs row and column features with all other previous features
    #
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    def __init__(self, args):
        super(RuleReadingOrderWithFullFeatures, self).__init__(args)
        s, c, b = self.shape_space, self.color_space, self.bucket_space
        unary_dim = s+c+b
        binary_dim = s*c + s*b + c*b 
        one_step_unary_dim = (s+1)*s + (c+1)*c + (b+1)*b
        one_step_binary_dim = (s+1)*(c+1)*(s*c + s*b + c*b) + (s+1)*(b+1)*(s*c + s*b + c*b) + (c+1)*(b+1)*(s*c + s*b + c*b)
        self.action_feature_dim = unary_dim + binary_dim + one_step_unary_dim + one_step_binary_dim + 2       # 3 * 4 + 3 * 16 + 3 * 20 + 9 * 400 = 3722

        # define observation space of the model, in our case it corresponds to the feature dimension(which is 3722 dimensional in this example) for each action
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.board_size*self.board_size*self.bucket_space,self.action_feature_dim), dtype=np.int)


    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #
    #   Extract feature vector ( unary, binary, last-current-unary, last-current-unary-binary )
    #
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def get_feature(self):
        features = np.zeros((self.board_size, self.board_size, self.bucket_space, self.action_feature_dim))
        for object_tuple in self.board:
            o_row, o_col = object_tuple['y']-1, object_tuple['x']-1     # cgs index start with (1,1) so subtract 1
            o_color, o_shape = self.color_id[object_tuple['color']], self.shape_id[object_tuple['shape']]

            s_i0 = 0                                                                                                                # start index of current shape
            c_i0 = s_i0 + self.shape_space                                                                                          # start index of current color
            b_i0 = c_i0 + self.color_space                                                                                          # start index of current bucket
            
            sc_i0 = b_i0 + self.bucket_space                                                                                        # start index (current shape, current color) 
            sb_i0 = sc_i0 + self.shape_space*self.color_space                                                                       # start index (current shape, current bucekt)
            cb_i0 = sb_i0 + self.shape_space*self.bucket_space                                                                      # start index (current color, current bucket)
            
            lc_s_i0 = cb_i0 + self.color_space*self.bucket_space                                                                    # start index (last_shape, current_shape) 
            lc_c_i0 = lc_s_i0 + (self.shape_space+1)*self.shape_space                                                               # start index (last_color, current_color)
            lc_b_i0 = lc_c_i0 + (self.color_space+1)*self.color_space                                                               # start index (last_bucket, current_bucket)

            lc_sc_sc_i0 = lc_b_i0 + (self.bucket_space+1)*self.bucket_space                                                         # (last_shape, last_color, current_shape, current_color) sc_sc in the lhs name stands for this 4-tuple
            lc_sc_sb_i0 = lc_sc_sc_i0 + (self.shape_space+1)*(self.color_space+1)*self.shape_space*self.color_space                 # (last_shape, last_color, current_shape, current_bucket) sc_sb in the lhs name stands for this 4-tuple and so on for other features.
            lc_sc_cb_i0 = lc_sc_sb_i0 + (self.shape_space+1)*(self.color_space+1)*self.shape_space*self.bucket_space

            lc_sb_sc_i0 = lc_sc_cb_i0 + (self.shape_space+1)*(self.color_space+1)*self.color_space*self.bucket_space
            lc_sb_sb_i0 = lc_sb_sc_i0 + (self.shape_space+1)*(self.bucket_space+1)*self.shape_space*self.color_space
            lc_sb_cb_i0 = lc_sb_sb_i0 + (self.shape_space+1)*(self.bucket_space+1)*self.shape_space*self.bucket_space

            lc_cb_sc_i0 = lc_sb_cb_i0 + (self.shape_space+1)*(self.bucket_space+1)*self.color_space*self.bucket_space
            lc_cb_sb_i0 = lc_cb_sc_i0 + (self.color_space+1)*(self.bucket_space+1)*self.shape_space*self.color_space
            lc_cb_cb_i0 = lc_cb_sb_i0 + (self.color_space+1)*(self.bucket_space+1)*self.shape_space*self.bucket_space

            rc_i0 = lc_cb_cb_i0 + (self.color_space+1)*(self.bucket_space+1)*self.color_space*self.bucket_space
            
            for o_bucket in range(self.bucket_space):
                ## unaries
                features[o_row][o_col][o_bucket][s_i0+o_shape] = 1
                features[o_row][o_col][o_bucket][c_i0+o_color] = 1
                features[:,:,o_bucket, b_i0+o_bucket] = 1

                ## binaries
                features[o_row][o_col][o_bucket][sc_i0+o_shape*self.color_space+o_color] = 1
                features[o_row][o_col][o_bucket][sb_i0+o_shape*self.bucket_space+o_bucket] = 1
                features[o_row][o_col][o_bucket][cb_i0+o_color*self.bucket_space+o_bucket] = 1
                
                ## one_step unaries
                features[o_row][o_col][o_bucket][lc_s_i0+(self.l_shape+1)*(self.shape_space)+o_shape] = 1
                features[o_row][o_col][o_bucket][lc_c_i0+(self.l_color+1)*(self.color_space)+o_color] = 1
                features[o_row][o_col][o_bucket][lc_b_i0+(self.l_bucket+1)*(self.bucket_space)+o_bucket] = 1

                ## one_step binaries
                features[o_row][o_col][o_bucket][lc_sc_sc_i0+ (self.l_shape+1)*(self.color_space+1)*self.shape_space*self.color_space+
                                                            (self.l_color+1)*self.shape_space*self.color_space+
                                                            o_shape*self.color_space+
                                                            o_color] = 1
                
                features[o_row][o_col][o_bucket][lc_sc_sb_i0+ (self.l_shape+1)*(self.color_space+1)*self.shape_space*self.bucket_space+
                                                            (self.l_color+1)*self.shape_space*self.bucket_space+
                                                            o_shape*self.shape_space+
                                                            o_bucket] = 1

                features[o_row][o_col][o_bucket][lc_sc_cb_i0+ (self.l_shape+1)*(self.color_space+1)*self.color_space*self.bucket_space+
                                                            (self.l_color+1)*self.color_space*self.bucket_space+
                                                            o_color*self.bucket_space+
                                                            o_bucket] = 1
                
                features[o_row][o_col][o_bucket][lc_sb_sc_i0+ (self.l_shape+1)*(self.bucket_space+1)*self.shape_space*self.color_space+
                                                            (self.l_bucket+1)*self.shape_space*self.color_space+
                                                            o_shape*self.color_space+
                                                            o_color] = 1
                
                features[o_row][o_col][o_bucket][lc_sb_sb_i0+ (self.l_shape+1)*(self.bucket_space+1)*self.shape_space*self.bucket_space+
                                                            (self.l_bucket+1)*self.shape_space*self.bucket_space+
                                                            o_shape*self.bucket_space+
                                                            o_bucket] = 1

                features[o_row][o_col][o_bucket][lc_sb_cb_i0+ (self.l_shape+1)*(self.bucket_space+1)*self.color_space*self.bucket_space+
                                                            (self.l_bucket+1)*self.color_space*self.bucket_space+
                                                            o_color*self.bucket_space+
                                                            o_bucket] = 1

                features[o_row][o_col][o_bucket][lc_cb_sc_i0+ (self.l_color+1)*(self.bucket_space+1)*self.shape_space*self.color_space+
                                                            (self.l_bucket+1)*self.shape_space*self.color_space+
                                                            o_shape*self.color_space+
                                                            o_color] = 1
                
                features[o_row][o_col][o_bucket][lc_cb_sb_i0+ (self.l_color+1)*(self.bucket_space+1)*self.shape_space*self.bucket_space+
                                                            (self.l_bucket+1)*self.shape_space*self.bucket_space+
                                                            o_shape*self.bucket_space+
                                                            o_bucket] = 1

                features[o_row][o_col][o_bucket][lc_cb_cb_i0+ (self.l_color+1)*(self.bucket_space+1)*self.color_space*self.bucket_space+
                                                            (self.l_bucket+1)*self.color_space*self.bucket_space+
                                                            o_color*self.bucket_space+
                                                            o_bucket] = 1

                features[o_row][o_col][o_bucket][rc_i0] = o_row+1
                features[o_row][o_col][o_bucket][rc_i0+1] = o_col+1

        return features.flatten()

class RuleNearestBucketReadingOrder(RuleGameEnv):
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #
    #      This class constructs five dimensional feature sufficient to learn reading order rule-game.
    #
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    def __init__(self, args):
        super(RuleNearestBucketReadingOrder, self).__init__(args)
         
        self.action_feature_dim = 4

        if(self.verbose==0):
            print("Total number of features : ", self.action_feature_dim)
            
        # define observation space of the model, in our case it corresponds to the feature dimension(which is 4 dimensional in this example) for each action
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.board_size*self.board_size*self.bucket_space,self.action_feature_dim), dtype=np.int)
    
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #
    #   Extract feature vector defined in the constructor
    #
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def get_feature(self):
        features = np.zeros((self.board_size, self.board_size, self.bucket_space, self.action_feature_dim))
        features[:,:,:,3] = 1
        for object_tuple in self.board:

            o_row, o_col = object_tuple['y']-1, object_tuple['x']-1     # cgs index start with (1,1) so subtract 1
            
            for o_bucket in range(self.bucket_space):
                b_row, b_col = self.bucket_tuple[o_bucket]
                features[o_row][o_col][o_bucket][0] = o_row+1
                features[o_row][o_col][o_bucket][1] = o_col+1
                features[o_row][o_col][o_bucket][2] = np.sqrt((b_row-o_row-1)**2+(b_col-o_col-1)**2)
                features[o_row][o_col][o_bucket][3] = 0

        return features.flatten()


class RuleNearestBucketFromTop(RuleGameEnv):
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #
    #      This class constructs five dimensional feature sufficient to learn reading order rule-game.
    #
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    def __init__(self, args):
        super(RuleNearestBucketFromTop, self).__init__(args)
         
        self.action_feature_dim = 3

        if(self.verbose==0):
            print("Total number of features : ", self.action_feature_dim)
            
        # define observation space of the model, in our case it corresponds to the feature dimension(which is 4 dimensional in this example) for each action
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.board_size*self.board_size*self.bucket_space,self.action_feature_dim), dtype=np.int)
    
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #
    #   Extract feature vector defined in the constructor
    #
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def get_feature(self):
        features = np.zeros((self.board_size, self.board_size, self.bucket_space, self.action_feature_dim))
        features[:,:,:,2] = 1
        for object_tuple in self.board:

            o_row, o_col = object_tuple['y']-1, object_tuple['x']-1     # cgs index start with (1,1) so subtract 1
            
            for o_bucket in range(self.bucket_space):
                b_row, b_col = self.bucket_tuple[o_bucket]
                features[o_row][o_col][o_bucket][0] = o_row+1
                features[o_row][o_col][o_bucket][1] = np.sqrt((b_row-o_row-1)**2+(b_col-o_col-1)**2)
                features[o_row][o_col][o_bucket][2] = 0

        return features.flatten()



class RuleNearestBucketReadingOrderWithFullFeatures(RuleGameEnv):
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #
    #      This class constructs five dimensional feature sufficient to learn reading order rule-game.
    #
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    def __init__(self, args):
        super(RuleNearestBucketReadingOrderWithFullFeatures, self).__init__(args)
         
        s, c, b = self.shape_space, self.color_space, self.bucket_space
        unary_dim = s+c+b
        binary_dim = s*c + s*b + c*b 
        one_step_unary_dim = (s+1)*s + (c+1)*c + (b+1)*b
        one_step_binary_dim = (s+1)*(c+1)*(s*c + s*b + c*b) + (s+1)*(b+1)*(s*c + s*b + c*b) + (c+1)*(b+1)*(s*c + s*b + c*b)
        self.action_feature_dim = unary_dim + binary_dim + one_step_unary_dim + one_step_binary_dim + 4       # 3 * 4 + 3 * 16 + 3 * 20 + 9 * 400 = 3724

        # define observation space of the model, in our case it corresponds to the feature dimension(which is 3722 dimensional in this example) for each action
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.board_size*self.board_size*self.bucket_space,self.action_feature_dim), dtype=np.int)

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #
    #   Extract feature vector defined in the constructor
    #
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def get_feature(self):
        features = np.zeros((self.board_size, self.board_size, self.bucket_space, self.action_feature_dim))
        features[:,:,:,self.action_feature_dim-1] = 1
        
        for object_tuple in self.board:
            o_row, o_col = object_tuple['y']-1, object_tuple['x']-1               
            o_color, o_shape = self.color_id[object_tuple['color']], self.shape_id[object_tuple['shape']]

            s_i0 = 0                                                                                                                # start index of current shape
            c_i0 = s_i0 + self.shape_space                                                                                          # start index of current color
            b_i0 = c_i0 + self.color_space                                                                                          # start index of current bucket
            
            sc_i0 = b_i0 + self.bucket_space                                                                                        # start index (current shape, current color) 
            sb_i0 = sc_i0 + self.shape_space*self.color_space                                                                       # start index (current shape, current bucekt)
            cb_i0 = sb_i0 + self.shape_space*self.bucket_space                                                                      # start index (current color, current bucket)
            
            lc_s_i0 = cb_i0 + self.color_space*self.bucket_space                                                                    # start index (last_shape, current_shape) 
            lc_c_i0 = lc_s_i0 + (self.shape_space+1)*self.shape_space                                                               # start index (last_color, current_color)
            lc_b_i0 = lc_c_i0 + (self.color_space+1)*self.color_space                                                               # start index (last_bucket, current_bucket)

            lc_sc_sc_i0 = lc_b_i0 + (self.bucket_space+1)*self.bucket_space                                                         # (last_shape, last_color, current_shape, current_color) sc_sc in the lhs name stands for this 4-tuple
            lc_sc_sb_i0 = lc_sc_sc_i0 + (self.shape_space+1)*(self.color_space+1)*self.shape_space*self.color_space                 # (last_shape, last_color, current_shape, current_bucket) sc_sb in the lhs name stands for this 4-tuple and so on for other features.
            lc_sc_cb_i0 = lc_sc_sb_i0 + (self.shape_space+1)*(self.color_space+1)*self.shape_space*self.bucket_space

            lc_sb_sc_i0 = lc_sc_cb_i0 + (self.shape_space+1)*(self.color_space+1)*self.color_space*self.bucket_space
            lc_sb_sb_i0 = lc_sb_sc_i0 + (self.shape_space+1)*(self.bucket_space+1)*self.shape_space*self.color_space
            lc_sb_cb_i0 = lc_sb_sb_i0 + (self.shape_space+1)*(self.bucket_space+1)*self.shape_space*self.bucket_space

            lc_cb_sc_i0 = lc_sb_cb_i0 + (self.shape_space+1)*(self.bucket_space+1)*self.color_space*self.bucket_space
            lc_cb_sb_i0 = lc_cb_sc_i0 + (self.color_space+1)*(self.bucket_space+1)*self.shape_space*self.color_space
            lc_cb_cb_i0 = lc_cb_sb_i0 + (self.color_space+1)*(self.bucket_space+1)*self.shape_space*self.bucket_space

            rc_dis_i0 = lc_cb_cb_i0 + (self.color_space+1)*(self.bucket_space+1)*self.color_space*self.bucket_space
            
            for o_bucket in range(self.bucket_space):
                b_row, b_col = self.bucket_tuple[o_bucket]

                ## unaries
                features[o_row][o_col][o_bucket][s_i0+o_shape] = 1
                features[o_row][o_col][o_bucket][c_i0+o_color] = 1
                features[:,:,o_bucket, b_i0+o_bucket] = 1

                ## binaries
                features[o_row][o_col][o_bucket][sc_i0+o_shape*self.color_space+o_color] = 1
                features[o_row][o_col][o_bucket][sb_i0+o_shape*self.bucket_space+o_bucket] = 1
                features[o_row][o_col][o_bucket][cb_i0+o_color*self.bucket_space+o_bucket] = 1
                
                ## one_step unaries
                features[o_row][o_col][o_bucket][lc_s_i0+(self.l_shape+1)*(self.shape_space)+o_shape] = 1
                features[o_row][o_col][o_bucket][lc_c_i0+(self.l_color+1)*(self.color_space)+o_color] = 1
                features[o_row][o_col][o_bucket][lc_b_i0+(self.l_bucket+1)*(self.bucket_space)+o_bucket] = 1

                ## one_step binaries
                features[o_row][o_col][o_bucket][lc_sc_sc_i0+ (self.l_shape+1)*(self.color_space+1)*self.shape_space*self.color_space+
                                                            (self.l_color+1)*self.shape_space*self.color_space+
                                                            o_shape*self.color_space+
                                                            o_color] = 1
                
                features[o_row][o_col][o_bucket][lc_sc_sb_i0+ (self.l_shape+1)*(self.color_space+1)*self.shape_space*self.bucket_space+
                                                            (self.l_color+1)*self.shape_space*self.bucket_space+
                                                            o_shape*self.shape_space+
                                                            o_bucket] = 1

                features[o_row][o_col][o_bucket][lc_sc_cb_i0+ (self.l_shape+1)*(self.color_space+1)*self.color_space*self.bucket_space+
                                                            (self.l_color+1)*self.color_space*self.bucket_space+
                                                            o_color*self.bucket_space+
                                                            o_bucket] = 1
                
                features[o_row][o_col][o_bucket][lc_sb_sc_i0+ (self.l_shape+1)*(self.bucket_space+1)*self.shape_space*self.color_space+
                                                            (self.l_bucket+1)*self.shape_space*self.color_space+
                                                            o_shape*self.color_space+
                                                            o_color] = 1
                
                features[o_row][o_col][o_bucket][lc_sb_sb_i0+ (self.l_shape+1)*(self.bucket_space+1)*self.shape_space*self.bucket_space+
                                                            (self.l_bucket+1)*self.shape_space*self.bucket_space+
                                                            o_shape*self.bucket_space+
                                                            o_bucket] = 1

                features[o_row][o_col][o_bucket][lc_sb_cb_i0+ (self.l_shape+1)*(self.bucket_space+1)*self.color_space*self.bucket_space+
                                                            (self.l_bucket+1)*self.color_space*self.bucket_space+
                                                            o_color*self.bucket_space+
                                                            o_bucket] = 1

                features[o_row][o_col][o_bucket][lc_cb_sc_i0+ (self.l_color+1)*(self.bucket_space+1)*self.shape_space*self.color_space+
                                                            (self.l_bucket+1)*self.shape_space*self.color_space+
                                                            o_shape*self.color_space+
                                                            o_color] = 1
                
                features[o_row][o_col][o_bucket][lc_cb_sb_i0+ (self.l_color+1)*(self.bucket_space+1)*self.shape_space*self.bucket_space+
                                                            (self.l_bucket+1)*self.shape_space*self.bucket_space+
                                                            o_shape*self.bucket_space+
                                                            o_bucket] = 1

                features[o_row][o_col][o_bucket][lc_cb_cb_i0+ (self.l_color+1)*(self.bucket_space+1)*self.color_space*self.bucket_space+
                                                            (self.l_bucket+1)*self.color_space*self.bucket_space+
                                                            o_color*self.bucket_space+
                                                            o_bucket] = 1

                features[o_row][o_col][o_bucket][rc_dis_i0] = o_row+1
                features[o_row][o_col][o_bucket][rc_dis_i0+1] = o_col+1
                features[o_row][o_col][o_bucket][rc_dis_i0+2] = np.sqrt((b_row-o_row-1)**2+(b_col-o_col-1)**2)
                features[o_row][o_col][o_bucket][rc_dis_i0+3] = 0

        return features.flatten()

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

        # PENDING MODIFICATION
        # define observation space of the model, in our case it corresponds to the feature dimension(which is 10 dimensional in this example) for each action
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.board_size*self.board_size*self.bucket_space,self.action_feature_dim), dtype=int)

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #
    #   Create feature vector with 1's corresponding to objects on the board
    #
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def get_feature(self):

        features = np.zeros((self.board_size, self.board_size, self.shape_space+self.color_space))

        # Loop over the corresponding objects on the board (features are already initialized to zero otherwise)
        for object_tuple in self.board:
            # Extract information associated with current object
            o_row, o_col, o_color, o_shape = object_tuple['y'], object_tuple['x'], self.color_id[object_tuple['color']], self.shape_id[object_tuple['shape']]
            #print(o_row,o_col,o_shape,o_color)
            
            # Write out 1's for the objects shape and color (in the correct row,col position in the feature array)
            features[o_row-1][o_col-1][o_shape]=1
            features[o_row-1][o_col-1][self.shape_space+o_color]=1
        #print(features)
        return features.flatten()

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
            'SEED' : 0,
            'RUN_MODE' : "RULE"
        }

    test_featurization(args)
