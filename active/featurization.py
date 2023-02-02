# Adapted from code by Shubham Bharti and Yiding Chen

import numpy as np
import os, sys
#from gym import spaces
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
class Naive_N_Board_Dense_Action_Dense(RuleGameEnv):
   
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #
    #      This class constructs a one hot representation of the board state with n-steps of memory, using dense representations of the board and actions
    #
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def __init__(self, args):
        super(Naive_N_Board_Dense_Action_Dense, self).__init__(args)
        # Action space (<bucket_space> actions per cell (<board_size>*<board_size>))
        self.out_dim = self.board_size*self.board_size*self.bucket_space
        # Dimension of representation of past actions (row+column+bucket of the chosen action action, each is essentially one-hot-encoded in its respective feature space)
        self.dense_action_dim = self.board_size+self.board_size+self.bucket_space
        # Input dimension (each step of past information represented with color+shape+row+column+bucket, current board represented using shape+color information for each cell)
        self.in_dim = self.n_steps*(self.color_space+self.shape_space + self.dense_action_dim) + self.board_size*self.board_size*(self.shape_space+self.color_space)

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #
    #   Create feature vector with 1's corresponding to objects on the board
    #
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def get_feature(self):
        # Dictionary contains the features, an action mask, and valid moves
        feature_dict = {}
        # Masks the board, assumes no actions are allowed
        mask = np.zeros(self.out_dim)
        # Inverse of the mask, assumes all actions are allowed
        inv_mask = np.ones(self.out_dim)
        # Preallocate feature representation of the current board
        features = np.zeros((self.board_size,self.board_size,self.shape_space+self.color_space))
        # Preallocate feature representation of past accepted shape/colors
        step_features = np.zeros(self.shape_space+self.color_space)
        # Preallocate feature representation of past moves
        step_move = np.zeros(self.dense_action_dim)

        # Current board
        # Loop over the corresponding objects on the board (features are already initialized to zero otherwise)
        for object_tuple in self.board:
            # Extract information associated with current object
            o_row, o_col, o_color, o_shape = object_tuple['y'], object_tuple['x'], self.color_id[object_tuple['color']], self.shape_id[object_tuple['shape']]
            # Loop over buckets available to current piece
            for i in range(self.bucket_space):
                # Identify the action index associated with this piece and bucket
                idx = np.ravel_multi_index((o_row-1,o_col-1,i),(self.board_size,self.board_size,self.bucket_space))
                # Allow this move through the action mask
                mask[idx]=1
                # Remove this move from the inverse mask
                inv_mask[idx]=0
            # Write out 1's for the objects shape and color (in the correct row,col position in the feature array)
            features[o_row-1][o_col-1][o_shape]=1
            features[o_row-1][o_col-1][self.shape_space+o_color]=1
        # Collect representation of current board into flattened vector
        features = features.flatten()

        # Past information
        for step in range(self.n_steps):
            # Information of past moves stored in last_attributes, check to see if there is information available for current step of memory
            # last_attributes is a list with elements of the form (o_shape, o_color)
            if self.last_attributes[step] is not None:
                # One hot encode the shape, then the color
                step_features[self.last_attributes[step][0]]=1
                step_features[self.last_attributes[step][1]+self.shape_space]=1
            # Check to see if these was a recorded move for the current step of memory
            if self.last_moves[step] is not None:
                # Take the action index and turn it into a row, column, and bucket (all zero-indexed)
                row,col,bucket = self.action_index_to_tuple(self.last_moves[step])
                # One hot encode the row, column, and bucket
                step_move[row] = 1
                step_move[self.board_size+col]=1
                step_move[self.board_size+self.board_size+bucket]=1
            # Append the current step information to the feature vector
            features = np.concatenate((features,step_features,step_move),axis=0)
            # Reset the features and move information for next loop iteration
            step_features = np.zeros(self.shape_space+self.color_space)
            step_move = np.zeros(self.dense_action_dim)
        
        # Also rule out rules associated with incorrect moves made since the last correct move
        # self.move_list is a list of such action indices
        for inv in self.move_list:
            # Add current index to mask
            mask[inv] = 0
            # Flip value for inverse mask
            inv_mask[inv]=1
        
        # Put all information into dictionary
        feature_dict['features']=features
        feature_dict['mask']=inv_mask
        feature_dict['valid']=np.nonzero(mask)[0]
        
        return feature_dict 

class Naive_N_Board_Sparse_Action_Sparse(RuleGameEnv):
   
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #
    #      This class constructs a one hot representation of the board state with n-steps of memory, using sparse representations of the board and action
    #
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def __init__(self, args):
        super(Naive_N_Board_Sparse_Action_Sparse, self).__init__(args)
        # Current board represented as 'layers' of boards, one-hot encoding shape and color information
        self.board_representation_size = self.board_size*self.board_size*(self.shape_space+self.color_space)
        # Output represents all possible actions
        self.out_dim = self.board_size*self.board_size*self.bucket_space
        # Input representation has board state and action index for every step of memory + representation of current board
        self.in_dim = self.n_steps*(self.board_representation_size + self.out_dim) + self.board_representation_size

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #
    #   Create feature vector with 1's corresponding to objects on the board
    #
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def get_feature(self):
        # Dictionary contains the features, an action mask, and valid moves
        feature_dict = {}
        # Masks the board, assumes no actions are allowed
        mask = np.zeros(self.out_dim)
        # Inverse of the mask, assumes all actions are allowed
        inv_mask = np.ones(self.out_dim)
        # Preallocate feature representation of the current board
        features = np.zeros((self.board_size,self.board_size,self.shape_space+self.color_space))
        # Preallocate feature representation of past accepted shape/colors
        step_features = np.zeros((self.board_size,self.board_size,self.shape_space+self.color_space))
        # Preallocate feature representation of past moves
        step_move = np.zeros(self.out_dim)

        # Current board
        # Loop over the corresponding objects on the board (features are already initialized to zero otherwise)
        for object_tuple in self.board:
            # Extract information associated with current object
            o_row, o_col, o_color, o_shape = object_tuple['y'], object_tuple['x'], self.color_id[object_tuple['color']], self.shape_id[object_tuple['shape']]
            # Loop over buckets available to current piece
            for i in range(self.bucket_space):
                # Identify the index associated with this cell and bucket
                idx = np.ravel_multi_index((o_row-1,o_col-1,i),(self.board_size,self.board_size,self.bucket_space))
                # Adjust mask values
                mask[idx]=1
                inv_mask[idx]=0
            # Write out 1's for the objects shape and color (in the correct row,col position in the feature array)
            features[o_row-1][o_col-1][o_shape]=1
            features[o_row-1][o_col-1][self.shape_space+o_color]=1
        # Take current board representation and flatten into a vector
        features = features.flatten()

        # Past moves
        # Loop over each past step
        for step in range(self.n_steps):
            # If any information is in the list 'last_boards', it should be added to the featurization
            if self.last_boards[step] is not None:
                # Follow same structure as the current board
                for object_tuple in self.last_boards[step]:
                    o_row, o_col, o_color, o_shape = object_tuple['y'], object_tuple['x'], self.color_id[object_tuple['color']], self.shape_id[object_tuple['shape']]
                    step_features[o_row-1][o_col-1][o_shape]=1
                    step_features[o_row-1][o_col-1][self.shape_space+o_color]=1
            # Flatten the tensor to a list (it is preallocated to 0's, so no need to consider the else case)
            step_features = step_features.flatten()
            # Look at list of past moves and see if these is a recorded action index
            if self.last_moves[step] is not None:
                # Flip the associated value in the action list
                step_move[self.last_moves[step]] = 1
            
            # Concatenate the step information with the existing feature vector
            features = np.concatenate((features,step_features,step_move),axis=0)
            # Reset quantities for next loop iteration
            step_features = np.zeros((self.board_size,self.board_size,self.shape_space+self.color_space))
            step_move = np.zeros(self.out_dim)

        # Also rule out rules associated with incorrect moves made since the last correct move
        # self.move_list is a list of such action indices 
        for inv in self.move_list:
            mask[inv] = 0
            inv_mask[inv]=1

        # Bundle information into final dictionary
        feature_dict['features']=features
        feature_dict['mask']=inv_mask
        feature_dict['valid']=np.nonzero(mask)[0]
     
        return feature_dict

class Naive_N_Board_Dense_Action_Sparse(RuleGameEnv):
   
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #
    #      This class constructs a one hot representation of the board state with n-steps of memory, with semi-dense representation of past boards and sparse representation of actions
    #
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def __init__(self, args):
        super(Naive_N_Board_Dense_Action_Sparse, self).__init__(args)
        # Representation of the current board (layers of one-hot encoded boards associated with shapes and colors)
        self.board_representation_size = self.board_size*self.board_size*(self.shape_space+self.color_space)
        # Complete set of available actions
        self.out_dim = self.board_size*self.board_size*self.bucket_space
        # Input representation has the shapes/colors of past objects and action indices for each step of memory + current board representation
        self.in_dim = self.n_steps*(self.color_space+self.shape_space + self.out_dim) + self.board_representation_size

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #
    #   Create feature vector with 1's corresponding to objects on the board
    #
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def get_feature(self):
        # Dictionary contains the features, an action mask, and valid moves
        feature_dict = {}
        # Masks the board, assumes no actions are allowed
        mask = np.zeros(self.out_dim)
        # Inverse of the mask, assumes all actions are allowed
        inv_mask = np.ones(self.out_dim)
        # Preallocate feature representation of the current board
        features = np.zeros((self.board_size,self.board_size,self.shape_space+self.color_space))
        # Preallocate feature representation of past accepted shape/colors
        step_features = np.zeros(self.shape_space+self.color_space)
        # Preallocate feature representation of past moves
        step_move = np.zeros(self.out_dim)

        # Current board
        # Loop over the corresponding objects on the board (features are already initialized to zero otherwise)
        for object_tuple in self.board:
            # Extract information associated with current object
            o_row, o_col, o_color, o_shape = object_tuple['y'], object_tuple['x'], self.color_id[object_tuple['color']], self.shape_id[object_tuple['shape']]
            # Loop over buckets available to current piece
            for i in range(self.bucket_space):
                # Identify the index associated with this cell and bucket
                idx = np.ravel_multi_index((o_row-1,o_col-1,i),(self.board_size,self.board_size,self.bucket_space))
                # Adjust mask values
                mask[idx]=1
                inv_mask[idx]=0
            # Write out 1's for the objects shape and color (in the correct row,col position in the feature array)
            features[o_row-1][o_col-1][o_shape]=1
            features[o_row-1][o_col-1][self.shape_space+o_color]=1
        # Take current board representation and flatten into a vector
        features = features.flatten()

        # Past moves
        # Loop over each past step
        for step in range(self.n_steps):
            # Information of past moves stored in last_attributes, check to see if there is information available for current step of memory
            # last_attributes is a list with elements of the form (o_shape, o_color)
            if self.last_attributes[step] is not None:
                # One hot encode the shape, then the color
                step_features[self.last_attributes[step][0]]=1
                step_features[self.last_attributes[step][1]+self.shape_space]=1
            # Look at list of past moves and see if these is a recorded action index
            if self.last_moves[step] is not None:
                # Flip the associated value in the action list
                step_move[self.last_moves[step]] = 1
            
            # Concatenate the step information with the existing feature vector
            features = np.concatenate((features,step_features,step_move),axis=0)
            # Reset quantities for next loop iteration
            step_features = np.zeros(self.shape_space+self.color_space)
            step_move = np.zeros(self.out_dim)

        # Also rule out rules associated with incorrect moves made since the last correct move
        # self.move_list is a list of such action indices 
        for inv in self.move_list:
            mask[inv] = 0
            inv_mask[inv]=1

        # Bundle information into final dictionary
        feature_dict['features']=features
        feature_dict['mask']=inv_mask
        feature_dict['valid']=np.nonzero(mask)[0]
     
        return feature_dict

class Naive_N_Board_Dense_alt_Action_Sparse(RuleGameEnv):
   
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #
    #      This class constructs a one hot representation of the board state with n-steps of memory, with semi-dense representation of past boards and sparse representation of actions
    #
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def __init__(self, args):
        super(Naive_N_Board_Dense_alt_Action_Sparse, self).__init__(args)
        # Representation of the current board (layers of one-hot encoded boards associated with shapes and colors)
        self.board_representation_size = self.board_size*self.board_size*(self.shape_space+self.color_space)
        # Complete set of available actions
        self.out_dim = self.board_size*self.board_size*self.bucket_space
        # Input representation has the shapes/colors of past objects and action indices for each step of memory + current board representation
        self.in_dim = self.n_steps*(self.color_space+self.shape_space+self.board_size+self.board_size + self.out_dim) + self.board_representation_size

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #
    #   Create feature vector with 1's corresponding to objects on the board
    #
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def get_feature(self):
        # Dictionary contains the features, an action mask, and valid moves
        feature_dict = {}
        # Masks the board, assumes no actions are allowed
        mask = np.zeros(self.out_dim)
        # Inverse of the mask, assumes all actions are allowed
        inv_mask = np.ones(self.out_dim)
        # Preallocate feature representation of the current board
        features = np.zeros((self.board_size,self.board_size,self.shape_space+self.color_space))
        # Preallocate feature representation of past accepted shape/colors
        step_features = np.zeros(self.color_space+self.shape_space+self.board_size+self.board_size)
        # Preallocate feature representation of past moves
        step_move = np.zeros(self.out_dim)

        # Current board
        # Loop over the corresponding objects on the board (features are already initialized to zero otherwise)
        for object_tuple in self.board:
            # Extract information associated with current object
            o_row, o_col, o_color, o_shape = object_tuple['y'], object_tuple['x'], self.color_id[object_tuple['color']], self.shape_id[object_tuple['shape']]
            # Loop over buckets available to current piece
            for i in range(self.bucket_space):
                # Identify the index associated with this cell and bucket
                idx = np.ravel_multi_index((o_row-1,o_col-1,i),(self.board_size,self.board_size,self.bucket_space))
                # Adjust mask values
                mask[idx]=1
                inv_mask[idx]=0
            # Write out 1's for the objects shape and color (in the correct row,col position in the feature array)
            features[o_row-1][o_col-1][o_shape]=1
            features[o_row-1][o_col-1][self.shape_space+o_color]=1
        # Take current board representation and flatten into a vector
        features = features.flatten()

        # Past moves
        # Loop over each past step
        for step in range(self.n_steps):
            # Information of past moves stored in last_attributes, check to see if there is information available for current step of memory
            # last_attributes is a list with elements of the form (o_shape, o_color)
            if self.last_attributes[step] is not None:
                # One hot encode the shape, then the color
                step_features[self.last_attributes[step][0]]=1
                step_features[self.last_attributes[step][1]+self.shape_space]=1
            # Look at list of past moves and see if these is a recorded action index
            if self.last_moves[step] is not None:
                # Flip the associated value in the action list
                step_move[self.last_moves[step]] = 1
                row,col,bucket = self.action_index_to_tuple(self.last_moves[step])
                step_features[self.shape_space+self.color_space+row]=1
                step_features[self.shape_space+self.color_space+self.board_size+col]=1
            
            # Concatenate the step information with the existing feature vector
            features = np.concatenate((features,step_features,step_move),axis=0)
            # Reset quantities for next loop iteration
            step_features = np.zeros(self.color_space+self.shape_space+self.board_size+self.board_size)
            step_move = np.zeros(self.out_dim)

        # Also rule out rules associated with incorrect moves made since the last correct move
        # self.move_list is a list of such action indices 
        for inv in self.move_list:
            mask[inv] = 0
            inv_mask[inv]=1

        # Bundle information into final dictionary
        feature_dict['features']=features
        feature_dict['mask']=inv_mask
        feature_dict['valid']=np.nonzero(mask)[0]
     
        return feature_dict
class Naive_N_Board_Sparse_Action_Dense(RuleGameEnv):
   
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #
    #      This class constructs a one hot representation of the board state with n-steps of memory, with sparse past board representation and dense past action representation
    #
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def __init__(self, args):
        super(Naive_N_Board_Sparse_Action_Dense, self).__init__(args)
        # Representation of the current board (layers of one-hot encoded boards associated with shapes and colors)
        self.board_representation_size = self.board_size*self.board_size*(self.shape_space+self.color_space)
        # Action space (<bucket_space> actions per cell (<board_size>*<board_size>))
        self.out_dim = self.board_size*self.board_size*self.bucket_space
        # Dimension of representation of past actions (row+column+bucket of the chosen action action, each is essentially one-hot-encoded in its respective feature space)
        self.dense_action_dim = self.board_size+self.board_size+self.bucket_space
        # Input representation has the shapes/colors of past objects and action indices for each step of memory + current board representation
        self.in_dim = self.n_steps*(self.board_representation_size + self.dense_action_dim) + self.board_representation_size

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #
    #   Create feature vector with 1's corresponding to objects on the board
    #
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def get_feature(self):
        # Dictionary contains the features, an action mask, and valid moves
        feature_dict = {}
        # Masks the board, assumes no actions are allowed
        mask = np.zeros(self.out_dim)
        # Inverse of the mask, assumes all actions are allowed
        inv_mask = np.ones(self.out_dim)
        # Preallocate feature representation of the current board
        features = np.zeros((self.board_size,self.board_size,self.shape_space+self.color_space))
        # Preallocate feature representation of past accepted shape/colors
        step_features = np.zeros((self.board_size,self.board_size,self.shape_space+self.color_space))
        # Preallocate feature representation of past moves
        step_move = np.zeros(self.dense_action_dim)

        # Current board
        # Loop over the corresponding objects on the board (features are already initialized to zero otherwise)
        for object_tuple in self.board:
            # Extract information associated with current object
            o_row, o_col, o_color, o_shape = object_tuple['y'], object_tuple['x'], self.color_id[object_tuple['color']], self.shape_id[object_tuple['shape']]
            # Loop over buckets available to current piece
            for i in range(self.bucket_space):
                # Identify the index associated with this cell and bucket
                idx = np.ravel_multi_index((o_row-1,o_col-1,i),(self.board_size,self.board_size,self.bucket_space))
                # Adjust mask values
                mask[idx]=1
                inv_mask[idx]=0
            # Write out 1's for the objects shape and color (in the correct row,col position in the feature array)
            features[o_row-1][o_col-1][o_shape]=1
            features[o_row-1][o_col-1][self.shape_space+o_color]=1
        # Take current board representation and flatten into a vector
        features = features.flatten()

        # Past moves
        # Loop over each past step
        for step in range(self.n_steps):
            # If any information is in the list 'last_boards', it should be added to the featurization
            if self.last_boards[step] is not None:
                # Follow same structure as the current board
                for object_tuple in self.last_boards[step]:
                    o_row, o_col, o_color, o_shape = object_tuple['y'], object_tuple['x'], self.color_id[object_tuple['color']], self.shape_id[object_tuple['shape']]
                    step_features[o_row-1][o_col-1][o_shape]=1
                    step_features[o_row-1][o_col-1][self.shape_space+o_color]=1
            # Flatten the tensor to a list (it is preallocated to 0's, so no need to consider the else case)
            step_features = step_features.flatten()
            # Check to see if these was a recorded move for the current step of memory
            if self.last_moves[step] is not None:
                # Take the action index and turn it into a row, column, and bucket (all zero-indexed)
                row,col,bucket = self.action_index_to_tuple(self.last_moves[step])
                # One hot encode the row, column, and bucket
                step_move[row] = 1
                step_move[self.board_size+col]=1
                step_move[self.board_size+self.board_size+bucket]=1
            
            # Concatenate the step information with the existing feature vector
            features = np.concatenate((features,step_features,step_move),axis=0)
            # Reset quantities for next loop iteration
            step_features = np.zeros((self.board_size,self.board_size,self.shape_space+self.color_space))
            step_move = np.zeros(self.dense_action_dim)

        # Also rule out rules associated with incorrect moves made since the last correct move
        # self.move_list is a list of such action indices 
        for inv in self.move_list:
            mask[inv] = 0
            inv_mask[inv]=1

        # Bundle information into final dictionary
        feature_dict['features']=features
        feature_dict['mask']=inv_mask
        feature_dict['valid']=np.nonzero(mask)[0]
     
        return feature_dict
    
class Naive_N_Board_SparseDense_Action_SparseDense(RuleGameEnv):
   
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #
    #      This class constructs a one hot representation of the board state with n-steps of memory, using both dense and sparse representations for past boards and actions
    #
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def __init__(self, args):
        super(Naive_N_Board_SparseDense_Action_SparseDense, self).__init__(args)
        # Representation of the current board (layers of one-hot encoded boards associated with shapes and colors)
        self.board_representation_size = self.board_size*self.board_size*(self.shape_space+self.color_space)
        # Action space (<bucket_space> actions per cell (<board_size>*<board_size>))
        self.out_dim = self.board_size*self.board_size*self.bucket_space
        # Dimension of representation of past actions (row+column+bucket of the chosen action action, each is essentially one-hot-encoded in its respective feature space)
        self.dense_action_dim = self.board_size+self.board_size+self.bucket_space
        # Input dimension (each step of past information represented with color+shape+row+column+bucket, current board represented using shape+color information for each cell)
        self.in_dim = self.n_steps*(self.board_representation_size+self.color_space+self.shape_space + self.out_dim+self.dense_action_dim) + self.board_representation_size

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #
    #   Create feature vector with 1's corresponding to objects on the board
    #
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def get_feature(self):
        # Dictionary contains the features, an action mask, and valid moves
        feature_dict = {}
        # Masks the board, assumes no actions are allowed
        mask = np.zeros(self.out_dim)
        # Inverse of the mask, assumes all actions are allowed
        inv_mask = np.ones(self.out_dim)
        # Preallocate feature representation of the current board
        features = np.zeros((self.board_size,self.board_size,self.shape_space+self.color_space))

        # Current board
        # Loop over the corresponding objects on the board (features are already initialized to zero otherwise)
        for object_tuple in self.board:
            # Extract information associated with current object
            o_row, o_col, o_color, o_shape = object_tuple['y'], object_tuple['x'], self.color_id[object_tuple['color']], self.shape_id[object_tuple['shape']]
            # Loop over buckets available to current piece
            for i in range(self.bucket_space):
                # Identify the action index associated with this piece and bucket
                idx = np.ravel_multi_index((o_row-1,o_col-1,i),(self.board_size,self.board_size,self.bucket_space))
                # Allow this move through the action mask
                mask[idx]=1
                # Remove this move from the inverse mask
                inv_mask[idx]=0
            # Write out 1's for the objects shape and color (in the correct row,col position in the feature array)
            features[o_row-1][o_col-1][o_shape]=1
            features[o_row-1][o_col-1][self.shape_space+o_color]=1
        # Collect representation of current board into flattened vector
        features = features.flatten()

        # Past information
        # Sparse first, dense second
        # Preallocate feature representation of past accepted shape/colors
        sparse_step_features = np.zeros((self.board_size,self.board_size,self.shape_space+self.color_space))
        # Preallocate feature representation of past moves
        sparse_step_move = np.zeros(self.out_dim)
        # Preallocate feature representation of past accepted shape/colors
        dense_step_features = np.zeros(self.shape_space+self.color_space)
        # Preallocate feature representation of past moves
        dense_step_move = np.zeros(self.dense_action_dim)
        # Loop over past moves
        for step in range(self.n_steps):
            # If any information is in the list 'last_boards', it should be added to the featurization
            if self.last_boards[step] is not None:
                # Follow same structure as the current board
                for object_tuple in self.last_boards[step]:
                    o_row, o_col, o_color, o_shape = object_tuple['y'], object_tuple['x'], self.color_id[object_tuple['color']], self.shape_id[object_tuple['shape']]
                    sparse_step_features[o_row-1][o_col-1][o_shape]=1
                    sparse_step_features[o_row-1][o_col-1][self.shape_space+o_color]=1
            # Flatten the tensor to a list (it is preallocated to 0's, so no need to consider the else case)
            sparse_step_features = sparse_step_features.flatten()
            # Information of past moves stored in last_attributes, check to see if there is information available for current step of memory
            # last_attributes is a list with elements of the form (o_shape, o_color)
            if self.last_attributes[step] is not None:
                # One hot encode the shape, then the color
                dense_step_features[self.last_attributes[step][0]]=1
                dense_step_features[self.last_attributes[step][1]+self.shape_space]=1
            # Look at list of past moves and see if these is a recorded action index
            if self.last_moves[step] is not None:
                # Flip the associated value in the action list
                sparse_step_move[self.last_moves[step]] = 1
                # Take the action index and turn it into a row, column, and bucket (all zero-indexed)
                row,col,bucket = self.action_index_to_tuple(self.last_moves[step])
                # One hot encode the row, column, and bucket
                dense_step_move[row] = 1
                dense_step_move[self.board_size+col]=1
                dense_step_move[self.board_size+self.board_size+bucket]=1
            # Concatenate the step information with the existing feature vector
            features = np.concatenate((features,sparse_step_features,sparse_step_move,dense_step_features,dense_step_move),axis=0)
            # Reset quantities for next loop iteration
            sparse_step_features = np.zeros((self.board_size,self.board_size,self.shape_space+self.color_space))
            sparse_step_move = np.zeros(self.out_dim)
            dense_step_features = np.zeros(self.shape_space+self.color_space)
            dense_step_move = np.zeros(self.dense_action_dim)

        # Also rule out rules associated with incorrect moves made since the last correct move
        # self.move_list is a list of such action indices
        for inv in self.move_list:
            # Add current index to mask
            mask[inv] = 0
            # Flip value for inverse mask
            inv_mask[inv]=1
        
        # Put all information into dictionary
        feature_dict['features']=features
        feature_dict['mask']=inv_mask
        feature_dict['valid']=np.nonzero(mask)[0]
        
        return feature_dict 

def test_featurization(args):
    # Testing code for this level of abstraction - this may not be updated reliably
    #env = NaiveBoard(args)
    env = Naive_N_Board_Dense_Action_Dense(args)
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
            'INIT_OBJ_COUNT'  : 1,              # initial number of objects on the board
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
