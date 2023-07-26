# Adapted from code by Shubham Bharti and Yiding Chen

import numpy as np
import os, sys, random
from rule_game_env import *
from itertools import combinations

def get_shape(pd,reference):
    piece = pd['piece']
    return reference[piece['shape']]

def get_color(pd,reference):
    piece = pd['piece']
    return reference[piece['color']]

# zero-indexed (row/col/cell normally 1 indexed)
def get_row(pd,reference):
    piece = pd['piece']
    return piece['y']-1+reference

# zero-indexed (row/col/cell normally 1 indexed)
def get_col(pd,reference):
    piece = pd['piece']
    return piece['x']-1+reference

# zero-indexed (row/col/cell normally 1 indexed)
def get_cell(pd,reference):
    piece = pd['piece']
    return (piece['y']-1)*6+piece['x']-1

# zero-indexed (buckets always are)
def get_bucket(pd,reference):
    buckets = list(pd['reduced_move_list'])[-reference:]
    bucket_id = {0:0,1:1,2:2,3:3,None:4}
    bucket_tuple = tuple([bucket_id[x] for x in buckets])
    #print(buckets)
    #print(bucket_tuple)
    return np.ravel_multi_index(bucket_tuple,tuple([5 for x in range(reference)]))

def get_quadrant(pd,reference):
    cell = get_cell(pd,None)+1
    if cell in [1,2,3,7,8,9,13,14,15]:
        return 0
    elif cell in [4,5,6,10,11,12,16,17,18]:
        return 1
    elif cell in [19,20,21,25,26,27,31,32,33]:
        return 2
    elif cell in [22,23,24,28,29,30,34,35,36]:
        return 3
    else:
        print("error calculating quadrant")
        breakpoint()

class ShapeOnly(RuleGameEnv):
    def __init__(self,args):
        super(ShapeOnly,self).__init__(args)
        self.in_dim = self.shape_space
        self.out_dim = self.bucket_space
        self.type = 'tabular'

    def get_feature(self):
        feature_dict = {'row':None, 'col':None,'features':None}
        if len(self.board)>0:
            piece = random.choice(self.board)
            #print(piece)
            #print(self.shape_id)
            o_row, o_col, o_shape = piece['y'], piece['x'], self.shape_id[piece['shape']]
            state = o_shape
            feature_dict = {'row':o_row, # 1 indexed
                            'col':o_col, # 1 indexed
                            'features':o_shape}
        return feature_dict

class ProcessedEnv(RuleGameEnv):
    def __init__(self,args):
        super(ProcessedEnv,self).__init__(args)
        self.model_features = args['MODEL_FEATURES']
        # Sanity check on feature count input
        if len(self.model_features)<1:
            print("Did you forget to add model features?")
            breakpoint()

        # Store all information about how to process each feature in following dictionary
        self.feature_info = {'shape':{'input_space':self.shape_space,'func':get_shape,'reference':self.shape_id},
                       'color':{'input_space':self.color_space, 'func':get_color,'reference':self.color_id},
                       'move_row':{'input_space':self.board_size, 'func':get_row,'reference':1},
                       'move_col':{'input_space':self.board_size, 'func':get_col,'reference':1},
                       'row':{'input_space':self.board_size, 'func':get_row,'reference':0},
                       'col':{'input_space':self.board_size, 'func':get_col,'reference':0},
                       'quadrant':{'input_space':4,'func':get_quadrant,'reference':None},
                       'cell':{'input_space':self.board_size*self.board_size,'func':get_cell,'reference':None},
                       'bucket1':{'input_space':self.bucket_space+1,'func':get_bucket,'reference':1},
                       'bucket2':{'input_space':(self.bucket_space+1)**2,'func':get_bucket,'reference':2},
                       'bucket3':{'input_space':(self.bucket_space+1)**3,'func':get_bucket,'reference':3},
                       'bucket4':{'input_space':(self.bucket_space+1)**4,'func':get_bucket,'reference':4}}

    def process_features(self):
        # List of features, each entry is a dict describing the piece attributes
        self.feature_list = []
        if len(self.board)>0:
            for piece in self.board:
                # Initialize
                feature_vals = dict.fromkeys(self.model_features)
                feature_vals['move_row']=None
                feature_vals['move_col']=None
                
                # Inherits the move lists from RuleGameEnv (stores each past move/board)
                processing_dict = {'piece':piece,'full_move_list':self.full_move_list,'reduced_move_list':self.reduced_move_list,'board_list':self.board_list}
                feature_vals['move_row'] = get_row(processing_dict,1)
                feature_vals['move_col'] = get_col(processing_dict,1)
                for feat in self.model_features:
                    func = self.feature_info[feat]['func']
                    ref = self.feature_info[feat]['reference']
                    feature_vals[feat]=func(processing_dict,ref)
                self.feature_list.append(feature_vals)
    
    def get_feature(self):
        self.process_features()
        return self.feature_list
    
    def return_feature(self,feat):
        return {'move_row':self.feature_vals['move_row'],'move_col':self.feature_vals['move_col'],'features':self.feature_vals[feat]}

    def calc_dim(self,feat_arr):
        dims = tuple(self.feature_info[feat]['input_space'] for feat in feat_arr)
        return dims


class RandomPieceSelection(RuleGameEnv):
    def __init__(self,args):
        super(RandomPieceSelection,self).__init__(args)
        self.type = 'tabular'
        self.model_features = args['MODEL_FEATURES']
        feature_info = {'shape':{'input_space':self.shape_space,'func':get_shape,'reference':self.shape_id},
                       'color':{'input_space':self.color_space, 'func':get_color,'reference':self.color_id},
                       'row':{'input_space':self.board_size, 'func':get_row,'reference':None},
                       'col':{'input_space':self.board_size, 'func':get_col,'reference':None},
                       'cell':{'input_space':self.board_size*self.board_size,'func':get_cell,'reference':None},
                       'bucket1':{'input_space':self.bucket_space+1,'func':get_bucket,'reference':1},
                       'bucket2':{'input_space':(self.bucket_space+1)**2,'func':get_bucket,'reference':2},
                       'bucket3':{'input_space':(self.bucket_space+1)**3,'func':get_bucket,'reference':3},
                       'bucket4':{'input_space':(self.bucket_space+1)**4,'func':get_bucket,'reference':4}}
        
        self.feature_dims = tuple([feature_info[x]['input_space'] for x in self.model_features])
        self.functions = tuple([feature_info[x]['func'] for x in self.model_features])
        self.references = tuple([feature_info[x]['reference'] for x in self.model_features])
        self.in_dim = int(np.prod(self.feature_dims))
        self.out_dim = self.bucket_space
        
    def get_feature(self):
        feature_dict={'row':None,'col':None,'features':None}
        if len(self.board)>0:
            piece=random.choice(self.board)
            processing_dict = {'piece':piece,'full_move_list':self.full_move_list,'reduced_move_list':self.reduced_move_list,'board_list':self.board_list}
            feature_vals = tuple([f(processing_dict,reference) for (f,reference) in zip(self.functions,self.references)])
            feature_dict={'row':get_row(processing_dict,None)+1,
                          'col':get_col(processing_dict,None)+1,
                          'features':np.ravel_multi_index(feature_vals,self.feature_dims)}
        return feature_dict


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
    env = ShapeOnly(args)
    phi = env.get_feature()
    breakpoint()

if __name__ == "__main__":
    print("starting")
    rule_dir_path = sys.argv[1]
    rule_name = 'rules-05.txt'                 # select the rule-name here
    rule_file_path = os.path.join(rule_dir_path, rule_name)

    args = { 
            'INIT_OBJ_COUNT'  : 3,              # initial number of objects on the board
            'R_ACCEPT' : 0,                     # reward for a reject move
            'R_REJECT' : -1,                    # reward for an accept move
            'TRAIN_HORIZON' : 200,              # horizon for each training episode
            'TRAIN_EPISODES' :  100,            # run this many training episodes
            'VERBOSE' : 1,                      # for descriptive output               
            'RULE_FILE_PATH' : rule_file_path,  # full rule-file path      
            'RULE_NAME'  : rule_name,           # rule-name
            'BOARD_SIZE'  : 6,                  # for board of size 6x6
            'OBJECT_SPACE'  : 16,               # total distinct types of objects possible
            'COLOR_SPACE'  : 4,                 # total possible colors
            'SHAPE_SPACE'  : 4,                 # total possible shapes
            'BUCKET_SPACE'  : 4,                # total buckets
            'SEED' : 1,
            'RUN_MODE' : "RULE"
        }

    test_featurization(args)
