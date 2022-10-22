# Base class for interaction with the captive game server
# Originally written by Shubham Bharti and Yiding Chen

import numpy as np
import subprocess, sys, re, json, os
import time

class RuleGameEngine():

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #   This class contains the code to initialize a connection with the backend CGS server and further communicate with it.
    #   object_space : - types of object - #shapes \times #colors 
    #                  - index mapping 16 : [STAR, CIRCLE, SQUARE, TRIANGLE] x [RED, BLACK, YELLOW, BLUE] -> [0,1,...,14,15]
    #                  - index mapping 4  : [STAR, CIRCLE, SQUARE, TRIANGLE] -> [0,1,2,3]
    #
    #   bucket_space : - #buckets [4]
    #                  - index mapping [(7,0),(7,7),(0,7),(0,0)] -> [0,1,2,3]
    #   
    #   board_size   : - #cells in a row of the squre board   [6 x 6]
    #                  - index mapping [(1,1)(1,2),...,(6,5),(6,6,)] -> [0,1,...,34,35]
    #              
    #   r_accept, r_reject : - rewards for accept and reject move respectively
    #   rule_file_path     : - path of the rule file 
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def __init__(self, args):
        # Parse incoming arguments to internal variables
        self.board_size = args['BOARD_SIZE']
        # This parsing makes some assumptions about the base set of colors and shapes used
        self.shape_id = {'STAR':0, 'SQUARE':1, 'TRIANGLE':2, 'CIRCLE':3}
        self.color_id = {'RED':0, 'BLUE':1, 'BLACK':2, 'YELLOW':3}
        # Note: tuples denoted by (row,col) rather than (x,y)
        
        self.bucket_tuple = {0:(7,0), 1:(7,7), 2:(0,7), 3:(0,0)}
        #self.bucket_tuple = {0:(7,0), 1:(7,7)}
        self.initial_object_count = int(float(args['INIT_OBJ_COUNT']))
        self.object_space, self.bucket_space, self.color_space, self.shape_space = args['OBJECT_SPACE'], args['BUCKET_SPACE'], args['COLOR_SPACE'], args['SHAPE_SPACE']
        self.index_space = self.board_size*self.board_size
        self.r_accept, self.r_reject = float(args['R_ACCEPT']), float(args['R_REJECT'])
        self.horizon, self.time_left, self.shaping = args['TRAIN_HORIZON'], args['TRAIN_HORIZON'], args['SHAPING']
        self.seed, self.alpha, self.board = args['SEED'], args['ALPHA'], None
        self.rule_file_path, self.verbose = args['RULE_FILE_PATH'], args['VERBOSE']

        self.cgs = None
        # Open communication with the CGS
        self.open_channel(args)
        
        # Read the initial data
        self.read_data()
    
    def increase_init_obj(self):
        print("add initial obj")
        print(self.initial_object_count)
        self.initial_object_count += 1
        print(self.initial_object_count)

    # Open communication with the CGS via pipes
    # RULE call fixes object count, RULE_PARAM allows for flexible call based on object ranges
    def open_channel(self, args):
        if(args['RUN_MODE']=='RULE'):
            # Spawn a CGS as a child process
            self.cgs = subprocess.Popen(['java', '-Dseed='+str(self.seed), '-Doutput=STANDARD', 'edu.wisc.game.engine.Captive', self.rule_file_path, str(self.initial_object_count)], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        else:
            rp = args['RULE_PARAM']
            objectR, shapeR, colorR = str(rp['minO'])+":"+str(rp['maxO']), str(rp['minS'])+":"+str(rp['maxS']), str(rp['minC'])+":"+str(rp['maxC'])
            self.cgs = subprocess.Popen( ['java', '-Dseed='+str(self.seed), '-Doutput=STANDARD', 'edu.wisc.game.engine.Captive', self.rule_file_path, objectR, shapeR, colorR], stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    # Send command to CGS to end game instance
    def close_channel(self):
        self.cgs.stdin.write(b"EXIT\n")
        self.cgs.stdin.flush()

    def read_data(self):
        # child's stdout is where python process will read data from 
        statusLine = readLine(self.cgs.stdout, self.verbose).strip()
        jsonLine = readLine(self.cgs.stdout, self.verbose).strip()

        #print(statusLine)
        [self.response_code, self.status, self.move_count] = map( int, re.split("\s+", statusLine))
        self.board = (json.loads(jsonLine))["value"]

        if(self.verbose>=1):
            sys.stdout.write("Code=" +repr(self.response_code)+ ", status=" +repr(self.status)+ ", stepNo="+repr(self.move_count)+ "\n")

    def write_data(self, data):
        if(self.verbose>=1):
            sys.stdout.write("Sending: "+ data + "\n")
        data_as_bytes = data.encode('utf-8')
        self.cgs.stdin.write(data_as_bytes + b"\n")
        self.cgs.stdin.flush()


    def sample_new_board(self):
        # call 'NEW' function on backened and update internal representation(could just be json)
        data = "NEW"

        # reset clock and current object count
        self.write_data(data)
        self.read_data()
        self.initial_object_count, self.time_left = len(self.board), self.horizon


    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #   o_row, o_col : row and column index of the object to be moved (rows and columns of the board are 1-indexed, buckets lie on (0,:), (7,:))
    #   b_row, b_col : row and column index of the bucket to which the object is to be moved to
    #   return episode_status(=done), reward
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def take_action(self, o_row, o_col, b_row, b_col):
        # call 'MOVE' function on the backend
        # get the next data
        data = "MOVE " + str(o_row) + " " + str(o_col) + " " + str(b_row) + " " + str(b_col)
        self.write_data(data)
        self.read_data()

        if(self.shaping):
            if(len(self.board)>=self.time_left):
                reward = 0
            else:
                if(self.response_code == 0):
                    reward = 0
                else: 
                    reward = -1
        else:
            if(self.response_code == 0):
                reward = self.r_accept
            else:
                reward = self.r_reject
            
        self.time_left -= 1
        return self.status, self.response_code, reward 

    def print_board(self, ifprint = True):
        board = np.zeros((self.board_size, self.board_size))
        board.fill(-1)

        for object_tuple in self.board:
            o_row, o_col = object_tuple['y'], object_tuple['x']
            print("row",o_row)
            print("col",o_col)
            board[o_row-1][o_col-1] = self.color_id[object_tuple['color']]

        if ifprint:
            print(board)
        return board 
#----------------------------------------------------------------------
#-- Reads and returns one line of "informative" text (ignoring any
#-- comment lines
#-- inx = input channel (associated with the stdout of the pipe-based
#-- Captive Game Server, or with a socket used to talk to a socket-based CGS)
#----------------------------------------------------------------------
def readLine(inx, verbose):
    while True:
        s = inx.readline().decode("utf-8")
        if(verbose>=1):
            print("Received: "+ s)
        if not s:
            return s
        if s.startswith('#'):
            continue
        else:
            return s


def test_engine(args):
    engine = RuleGameEngine(args)
    breakpoint()

if __name__ == "__main__":
    print("starting")
    print("in the engine script")
    rule_dir_path, rule_name = sys.argv[1], 'rules-05.txt'
    print("rule dir path", rule_dir_path)
    rule_file_path = os.path.join(rule_dir_path, rule_name)
    print("rule file path", rule_file_path)

    # Original debugging args (must have been early since they don't include all necessary args)
    # args = {
    #         'RULE_FILE_PATH' : rule_file_path,
    #         'RULE_NAME'  : rule_name,
    #         'BOARD_SIZE'  : 6,
    #         'OBJECT_SPACE'  : 16,
    #         'COLOR_SPACE'  : 4,
    #         'SHAPE_SPACE'  : 4,
    #         'BUCKET_SPACE'  : 4,
    #         'INIT_OBJ_COUNT'  : 3, 
    #         'R_ACCEPT' : -1,
    #         'R_REJECT' : -1,
    #         'TRAIN_HORIZON' : 300,
    #         'ALPHA' :   1,   
    #         'TEST_EPISODES' :  100,
    #         'TEST_FREQ' :   1000,
    #         'VERBOSE' : 0,
    #         'LR' : 1e-2
    #     }
    # Updated args for debugging
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
        'SEED' : 1,
        'RUN_MODE' : "RULE"
    }
    test_engine(args)
