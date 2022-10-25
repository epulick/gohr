# Adapted from work by Shubham Bharti and Yiding Chen

import numpy as np
import gym, os, sys, yaml
from gym import spaces
from rule_game_engine import *
from rule_game_env import *
from rule_sets import *
from featurization import *
from dqn import DQN

def single_execution(args):
    pass

def run_experiment(args):

    # Generate random seeds for engine, pytorch, numpy, and random
    if args['SEED'] == -1:
        seeds1 = np.random.randint(1, 2**32-2, size = args["REPEAT"])
        seeds2 = np.random.randint(1, 2**32-2, size = args["REPEAT"])
        seeds3 = np.random.randint(1, 2**32-2, size = args["REPEAT"])
        seeds4 = np.random.randint(1, 2**32-2, size = args["REPEAT"])
        # Update the seeds in the parameter file
        args.update({"SEEDS1": seeds1, "SEEDS2": seeds2, "SEEDS3": seeds3, "SEEDS4": seeds4})

    exp = None

    if(args['RECORD']):
        exp_dir = args['EXP_DIR']
        
def test_driver(args):
    # Test the environment using the yaml input
    if args['FEATURIZATION']=='NAIVE_BOARD':
        env = NaiveBoard(args)
    elif args['FEATURIZATION']=='NAIVE_M1':
        env = NaiveBoard_m1(args)
    else:
        breakpoint()
    phi = env.get_feature()

    # Test the DQN creation
    agent = DQN(env,args)
    #breakpoint()
    agent.train()
    
    output_dir = args["OUTPUT_DIR"]

    agent.all_data_df.to_csv(os.path.join(output_dir, 'move_data.csv'))
    agent.episode_df.to_csv(os.path.join(output_dir, 'episode_data.csv'))
    agent.loss_df.to_csv(os.path.join(output_dir, 'loss_data.csv'))
    
    #breakpoint()

if __name__ == "__main__":
    print("starting driver")

    rule_dir_path = sys.argv[1]
    yaml_path = sys.argv[2]

    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
    u'tag:yaml.org,2002:float',
    re.compile(u'''^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$''', re.X),
    list(u'-+0123456789.'))

    with open(yaml_path, 'r') as param_file:
        args = yaml.load(param_file, Loader = yaml.SafeLoader)

    rule_file_path = os.path.join(rule_dir_path, args["RULE_NAME"])
    args.update({'RULE_FILE_PATH' : rule_file_path})

    test_driver(args)
