# Adapted from work by Shubham Bharti and Yiding Chen

import numpy as np
import neptune.new as neptune
import gym, os, sys, yaml, random, torch, copy
from gym import spaces
from joblib import Parallel, delayed
from math import ceil
from rule_game_engine import *
from rule_game_env import *
from rule_sets import *
from featurization import *
from dqn import DQN

# Run a single training trajectory for the learner
def single_execution(args):
    # Gather relevant variables from args for this run
    run_id, exp_dir, exp_id = args['RUN_ID'], args['EXP_DIR'], args['EXP_ID']
    print("RUN : ",run_id)
    #breakpoint()
    # Set the seeds for various aspects of the experiment (generated in the run_experiments function)
    if args['SEED'] == -1:
        args['SEED'] =  int(args["SEEDS1"][run_id])    
        torch.manual_seed(int(args["SEEDS2"][run_id]))     
        np.random.seed(int(args["SEEDS3"][run_id]))
        random.seed(int(args["SEEDS4"][run_id]))
    # Debug run
    elif args['SEED']==-2:
        run_id = args["DEBUG_RUN"]
        args['SEED'] =  int(args["SEEDS1"][run_id])    
        torch.manual_seed(int(args["SEEDS2"][run_id]))     
        np.random.seed(int(args["SEEDS3"][run_id]))
        random.seed(int(args["SEEDS4"][run_id]))
    # Copied from Shubham/Yiding, this structure isn't currently used (10/30/22)
    else:
        seed = run_id + args['SEED']
        args['SEED'] = seed      # if fix_cgs_seed is true then cgs_seed is args['SEED'] for all runs else cgs_seed is run_id 
        torch.manual_seed(seed)     # for each run_id, set seed = run_id + 1 
        np.random.seed(seed)
        random.seed(seed)

    # TO-DO - Make this configurable in the parameter file
    # Choose the desired featurization
    if args['FEATURIZATION']=='NAIVE_BOARD':
        env = NaiveBoard(args)
    elif args['FEATURIZATION']=='NAIVE_N_BS_AS':
        env = Naive_N_Board_Sparse_Action_Sparse(args)
    elif args['FEATURIZATION']=='NAIVE_N_BD_AD':
        env= Naive_N_Board_Dense_Action_Dense(args)
    else:
        breakpoint()

    # Create the agent and train it
    agent = DQN(env,args)
    agent.train()
    error_count = agent.env.error_count
    env.close_channel()
    if args["RECORD"] and agent.run is not None:
        agent.run.stop()

    # print("--------------------")
    # print("--------------------")
    # print("Errors made in this trajectory: ", agent.env.error_count)
    # print("--------------------")
    # print("--------------------")

    # Create the output directory for generated files
    run_dir = os.path.join(exp_dir, str(run_id))
    if (not os.path.exists(run_dir)):
        os.makedirs(run_dir)
    # Write data out
    agent.all_data_df.to_csv(os.path.join(run_dir, 'move_data.csv'))
    agent.episode_df.to_csv(os.path.join(run_dir, 'episode_data.csv'))
    agent.loss_df.to_csv(os.path.join(run_dir, 'loss_data.csv'))
    with open(run_dir+'/data.yaml', 'w') as outfile:
        yaml.dump(args, outfile)
    return agent.env.error_count

def debug_execution(args):
    exp_id =  'debug'
    run_id = 10000
    # Create directory for export, update the args, make the folder if needed
    exp_dir =  os.path.join(args['OUTPUT_DIR'], exp_id +"_"+ args['RULE_NAME'].split('/')[-1].split('.')[0])
    args.update({'EXP_DIR' : exp_dir, 'EXP_ID' : exp_id,'SEED':-2,'RUN_ID':run_id})
    if(not os.path.exists(exp_dir)):
        os.makedirs(exp_dir)
    single_execution(args)
    return
    # outputs = []
    # if args["PARALLEL"] == True:
    #     # Pull in the number of trials and the number of runs to do in parallel (batch size)
    #     num_jobs, batch_size = args['REPEAT'], args['BATCH_SIZE']

    #     # Work through all the jobs, knocking out up to <batch_size> at a time
    #     for batch_id in range(int(num_jobs/batch_size)):
    #         # Generate the list of trials
    #         id_list = np.arange(batch_id*batch_size, (batch_id+1)*batch_size)
    #         #print(id_list)
    #         # Create a list of argument dictionaries to be used across the trials
    #         args_list = []
    #         # Loop over all the trials in this particular batch
    #         for run_id in id_list:
    #             nargs = copy.deepcopy(args)
    #             nargs.update({'RUN_ID':run_id})
    #             args_list.append(nargs)

    #         # Parallelize the runs in this batch
    #         output_list = Parallel(n_jobs=batch_size)(delayed(single_execution)(args) for args in args_list)
    #         outputs.append(output_list)
    # else:
    #     id_list = np.arange(0,args['REPEAT'])
    #     args_list = []
    #     for run_id in id_list:
    #         nargs = copy.deepcopy(args)
    #         nargs.update({'RUN_ID':run_id})
    #         output = single_execution(nargs)
    #         outputs.append(output)

    # # Close out the run as needed
    # if args['RECORD']:
    #     run["params"]=args
    #     run.sync()
    #     run.wait()
    #     run.stop()
    # return outputs

# Run a set of training trajectories for the learner (with the same parameters)
def run_experiment(args):

    # Generate random seeds for engine, pytorch, numpy, and random
    if args['SEED'] == -1:
        seeds1 = np.random.randint(1, 2**32-2, size = args["REPEAT"])
        seeds2 = np.random.randint(1, 2**32-2, size = args["REPEAT"])
        seeds3 = np.random.randint(1, 2**32-2, size = args["REPEAT"])
        seeds4 = np.random.randint(1, 2**32-2, size = args["REPEAT"])
        # Update the seeds in the parameter file
        args.update({"SEEDS1": seeds1, "SEEDS2": seeds2, "SEEDS3": seeds3, "SEEDS4": seeds4})

    # Set up the experiment id (more detailed if recording to Neptune)
    exp_id =  'other'
    if args['RECORD']:
        run = neptune.init_run(
            project="eric-pulick/gohr-test",
            source_files=["dqn.py, driver.py, featurization.py, rule_game_engine.py, rule_game_env.py"],
            mode="sync"
        )
        run_info = run.fetch()
        exp_id = run_info['sys']['id']

    # Create directory for export, update the args, make the folder if needed
    exp_dir =  os.path.join(args['OUTPUT_DIR'], exp_id +"_"+ args['RULE_NAME'].split('/')[-1].split('.')[0])
    args.update({'EXP_DIR' : exp_dir, 'EXP_ID' : exp_id})
    if(not os.path.exists(exp_dir)):
        os.makedirs(exp_dir)
    
    outputs = []
    if args["PARALLEL"] == True:
        # Pull in the number of trials and the number of runs to do in parallel (batch size)
        num_jobs, batch_size = args['REPEAT'], args['BATCH_SIZE']

        # Work through all the jobs, knocking out up to <batch_size> at a time
        for batch_id in range(int(num_jobs/batch_size)):
            # Generate the list of trials
            id_list = np.arange(batch_id*batch_size, (batch_id+1)*batch_size)
            #print(id_list)
            # Create a list of argument dictionaries to be used across the trials
            args_list = []
            # Loop over all the trials in this particular batch
            for run_id in id_list:
                nargs = copy.deepcopy(args)
                nargs.update({'RUN_ID':run_id})
                args_list.append(nargs)

            # Parallelize the runs in this batch
            output_list = Parallel(n_jobs=batch_size)(delayed(single_execution)(args) for args in args_list)
            outputs.append(output_list)
    else:
        id_list = np.arange(0,args['REPEAT'])
        args_list = []
        for run_id in id_list:
            nargs = copy.deepcopy(args)
            nargs.update({'RUN_ID':run_id})
            output = single_execution(nargs)
            outputs.append(output)

    # Close out the run as needed
    if args['RECORD']:
        run["params"]=args
        run.sync()
        run.wait()
        run.stop()
    return outputs

# Function for testing this level of abstraction - may not be updated reliably
def test_driver(args):
    # Test the environment using the yaml input
    if args['FEATURIZATION']=='NAIVE_BOARD':
        env = NaiveBoard(args)
    elif args['FEATURIZATION']=='NAIVE_N':
        env = NaiveBoard_N(args)
    elif args['FEATURIZATION']=='NAIVE_N_DENSE':
        env=NaiveBoard_N_dense(args)
    else:
        breakpoint()
    phi = env.get_feature()
    if args["DEBUG"]==True:
        debug_execution(args)
        return
    # Test the DQN creation
    agent = DQN(env,args)
    #breakpoint()
    agent.train()
    
    output_dir = args["OUTPUT_DIR"]

    agent.all_data_df.to_csv(os.path.join(output_dir, 'move_data.csv'))
    agent.episode_df.to_csv(os.path.join(output_dir, 'episode_data.csv'))
    agent.loss_df.to_csv(os.path.join(output_dir, 'loss_data.csv'))

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
