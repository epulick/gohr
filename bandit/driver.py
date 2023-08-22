# Adapted from work by Shubham Bharti and Yiding Chen

import numpy as np
import os, sys, yaml, random, torch, copy, shutil,gzip
from joblib import Parallel, delayed
from math import ceil
from rule_game_engine import *
from rule_game_env import *
#from rule_sets import *
from featurization import *
from tabular_bandit import tabular_bandit
from meta_bandit import memorization_bandit,meta_bandit

# Run a single training trajectory for the learner
def single_execution(args):
    # Gather relevant variables from args for this run
    run_id, exp_dir, exp_id = args['RUN_ID'], args['EXP_DIR'], args['EXP_ID']
    print("RUN : ",run_id)
    #breakpoint()
    # Set the seeds for various aspects of the experiment (generated in the run_experiments function)
    seed1 = int(args["SEEDS1"][run_id])
    seed2 = int(args["SEEDS2"][run_id])
    seed3 = int(args["SEEDS3"][run_id])
    seed4 = int(args["SEEDS4"][run_id])
    if args['SEED'] == -1:
        args['SEED'] =  seed1   
        torch.manual_seed(seed2)     
        np.random.seed(seed3)
        random.seed(seed4)
    # Debug run
    elif args['SEED']==-2:
        run_id = args["DEBUG_RUN"]
        args['SEED'] =  int(args["SEEDS1"][run_id])    
        torch.manual_seed(int(args["SEEDS2"][run_id]))     
        np.random.seed(int(args["SEEDS3"][run_id]))
        random.seed(int(args["SEEDS4"][run_id]))
    else:
        breakpoint()

    # Choose the desired featurization
    if args['FEATURIZATION']=='NAIVE_BOARD':
        env = NaiveBoard(args)
    elif args['FEATURIZATION']=='NAIVE_N_BS_AS':
        env = Naive_N_Board_Sparse_Action_Sparse(args)
    elif args['FEATURIZATION']=='NAIVE_N_BD_AD':
        env= Naive_N_Board_Dense_Action_Dense(args)
    elif args['FEATURIZATION']=='NAIVE_N_BD_AS':
        env= Naive_N_Board_Dense_Action_Sparse(args)
    elif args['FEATURIZATION']=='NAIVE_N_BS_AD':
        env= Naive_N_Board_Sparse_Action_Dense(args)
    elif args['FEATURIZATION']=='NAIVE_N_BSD_ASD':
        env= Naive_N_Board_SparseDense_Action_SparseDense(args)
    elif args['FEATURIZATION']=='NAIVE_N_BDa_AS':
        env=Naive_N_Board_Dense_alt_Action_Sparse(args)
    elif args['FEATURIZATION']=='INT':
        env=ProcessedEnv(args)
    else:
        breakpoint()

    # Create the output directory for generated files
    run_dir = os.path.join(exp_dir, str(run_id))
    if os.path.exists(run_dir):
        if args["OVERWRITE"]:
            shutil.rmtree(run_dir)
            os.makedirs(run_dir)
    else:
        os.makedirs(run_dir)

    # if (not os.path.exists(run_dir)):
    #     os.makedirs(run_dir)

    move_path = os.path.join(run_dir, 'move_data.csv')
    ep_path = os.path.join(run_dir, 'episode_data.csv')

    log_paths = [move_path,ep_path]
    # Create the agent and train it
    if args["LEARNER"]=="DQN":
        agent = DQN(env,args,log_paths)
    elif args["LEARNER"]=="REINFORCE":
        agent = REINFORCE(env,args,log_paths)
    elif args["LEARNER"]=='BANDIT':
        agent = meta_bandit(env,args,log_paths)
    else:
        breakpoint()

    agent.train()
    error_count = agent.env.error_count
    agent.env.close_channel()
    #env.close_channel()

    # Write data out
    if args["RUN_TYPE"]=='cluster':
        agent.all_data_df['cluster_id']=args['CLUSTER_ID']+'_'+str(run_id)
        agent.episode_df['cluster_id']=args['CLUSTER_ID']+'_'+str(run_id)
    
    args.update({'RUN_ID':run_id.item(),"SEEDS1":seed1,"SEEDS2":seed2,"SEEDS3":seed3,"SEEDS4":seed4})
    with open(run_dir+'/data.yaml', 'w') as outfile:
        yaml.dump(args, outfile)

    # with open(move_path,'rb') as f_in:
    #     with gzip.open(os.path.join(run_dir, 'move_data.gz'), 'wb') as f_out:
    #         shutil.copyfileobj(f_in,f_out)
    # os.remove(move_path)
    return agent.env.error_count

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

    # Set up the experiment id
    if args["RUN_TYPE"]=='cluster':
        exp_id=args["CLUSTER_ID"]
    else:
        exp_id = 'other'

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

    return outputs

# Function for testing this level of abstraction - may not be updated reliably
def test_driver(args):
    # Test the environment using the yaml input
    if args['FEATURIZATION']=='NAIVE_BOARD':
        env = NaiveBoard(args)
    elif args['FEATURIZATION']=='NAIVE_N_BS_AS':
        env = Naive_N_Board_Sparse_Action_Sparse(args)
    elif args['FEATURIZATION']=='NAIVE_N_BD_AD':
        env= Naive_N_Board_Dense_Action_Dense(args)
    elif args['FEATURIZATION']=='RPS':
        env = RandomPieceSelection(args)
    elif args['FEATURIZATION']=='INT':
        env = ProcessedEnv(args)
    else:
        breakpoint()
    #phi = env.get_feature()
    run_id = args['RUN_ID']
    # Create the output directory for generated files
    run_dir = os.path.join(exp_dir, str(run_id))
    if (not os.path.exists(run_dir)):
        os.makedirs(run_dir)

    move_path = os.path.join(run_dir, 'move_data.csv')
    ep_path = os.path.join(run_dir, 'episode_data.csv')

    log_paths = [move_path,ep_path]

    # Test the DQN creation
    agent = meta_bandit(env,args,log_paths)

    agent.train()
    #print(agent.q_values)
    output_dir = args["OUTPUT_DIR"]

    #agent.all_data_df.to_csv(os.path.join(output_dir, 'move_data.csv'))
    #agent.episode_df.to_csv(os.path.join(output_dir, 'episode_data.csv'))

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
    exp_id='debugging'
    # Create directory for export, update the args, make the folder if needed
    exp_dir =  os.path.join(args['OUTPUT_DIR'], exp_id +"_"+ args['RULE_NAME'].split('/')[-1].split('.')[0])
    args.update({'EXP_DIR' : exp_dir, 'EXP_ID' : exp_id})
    args.update({'RUN_ID':0})
    test_driver(args)
