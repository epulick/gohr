import numpy as np
import gym, os, sys, yaml, random, torch, copy
from rule_game_engine import *
from rule_game_env import *
from rule_sets import *
from featurization import *
from driver import *
from dqn import DQN

def objective(trial, args):
    n_layers = trial.suggest_int('n_layers', 1,3)
    LR = trial.suggest_float('LR',0.00001,.001)
    hidden_sizes = []
    gamma = trial.suggest_float('gamma',0.01,0.4)
    size = trial.suggest_int('size',100,500)
    decay = trial.suggest_int('decay',100,1000)
    batch = trial.suggest_int('grad_batch',50,300)
    clamp = trial.suggest_int('clamp',0,1)
    optimizer = trial.suggest_categorical('optimizer',['ADAM','RMSprop'])

    for i in range(n_layers):
        hidden_sizes.append(size)
    args.update({'HIDDEN_SIZES':hidden_sizes})
    args.update({'LR':LR})
    args.update({'EPS_DECAY':decay})
    args.update({'GRAD_BATCH_SIZE':batch})
    args.update({'CLAMP':clamp})
    args.update({'GAMMA':gamma})
    args.update({'OPTIMIZER':optimizer})
    results = run_experiment(args)
    return np.median(results)

def hyperparameter_tuning(args):
    import optuna
    study_name = args["YAML_NAME"]
    storage_name = "sqlite:///{}{}.db".format(args["OUTPUT_DIR"]+"/",study_name)
    study = optuna.create_study(study_name=study_name,storage=storage_name,direction = "minimize",load_if_exists=True)
    study.optimize(lambda trial: objective(trial,args),n_trials=20)

    #pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    #complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    #print("  Number of pruned trials: ", len(pruned_trials))
    #print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    
def rule_run(args, rule_dir_path):
    rules_list = ["1_1_shape_4m.txt","1_2_shape_4m.txt", "1_1_shape_3m_cua.txt", 
                "clockwiseZeroStart.txt","clockwiseTwoFree.txt","clockwiseTwoFreeAlt.txt",
                "quadrantNearby.txt","quadrantNearbyTwoFree.txt",
                "1_1_color_4m.txt","1_2_color_4m.txt","1_1_color_3m_cua.txt",
                "bottom_then_top.txt","bottomLeft_then_topRight.txt"]
                #"topLeft_then_bottomRight.txt","topRight_then_bottomLeft.txt"]
    #rules_list = ["1_2_color_4m.txt","1_1_color_3m_cua.txt"]
    computation_batch = 8
    repeats = 56
    for rule in rules_list:
        args.update({"RULE_NAME":rule})
        args.update({"BATCH_SIZE":computation_batch})
        args.update({"REPEAT":repeats})
        rule_file_path = os.path.join(rule_dir_path, args["RULE_NAME"])
        args.update({'RULE_FILE_PATH' : rule_file_path})
        run_experiment(args)

def cluster_rule_run(args, rule_dir_path):
    rules_list = ["1_1_shape_4m.txt","1_2_shape_4m.txt", "1_1_shape_3m_cua.txt", 
                "clockwiseZeroStart.txt","clockwiseTwoFree.txt","clockwiseTwoFreeAlt.txt",
                "quadrantNearby.txt","quadrantNearbyTwoFree.txt",
                "1_1_color_4m.txt","1_2_color_4m.txt","1_1_color_3m_cua.txt",
                "bottom_then_top.txt","bottomLeft_then_topRight.txt"]
                #"topLeft_then_bottomRight.txt","topRight_then_bottomLeft.txt"]
    #rules_list = ["1_2_color_4m.txt","1_1_color_3m_cua.txt"]
    computation_batch = 8
    repeats = 56
    for rule in rules_list:
        args.update({"RULE_NAME":rule})
        args.update({"BATCH_SIZE":computation_batch})
        args.update({"REPEAT":repeats})
        rule_file_path = os.path.join(rule_dir_path, args["RULE_NAME"])
        args.update({'RULE_FILE_PATH' : rule_file_path})
        run_experiment(args)

if __name__ == "__main__":
    rule_dir_path = sys.argv[1]
    yaml_path = sys.argv[2]
    # Eventually we can pass the tuning values in this way, for now they are hardcoded
    if len(sys.argv)>3:
        hyp_path = sys.argv[3]
        load_hyp = True
    else:
        load_hyp = False

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

    if args['RUN_TYPE']=='normal':
        rule_file_path = os.path.join(rule_dir_path, args["RULE_NAME"])
        args.update({'RULE_FILE_PATH' : rule_file_path})
        run_experiment(args)
    elif args['RUN_TYPE']=='tune':
        rule_file_path = os.path.join(rule_dir_path, args["RULE_NAME"])
        args.update({'RULE_FILE_PATH' : rule_file_path})
        yaml_name = yaml_path.split("/")[-1].split('.')[0]
        args.update({"YAML_NAME":yaml_name})
        hyperparameter_tuning(args)
    elif args['RUN_TYPE']=='rule_run':
        yaml_name = yaml_path.split("/")[-1].split('.')[0]
        args.update({"YAML_NAME":yaml_name})
        output_dir = "outputs/rule_runs/"+yaml_name
        args.update({"OUTPUT_DIR":output_dir})
        #rule_run(args,rule_dir_path)
        #if (not os.path.exists(output_dir)):
        #    rule_run(args,rule_dir_path)
        #else:
        #    breakpoint()
        rule_run(args,rule_dir_path)
    else:
        breakpoint()