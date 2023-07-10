import numpy as np
import os, sys, yaml
from rule_game_engine import *
from rule_game_env import *
#from rule_sets import *
from featurization import *
from driver import *

# def objective(trial, args):
#     n_layers = trial.suggest_int('n_layers', 1,4)
#     LR = trial.suggest_float('LR',0.0001,0.01)
#     hidden_sizes = []
#     size = trial.suggest_int('size',50,1000)
#     for i in range(n_layers):
#         hidden_sizes.append(size)
#     args.update({'HIDDEN_SIZES':hidden_sizes})
#     args.update({'LR':LR})
#     results = run_experiment(args)
#     return np.median(results)

# def hyperparameter_tuning(args):
#     import optuna
#     study_name = args["YAML_NAME"]
#     storage_name = "sqlite:///{}{}.db".format(args["OUTPUT_DIR"]+"/",study_name)
#     study = optuna.create_study(study_name=study_name,storage=storage_name,direction = "minimize",load_if_exists=True)
#     study.optimize(lambda trial: objective(trial,args),n_trials=40)

#     print("Study statistics: ")
#     print("  Number of finished trials: ", len(study.trials))

#     print("Best trial:")
#     trial = study.best_trial

#     print("  Value: ", trial.value)

#     print("  Params: ")
#     for key, value in trial.params.items():
#         print("    {}: {}".format(key, value))
    
def rule_run(args, rule_dir_path):
    # Add to rules list as desired
    rules_list = ["1_2_color_4m.txt"]
    computation_batch = 1
    repeats = 2
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
    # For local testing
    if args['RUN_TYPE']=='normal':
        rule_file_path = os.path.join(rule_dir_path, args["RULE_NAME"])
        args.update({'RULE_FILE_PATH' : rule_file_path})
        run_experiment(args)
    # # For hyperparameter tuning runs
    # elif args['RUN_TYPE']=='tune':
    #     rule_file_path = os.path.join(rule_dir_path, args["RULE_NAME"])
    #     args.update({'RULE_FILE_PATH' : rule_file_path})
    #     yaml_name = yaml_path.split("/")[-1].split('.')[0]
    #     args.update({"YAML_NAME":yaml_name})
    #     hyperparameter_tuning(args)
    # # For batch local runs (largely deprecated now with CHTC functionality)
    elif args['RUN_TYPE']=='rule_run':
        yaml_name = yaml_path.split("/")[-1].split('.')[0]
        args.update({"YAML_NAME":yaml_name})
        output_dir = "outputs/rule_runs/"+yaml_name
        args.update({"OUTPUT_DIR":output_dir})
        rule_run(args,rule_dir_path)
    else:
        breakpoint()