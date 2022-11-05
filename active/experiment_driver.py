import numpy as np
import gym, os, sys, yaml, random, torch, copy, optuna
from rule_game_engine import *
from rule_game_env import *
from rule_sets import *
from featurization import *
from driver import *
from dqn import DQN

def objective(trial, args):
    n_layers = trial.suggest_int('n_layers', 1,2)
    LR = trial.suggest_float('LR',0.00001,.001)
    hidden_sizes = []
    gamma = trial.suggest_float('gamma',0.01,0.5)
    size = trial.suggest_int('size',100,400)
    decay = trial.suggest_int('decay',200,2000)
    batch = trial.suggest_int('grad_batch',50,300)
    clamp = trial.suggest_int('clamp',0,1)
    #optimizer = trial.suggest_categorical('optimizer',['SGD','ADAM','RMSprop'])

    for i in range(n_layers):
        hidden_sizes.append(size)
    args.update({'HIDDEN_SIZES':hidden_sizes})
    args.update({'LR':LR})
    args.update({'EPS_DECAY':decay})
    args.update({'GRAD_BATCH':batch})
    args.update({'CLAMP':clamp})
    args.update({'GAMMA':gamma})
    #args.update({'OPTIMIZER':optimizer})
    results = run_experiment(args)
    return np.median(results)

def hyperparameter_tuning(args):
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

    rule_file_path = os.path.join(rule_dir_path, args["RULE_NAME"])
    args.update({'RULE_FILE_PATH' : rule_file_path})

    if args['RUN_TYPE']=='normal':
        run_experiment(args)
    elif args['RUN_TYPE']=='tune':
        yaml_name = yaml_path.split("/")[-1].split('.')[0]
        args.update({"YAML_NAME":yaml_name})
        hyperparameter_tuning(args)
    else:
        breakpoint()