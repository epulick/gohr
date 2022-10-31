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
    size = trial.suggest_int('size',100,300)
    decay = trial.suggest_int('decay',200,2000)
    for i in range(n_layers):
        hidden_sizes.append(size)
    args.update({'HIDDEN_SIZES':hidden_sizes})
    args.update({'LR':LR})
    args.update({'EPS_DECAY':decay})
    results = run_experiment(args)
    return np.median(results)

def hyperparameter_tuning(args,hyp):
    study = optuna.create_study(direction = "minimize")
    study.optimize(lambda trial: objective(trial,args),n_trials=200)

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
    if load_hyp:
        with open(hyp_path, 'r') as hyp_file:
            hyp = yaml.load(hyp_file, Loader = yaml.SafeLoader)
    else:
        hyp = None
    rule_file_path = os.path.join(rule_dir_path, args["RULE_NAME"])
    args.update({'RULE_FILE_PATH' : rule_file_path})

    if args['RUN_TYPE']=='normal':
        run_experiment(args)
    elif args['RUN_TYPE']=='tune':
        hyperparameter_tuning(args,hyp)
    else:
        breakpoint()