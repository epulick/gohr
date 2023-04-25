import os
import sys, yaml
from rule_game_engine import *
from rule_game_env import *
from featurization import *
from driver import *

if __name__ == "__main__":
    if(not os.path.exists("outputs")):
        os.mkdir("outputs")
    rule_dir_path = "gohr/active/captive/game/game-data/rules"
    #yaml_path = sys.argv[1]
    #yaml_path = "/work/active/params/test_cluster_param.yaml"
    yaml_dir= "gohr/active/params/"
    rule_name = str(sys.argv[1])+'.txt'
    yaml_config=str(sys.argv[2])
    cluster_job = str(sys.argv[3])
    cluster_process = str(sys.argv[4])
    cluster_id =cluster_job+"_"+cluster_process
    yaml_path=yaml_dir+yaml_config
    repeats = 3

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
    args.update({"RUN_TYPE":"cluster"})
    args.update({"PARALLEL":False})
    args.update({"BATCH_SIZE":1})
    args.update({"TRAIN_EPISODES":2000})
    yaml_name = yaml_path.split("/")[-1].split('.')[0]
    args.update({"YAML_NAME":yaml_name})
    output_dir = "outputs/"+yaml_name
    args.update({"OUTPUT_DIR":output_dir})
    args.update({"RULE_NAME":rule_name})
    args.update({"REPEAT":repeats})
    args.update({"RECORD":0})
    args.update({"CLUSTER_ID":cluster_id})
    rule_file_path = os.path.join(rule_dir_path, args["RULE_NAME"])
    args.update({'RULE_FILE_PATH' : rule_file_path})
    # Experiment updates for REINFORCE
    args.update({"LEARNER":"REINFORCE"})
    args.update({"LR":0.0006})
    args.update({"ACTIVATION":"LeakyReLU"})
    args.update({"OPTIMIZER":"RMSprop"})
    args.update({"TRAIN_EPISODES":20000})
    args.update({"HIDDEN_SIZES":[1500]})
    #print(args)
    run_experiment(args)

