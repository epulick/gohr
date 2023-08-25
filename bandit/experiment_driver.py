import numpy as np
import os, sys, yaml
from rule_game_engine import *
from rule_game_env import *
#from rule_sets import *
from featurization import *
from driver import *
    
def rule_run(args, rule_dir_path):
    # Add to rules list as desired
    rules_list = [  "1_1_color_4m.txt","1_1_color_3m_cua.txt","1_2_color_4m.txt",
                    "1_1_shape_4m.txt","1_1_shape_3m_cua.txt","1_2_shape_4m.txt",
                    "quadrantNearby.txt","quadrantNearbyTwoFree.txt",
                    "clockwiseZeroStart.txt", "clockwiseTwoFreeAlt.txt","clockwiseTwoFree.txt",
                    "bottomLeft_then_topRight.txt","bottom_then_top.txt"
                    ]
    computation_batch = 8
    repeats = 500
    for rule in rules_list:
        args.update({"RULE_NAME":rule})
        args.update({"PARALLEL":True})
        args.update({"BATCH_SIZE":computation_batch})
        args.update({"REPEAT":repeats})
        rule_file_path = os.path.join(rule_dir_path, args["RULE_NAME"])
        args.update({'RULE_FILE_PATH' : rule_file_path})
        args.update({'OUTPUT_DIR':'outputs/meta_bandit3'})
        args.update({'OVERWRITE':True})
        run_experiment(args)

def data_generator(args, rule_dir_path):
    rule_file_path = os.path.join(rule_dir_path, args["RULE_NAME"])
    args.update({"REPEAT":3})
    args.update({'RULE_FILE_PATH' : rule_file_path})
    feats = ['shape','color','row','col','quadrant','cell','bucket1','bucket2','bucket3','bucket4']
    rules_list = [  "1_1_color_4m.txt","1_1_color_3m_cua.txt","1_2_color_4m.txt",
                "1_1_shape_4m.txt","1_1_shape_3m_cua.txt","1_2_shape_4m.txt",
                "quadrantNearby.txt","quadrantNearbyTwoFree.txt",
                "clockwiseZeroStart.txt", "clockwiseTwoFreeAlt.txt","clockwiseTwoFree.txt",
                "bottomLeft_then_topRight.txt","bottom_then_top.txt"
                ]
    for feat in feats:
        args.update({'OUTPUT_DIR':os.path.join('outputs/data_generator',feat)})
        args.update({'MODEL_FEATURES': [feat]})
        for rule in ["1_1_shape_4m.txt"]:
            args.update({"RULE_NAME":rule})
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
    elif args['RUN_TYPE']=='data_generator':
        data_generator(args,rule_dir_path)
    else:
        breakpoint()