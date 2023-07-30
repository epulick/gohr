import numpy as np
import os, sys, yaml
from human_decision_model import hl_model

if __name__ == "__main__":
    #rule_dir_path = sys.argv[1]
    exp_path = "/Users/eric/data_analysis/ambiguity4/ep/1_1_color_3m_cua"
    rules = ["1_1_color_3m_cua"]

    loader = yaml.SafeLoader
    # loader.add_implicit_resolver(
    # u'tag:yaml.org,2002:float',
    # re.compile(u'''^(?:
    #  [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    # |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    # |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    # |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    # |[-+]?\\.(?:inf|Inf|INF)
    # |\\.(?:nan|NaN|NAN))$''', re.X),
    # list(u'-+0123456789.'))

    # with open(yaml_path, 'r') as param_file:
    #     args = yaml.load(param_file, Loader = yaml.SafeLoader)
    model = hl_model(exp_path,rules)