import argparse
import yaml
import torch
import numpy as np
import os

from .mmDiff import mmDiffRunner


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])

    args = dict()
    args["seed"] = 19960903
    args["config"] = "pysensing/mmwave/PC/model/hpe/mmDiff/mmDiff.yml"
    args["exp"] = "exp"
    args["verbose"] = "info"
    args["ni"] = True
    args["actions"] = "*"
    args["skip_type"] = "uniform"
    args["eta"] = 0.0
    args["sequence"] = True
    args["n_head"] = 4
    args["dim_model"] = 96
    args["n_layer"] = 5
    args["dropout"] = 0.25
    args["downsample"] = 1
    args["model_diff_path"] = None
    args["model_limb_path"] = None
    args["model_pose_path"] = None
    args["train"] = True
    args["test_times"] = 5
    args["test_timesteps"] = 50
    args["test_num_diffusion_timesteps"] = 500


    # parse config file
    with open(os.path.join(args["config"]), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)
    
    # add device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    new_config.device = device
    # update configure file



    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def load_mmDiff(dataset):
    args, config = parse_args_and_config()
    runner = mmDiffRunner(args, config, dataset=dataset)
    return runner
