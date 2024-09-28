import os
import torch
import torch.nn.functional as F
from torch import nn



PRETRAIN_ROOT = "/home/junqiao/pysensing/pretrained/"
PRETRAIN_DICT = {}
PRETRAIN_DICT["mmBody"] = {"P4Transformer": "mmBody_P4Transformer1.pth", 
                            "mmDiff": "mmBody_mmDiff.pth"}
PRETRAIN_DICT["MetaFi"] = {"PointTransformer": "MetaFi_PointTransformer.pth", 
                            "mmDiff": "MetaFi_mmDiff.pth"}
PRETRAIN_DICT["radHAR"] = {"MLP": "radHAR_MLP.pth", 
                            "LSTM": "radHAR_LSTM.pth"}
PRETRAIN_DICT["M-Gesture"] = {"EVL_NN": "M-Gesture_EVL_NN.pth"}



def load_pretrain(model, dataset_name, model_name):
    try:
        pretrain_path = os.path.join(PRETRAIN_ROOT + PRETRAIN_DICT[dataset_name][model_name])
        stat = torch.load(pretrain_path)['model_state_dict']
        model.load_state_dict(stat)
        print("Use pretrained model!")
    except:
        print("No registered pretrained model!")
    
    return model
