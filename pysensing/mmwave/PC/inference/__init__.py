import os
import torch
import torch.nn.functional as F
from torch import nn
from torch.hub import load_state_dict_from_url



PRETRAIN_ROOT = "https://pysensing.oss-ap-southeast-1.aliyuncs.com/pretrain/mmwave_pc/"
PRETRAIN_DICT = {}
PRETRAIN_DICT["mmBody"] = {"P4Transformer": "HPE/mmBody_P4Transformer.pth", 
                            "mmDiff_phase1": "HPE/mmBody_mmDiff_phase1.pth",
                            "mmDiff_phase2": "HPE/mmBody_mmDiff_phase2.pth"}
PRETRAIN_DICT["MetaFi"] = {"PointTransformer": "HPE/MetaFi_PointTransformer.pth", 
                            "mmDiff_phase1": "HPE/MetaFi_mmDiff_phase1.pth",
                            "mmDiff_phase2": "HPE/MetaFi_mmDiff_phase2.pth"}




def load_pretrain(model, dataset_name, model_name, progress=True):
    try:
        # pretrain_path = PRETRAIN_ROOT + PRETRAIN_DICT[dataset_name][model_name]
        # stat = torch.load(pretrain_path)['model_state_dict']
        pretrain_url = PRETRAIN_ROOT + PRETRAIN_DICT[dataset_name][model_name]
        stat = load_state_dict_from_url(pretrain_url, progress=progress)
        model.load_state_dict(stat['model_state_dict'])
        print("Use pretrained model!")
    except:
        print("No registered pretrained model!")
    
    return model
