from typing import Optional, Tuple,Union, Dict
from torch import Tensor, device
import torch
import torch.nn as nn
from pysensing.mmwave.PC.inference.utils import get_intermediate_output


from pysensing.mmwave.PC.dataset.har import load_har_dataset
from pysensing.mmwave.PC.dataset.hgr import load_hgr_dataset
from pysensing.mmwave.PC.dataset.hpe import load_hpe_dataset
from pysensing.mmwave.PC.model.hpe import load_hpe_model, load_hpe_pretrain
from pysensing.mmwave.PC.inference import load_pretrain


def embedding(input: Tensor, model: nn.Module, dataset_name: str, model_name: str, device: device) -> Tensor:
    r'''
    Obtain the embedding of the input mmWave radar data.

    Args:
        input (Tensor): data to be inferenced, which is required to be the same size as the dataset provided.
        model (nn.module): the pretrained model
        dataset_name (str): the dataset name. Slected from ['radHAR', 'M-Gesture', 'mmBody', 'MetaFi']
        model_name (str): the model name. Select from ['har_MLP', 'har_LSTM', 'EVL_NN', 'P4Transformer', 'PointTransformer']
        device(device): the devive used for inference

    Return:
        output (Tensor): the audio embedding

    Example:
        >>> input = torch.zeros(1,4,500,5)
        >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        >>> model = pysenesing.mmwave.PC.models.har.har_MLP()
        >>> output = har_embedding(input, model, dataset_name = "radHAR", model_name = "har_MLP", device)
    '''

    dataset_shape = {'radHAR': [1, 60, 10, 32, 32], 'M-Gesture': [1, 28, 50, 5], "mmBody": [1, 4, 5000, 6], "MetaFi": [1, 5, 150, 5]}
    multi_output = -1

    if dataset_name not in dataset_shape:
        raise ValueError("Unsupported dataset. Please choose from ['radHAR', 'M-Gesture', 'mmBody', 'MetaFi'].")

    expected_shape = dataset_shape[dataset_name]
    if torch.equal(torch.Tensor(input.shape), torch.Tensor(expected_shape)):
        print("Found input shape: ", input.shape)
        raise ValueError(f"The shape of the input PC data does not match the expected shape for dataset '{dataset_name}'.")
    
    if dataset_name == 'radHAR' and model_name == "har_MLP":
        intermediate_feature_layer = model.fc
    elif dataset_name == 'radHAR' and model_name == "har_lstm":
        intermediate_feature_layer = model.lstm
        multi_output = 0
    elif dataset_name == 'M-Gesture' and model_name == "EVL_NN":
        intermediate_feature_layer = model.F
        multi_output = 0
    elif dataset_name == 'mmBody' and model_name == "P4Transformer":
        intermediate_feature_layer = model.transformer
    elif dataset_name == 'MetaFi' and model_name == "PointTransformer":
        intermediate_feature_layer = model.fc2
    else:
        raise ValueError("Unsupported dataset. Please use ['radHAR', 'M-Gesture', 'mmBody', 'MetaFi'] and ['har_MLP', 'har_LSTM', 'EVL_NN', 'P4Transformer', 'PointTransformer'].")

    if torch.is_tensor(input) == False:
        torch.tensor(input, device=device)
    else: input.to(device)
    model.to(device)
    feature_embedding = get_intermediate_output(model,input,intermediate_feature_layer, multi_output=multi_output)
    return feature_embedding


