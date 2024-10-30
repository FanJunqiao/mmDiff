import numpy as np
import torch


def get_intermediate_output(model, input_data, layer_to_hook, multi_output=-1):
    """
    Get the intermediate output of a specific layer in the model.
    
    Parameters:
    - model: PyTorch model instance
    - input_data: Input data
    - layer_to_hook: The specific layer to hook
    - device: Device to run the model on (e.g., 'cpu' or 'cuda')
    
    Returns:
    - intermediate_output: Output of the specified layer
    """

    intermediate_outputs = []
    
    def hook_fn(module, input, output):
        intermediate_outputs.append(output)

    hook = layer_to_hook.register_forward_hook(hook_fn)
    _ = model(input_data)

    if multi_output != -1:
        intermediate_output = intermediate_outputs[0][multi_output].to('cpu').type(torch.FloatTensor)
    elif multi_output == -1:
        intermediate_output = intermediate_outputs[0].to('cpu').type(torch.FloatTensor)
        
    # intermediate_output = intermediate_output.flatten()
    hook.remove()
    return intermediate_output