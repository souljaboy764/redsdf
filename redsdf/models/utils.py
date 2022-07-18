import numpy as np
import torch


def compute_wmse_per_dim(prediction, ground_truth, weight, data_axis=0):
    '''WMSE = Weighted Mean Squared Error'''
    N_dim = ground_truth.shape[1]
    onedimensional_weight = weight.squeeze()
    assert (len(onedimensional_weight.shape) == 1)
    squared_error = ((prediction - ground_truth) ** 2)
    weighted_squared_error = squared_error * onedimensional_weight.unsqueeze(1).repeat(1, N_dim)
    wmse_per_dim = weighted_squared_error.mean(axis=data_axis)
    return wmse_per_dim


def convert_into_pytorch_tensor(variable, device='cpu'):
    if isinstance(variable, torch.Tensor):
        return variable.to(device)
    elif isinstance(variable, np.ndarray):
        return torch.tensor(variable, device=device).float()
    else:
        return torch.tensor(variable, device=device).float()


def convert_into_at_least_2d_pytorch_tensor(variable, device='cpu'):
    tensor_var = convert_into_pytorch_tensor(variable, device=device)
    if len(tensor_var.shape) == 1:
        return tensor_var.unsqueeze(0)
    else:
        return tensor_var
