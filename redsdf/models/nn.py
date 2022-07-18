import numpy as np
import torch
import torch.nn as nn


class Identity(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


ACTIVATIONS = {
    'tanh': torch.nn.Tanh,
    'relu': torch.nn.ReLU,
    'elu': torch.nn.ELU,
    'identity': Identity,
    'sigmoid': torch.nn.Sigmoid,
    'softplus': torch.nn.Softplus
}


class MLFC(torch.nn.Module):
    """ Fully connected multi-layers class.

    Args:
        input_size (int): input dimension, include position and pose
        hidden_sizes (list): number of elements in each of hidden layers, e.g. [512, 512, 512, 512]
        output_size (int): output dimension, generally 1
        center (numpy.array): object center coordinates
        activation (list or str): activation function of each layer.
        mode_switching_starting_layer (int): the number from which layer the features of sigma are calculated
        mode_switching_hidden_sizes (list or int): number of elements in each of hidden layers to derive the sigma
        mode_switching_alpha_last_layer_nonlinear (str): the activation function in the last layer to compute alpha
        mode_switching_alpha_scale_bias (list): 2-dimensional list contains the gain and bias of the activation function of alpha
        mode_switching_rho_last_layer_nonlinear (str): the activation function in the last layer to compute rho
        mode_switching_rho_scale_bias (list): 2-dimensional list contains the gain and bias of the activation function of rho
        mode_switching_activation (list or str): the activation function of hidden layers to compute sigma
        use_layer_norm (bool): whether to use layer normalization
        use_batch_norm (bool): whether to use batch normalization
        drop_p (float): probability of drop out
    """

    def __init__(self, input_size, hidden_sizes, output_size, center, activation='relu', mode_switching_starting_layer=2,
                 mode_switching_hidden_sizes=32, mode_switching_alpha_last_layer_nonlinear='softplus',
                 mode_switching_alpha_scale_bias=None, mode_switching_rho_last_layer_nonlinear='sigmoid',
                 mode_switching_rho_scale_bias=None, mode_switching_activation=None,
                 use_layer_norm=False, use_batch_norm=False, drop_p=0.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.center = center
        self.activation = activation
        self.gain_bias_alpha = [1, 1] if mode_switching_alpha_scale_bias is None else mode_switching_alpha_scale_bias
        self.gain_bias_rho = [1, 0.5] if mode_switching_rho_scale_bias is None else mode_switching_rho_scale_bias
        self.reg_alpha = ACTIVATIONS[mode_switching_alpha_last_layer_nonlinear]()
        self.reg_rho = ACTIVATIONS[mode_switching_rho_last_layer_nonlinear]()
        assert (mode_switching_starting_layer < len(hidden_sizes))
        topology = [input_size] + hidden_sizes + [output_size]
        if isinstance(mode_switching_hidden_sizes, int):
            mode_switching_hidden_sizes = [mode_switching_hidden_sizes]
        mode_switching_hidden_sizes = [topology[mode_switching_starting_layer]] + mode_switching_hidden_sizes

        if mode_switching_activation is None:
            if isinstance(activation, str):
                mode_switching_activation = [activation] * (len(mode_switching_hidden_sizes) - 1)
            else:
                mode_switching_activation = ["relu"] * (len(mode_switching_hidden_sizes) - 1)
        elif isinstance(mode_switching_activation, str):
            mode_switching_activation = [mode_switching_activation] * (len(mode_switching_hidden_sizes) - 1)
        else:
            mode_switching_activation = list(mode_switching_activation)

        if isinstance(activation, str):
            activation = [activation] * len(hidden_sizes)
        else:
            activation = list(activation)
            assert (len(activation) == len(hidden_sizes))

        layers1 = []
        layers2 = []
        layers_sig = []
        for i in range(mode_switching_starting_layer):
            layers1.append(torch.nn.Linear(topology[i], topology[i + 1]))
            if use_batch_norm:
                layers1.append(torch.nn.BatchNorm1d(topology[i + 1]))
            layers1.append(ACTIVATIONS[activation[i]]())
            if use_layer_norm:
                layers1.append(torch.nn.LayerNorm(topology[i + 1]))
            if not np.isclose(drop_p, 0):
                layers1.append(torch.nn.Dropout(p=drop_p))
        for i in range(mode_switching_starting_layer, len(topology) - 2):
            layers2.append(torch.nn.Linear(topology[i], topology[i + 1]))
            if use_batch_norm:
                layers2.append(torch.nn.BatchNorm1d(topology[i + 1]))
            layers2.append(ACTIVATIONS[activation[i]]())
            if use_layer_norm:
                layers2.append(torch.nn.LayerNorm(topology[i + 1]))
            if not np.isclose(drop_p, 0):
                layers2.append(torch.nn.Dropout(p=drop_p))
        for i in range(len(mode_switching_hidden_sizes) - 1):
            layers_sig.append(torch.nn.Linear(mode_switching_hidden_sizes[i], mode_switching_hidden_sizes[i + 1]))
            if use_batch_norm:
                layers_sig.append(torch.nn.BatchNorm1d(mode_switching_hidden_sizes[i + 1]))
            layers_sig.append(ACTIVATIONS[mode_switching_activation[i]]())
            if use_layer_norm:
                layers_sig.append(torch.nn.LayerNorm(mode_switching_hidden_sizes[i + 1]))
            if not np.isclose(drop_p, 0):
                layers_sig.append(torch.nn.Dropout(p=drop_p))

        layers2.append(torch.nn.Linear(topology[-2], topology[-1]))
        layers_sig.append(torch.nn.Linear(mode_switching_hidden_sizes[-1], 2))
        self.mlp1 = torch.nn.Sequential(*layers1)
        self.mlp2 = torch.nn.Sequential(*layers2)
        self.mlp_mode_switching = torch.nn.Sequential(*layers_sig)
        self.apply(initialize_weights)

    def forward(self, inp):
        x = self.mlp1(inp)
        bd = self.mlp_mode_switching(x)
        x = self.mlp2(x)
        alpha = self.gain_bias_alpha[0] * self.reg_alpha(bd[:, 0:1]) + self.gain_bias_alpha[1]
        self.rho = self.gain_bias_rho[0] * self.reg_rho(bd[:, 1:2]) + self.gain_bias_rho[1]
        n = torch.norm(inp[:, :3] - self.center.reshape(1, -1), dim=1).reshape(-1, 1)
        sigma = torch.sigmoid(alpha * (n - self.rho))
        return (1 - sigma) * x + sigma * n

 
def initialize_weights(m):
    if isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)
