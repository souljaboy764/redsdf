import numpy as np
import torch
import redsdf.models.nn as nn
from redsdf.models.utils import compute_wmse_per_dim


class RegularizedDeepSignedDistanceFields(torch.nn.Module):
    """ ReDSDF class.
    Args:
        input_dim (int): input dimension, include position and pose
        hidden_sizes (list): number of elements in each of hidden layers, e.g. [512, 512, 512, 512]
        output_dim (int): output dimension, generally 1
        center (numpy.array): object center coordinates
        activation (list or str): activation function of each layer.
        mode_switching_starting_layer (int): the number from which layer the features of sigma are calculated
        mode_switching_hidden_sizes (list or int): number of elements in each of hidden layers to derive the sigma
        mode_switching_alpha_last_layer_nonlinear (str): the nonlinear function in the last layer to compute alpha
        mode_switching_alpha_scale_bias (list): 2-dimensional list to compute scale * activation(alpha) + bias
        mode_switching_rho_last_layer_nonlinear (str): the nonlinear function in the last layer to compute rho
        mode_switching_rho_scale_bias (list): 2-dimensional list to compute scale * activation(rho) + bias
        use_layer_norm (bool): whether to use layer normalization
        use_batch_norm (bool): whether to use batch normalization
        drop_p (float): probability of drop out
        name (str): name of this model
        device (str): device to save this model
    """
    def __init__(self, input_dim, hidden_sizes, center, output_dim=1, activation='relu', mode_switching_starting_layer=2,
                 mode_switching_hidden_sizes=32, mode_switching_alpha_last_layer_nonlinear='softplus',
                 mode_switching_alpha_scale_bias=None, mode_switching_rho_last_layer_nonlinear='sigmoid',
                 mode_switching_rho_scale_bias=None, use_layer_norm=False, use_batch_norm=False,
                 drop_p=0.0, name='', device='cpu'):
        super().__init__()
        self.name = name
        self.device = device
        self.dim_ambient = input_dim
        self.N_constraints = output_dim
        self.nn_model = nn.MLFC(input_dim,
                                hidden_sizes,
                                output_dim,
                                torch.tensor(center).to(self.device),
                                activation=activation,
                                mode_switching_starting_layer=mode_switching_starting_layer,
                                mode_switching_hidden_sizes=mode_switching_hidden_sizes,
                                mode_switching_alpha_last_layer_nonlinear=mode_switching_alpha_last_layer_nonlinear,
                                mode_switching_alpha_scale_bias=mode_switching_alpha_scale_bias,
                                mode_switching_rho_last_layer_nonlinear=mode_switching_rho_last_layer_nonlinear,
                                mode_switching_rho_scale_bias=mode_switching_rho_scale_bias,
                                use_layer_norm=use_layer_norm,
                                use_batch_norm=use_batch_norm,
                                drop_p=drop_p)
        self.nn_model.to(self.device)
        self.is_training = False

    def train(self):
        self.is_training = True
        self.nn_model.train()

    def eval(self):
        self.is_training = False
        self.nn_model.eval()

    def y_torch(self, point_torch, pose_torch):
        x_torch = self.wrap_point_pose(point_torch, pose_torch)
        return self.nn_model(x_torch)

    def y(self, point, pose):
        y_torch = self.y_torch(point, pose)
        y_torch = torch.squeeze(y_torch, dim=0)
        return y_torch.cpu().detach().numpy()

    def J_torch(self, point_torch, pose_torch):
        [_, jac_torch] = self.y_torch_and_J_torch(point_torch, pose_torch)
        return jac_torch

    def J(self, point, pose):
        jac_torch = torch.squeeze(self.J_torch(point, pose), dim=0)
        return jac_torch.cpu().detach().numpy()

    def y_torch_and_J_torch(self, point_torch, pose_torch):
        with torch.enable_grad():
            y_torch = self.y_torch(point_torch, pose_torch)
            jac_torch = torch.stack([torch.autograd.grad(y_torch[:, d].sum(), point_torch,
                                                         retain_graph=True,
                                                         create_graph=self.is_training)[0]
                                     for d in range(y_torch.shape[1])], dim=1)
        return y_torch, jac_torch

    def y_and_J(self, point, pose):
        y_torch, jac_torch = self.y_torch_and_J_torch(point, pose)
        return y_torch.cpu().detach().numpy(), jac_torch.cpu().detach().numpy()

    def wrap_point_pose(self, point, pose):
        point_torch = point.requires_grad_()
        pose_torch = pose
        x_torch = torch.cat([point_torch, pose_torch], dim=1)
        return x_torch

    def forward(self, point, pose):
        return self.y_torch(point, pose)

    def get_loss_components(self, data, poses):
        point_torch = data[:, :3].requires_grad_()
        pose_torch = poses[data[:, -1].int().tolist()]
        norm_level_data_torch = data[:, 6:7]
        norm_level_weight_torch = data[:, 7:8]
        [y_torch, jac_torch] = self.y_torch_and_J_torch(point_torch, pose_torch)
        norm_level_wmse_per_dim = compute_wmse_per_dim(prediction=y_torch,
                                                       ground_truth=norm_level_data_torch,
                                                       weight=norm_level_weight_torch)
        cov_torch_nullspace = data[:, 3:6].unsqueeze(2)
        cov_torch_nullspace_projector = (cov_torch_nullspace @ cov_torch_nullspace.transpose(-2, -1))
        [_, _, J_torch_svd_V] = torch.svd(jac_torch, some=False)
        J_torch_nullspace = J_torch_svd_V[:, :, self.N_constraints:]
        J_torch_nullspace_projector = (J_torch_nullspace @ J_torch_nullspace.transpose(-2, -1))
        J_nspace_proj_error_per_dim = cov_torch_nullspace_projector @ J_torch_nullspace
        cov_nspace_proj_error_per_dim = J_torch_nullspace_projector @ cov_torch_nullspace
        J_nspace_proj_loss_per_dim = (J_nspace_proj_error_per_dim ** 2).mean(axis=0)
        cov_nspace_proj_loss_per_dim = (cov_nspace_proj_error_per_dim ** 2).mean(axis=0)
        deviation_regularization = self.nn_model.rho.mean(axis=0)
        return norm_level_wmse_per_dim, J_nspace_proj_loss_per_dim, cov_nspace_proj_loss_per_dim, \
               deviation_regularization

    def print_prediction_stats(self, data, axis=None):
        pred = self.y(data[:, :3], data[:, 8:])
        pred_mean = pred.mean(axis=axis)
        pred_std = pred.std(axis=axis)
        print("Prediction Stats: [mean, std] = [" + str(pred_mean) + ", " + str(pred_std) + "]")
        return None

    def to(self, device):
        super().to(device)
        self.device = device
        self.nn_model.center = self.nn_model.center.to(device)

