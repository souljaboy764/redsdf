import torch
import numpy as np
from scipy.spatial.transform import Rotation


def compute_transformation_from_base(smpl_base, robot_base, device):
    T_w_smpl = np.eye(4)
    T_w_robot = np.eye(4)

    T_w_smpl[:3, 3] = smpl_base[:3]
    T_w_robot[:3, 3] = robot_base[:3]

    T_w_smpl[:3, :3] = Rotation.from_quat(smpl_base[3:]).as_matrix()
    T_w_robot[:3, :3] = Rotation.from_quat(robot_base[3:]).as_matrix()

    T_smpl_robot = np.linalg.inv(T_w_smpl) @ T_w_robot
    return torch.tensor(T_smpl_robot, device=device).float()


def adjoint_inverse(vec, device):
    # Construct skew matrix of translation
    skew_mat = torch.zeros(vec.shape[0], 3, 3, device=device)
    skew_mat[:, 0, 1] -= vec[:, 2]
    skew_mat[:, 1, 0] += vec[:, 2]
    skew_mat[:, 0, 2] += vec[:, 1]
    skew_mat[:, 2, 0] -= vec[:, 1]
    skew_mat[:, 1, 2] -= vec[:, 0]
    skew_mat[:, 2, 1] += vec[:, 0]

    # Construct adjoint matrix
    adj = torch.zeros(vec.shape[0], 6, 6, device=device)
    adj[:, :3, :3] = torch.eye(3)
    adj[:, 3:, 3:] = torch.eye(3)
    adj[:, :3, 3:] += skew_mat
    return adj