import torch
import numpy as np
from redsdf.envs.dist_field.utils import adjoint_inverse


class DistanceField:
    def __init__(self, kinematics, device='cuda'):
        self.kinematics = kinematics
        self.device = device

        self.poi_num = 0
        self.poi_frame_id = list()
        self.poi_local_point = None

    def parse_poi_info(self, poi_config):
        raise NotImplementedError

    def get_distance(self, **kwargs):
        raise NotImplementedError

    @property
    def need_balance(self):
        return False

    def compute_poi_pos_and_J(self, joint_position):
        self.kinematics.forward_kinematics(joint_position)

        parent_trans = list()
        parent_rot = list()
        parent_jac = list()

        for i in range(self.poi_num):
            parent_frame = self.kinematics.get_frame(self.poi_frame_id[i])
            parent_trans.append(parent_frame.translation)
            parent_rot.append(parent_frame.rotation)
            parent_jac.append(self.kinematics.get_jacobian(self.poi_frame_id[i]))

        local_pos = self.poi_local_point
        parent_trans = torch.tensor(np.array(parent_trans), device=self.device).float()
        parent_rot = torch.tensor(np.array(parent_rot), device=self.device).float()
        parent_jac = torch.tensor(np.array(parent_jac), device=self.device).float()

        local_pos_W = torch.matmul(parent_rot, local_pos[:, :, None]).squeeze(2)
        adj_inverse = adjoint_inverse(-local_pos_W, device=self.device)

        point_pos_W = parent_trans + local_pos_W
        point_jac_W = adj_inverse @ parent_jac

        return point_pos_W, point_jac_W[:, :3]
