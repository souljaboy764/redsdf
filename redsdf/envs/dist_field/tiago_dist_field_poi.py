import torch
import numpy as np
from redsdf.models.utils import convert_into_at_least_2d_pytorch_tensor
from redsdf.envs.dist_field import DistanceField


class TiagoDistFieldPoI(DistanceField):
    def __init__(self, kinematics, tiago_manifold_model, poi_config, device='cuda'):
        super().__init__(kinematics, device)
        self.tiago_manifold_model = tiago_manifold_model
        self.tiago_manifold_model.to(device)

        self.poi_mirror = list()
        self.poi_joint_id = list()
        self.parse_poi_info(poi_config)

    def get_distance(self, **kwargs):
        joint_pos = kwargs['joint_pos']

        poi_pos, poi_jac = self.compute_poi_pos_and_J(joint_pos)

        poi_pos = convert_into_at_least_2d_pytorch_tensor(poi_pos, device=self.device).float()
        joint_pos = convert_into_at_least_2d_pytorch_tensor(joint_pos[self.poi_joint_id], device=self.device).float()
        poi_pos[self.poi_mirror, 1] = -poi_pos[self.poi_mirror, 1]

        dist, normals = self.tiago_manifold_model.y_torch_and_J_torch(poi_pos, joint_pos)
        normals = normals.squeeze(1)

        cart_pos = poi_pos.detach()
        cart_pos[self.poi_mirror, 1] = -cart_pos[self.poi_mirror, 1]
        normals[self.poi_mirror, 1] = -normals[self.poi_mirror, 1]

        dist = dist.detach()
        normals = normals.detach()

        return cart_pos, dist, normals, poi_jac

    def parse_poi_info(self, poi_config):
        l_arm_ids = [0, 1, 2, 3, 4, 5, 6, 7]
        l_frame_range = [49, 78]

        r_arm_ids = [0, 10, 11, 12, 13, 14, 15, 16]
        r_frame_range = [79, 108]

        poi_local_point = list()
        self.poi_mirror.clear()
        self.poi_joint_id.clear()
        for parent_name, points in poi_config.items():
            for local_point in points:
                local_point = np.array(local_point, dtype=float)
                frame_id = self.kinematics.pino_model.getFrameId(parent_name)
                if frame_id < self.kinematics.pino_model.nframes:
                    self.poi_frame_id.append(frame_id)
                    poi_local_point.append(local_point)
                    if l_frame_range[0] < frame_id < l_frame_range[1]:
                        self.poi_mirror.append(True)
                        self.poi_joint_id.append(r_arm_ids.copy())
                    elif r_frame_range[0] < frame_id < r_frame_range[1]:
                        self.poi_mirror.append(False)
                        self.poi_joint_id.append(l_arm_ids.copy())
                    else:
                        raise ValueError("Parent should be one of the left or right arm")

        self.poi_local_point = torch.tensor(np.vstack(poi_local_point), device=self.device).float()
        self.poi_num = len(self.poi_frame_id)
