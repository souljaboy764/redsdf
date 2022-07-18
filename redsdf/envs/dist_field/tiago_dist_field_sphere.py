import torch
import numpy as np
from redsdf.envs.dist_field import DistanceField


class TiagoDistFieldSphere(DistanceField):
    def __init__(self, kinematics, poi_config, device='cuda'):
        super().__init__(kinematics, device)
        self.kinematics = kinematics
        self.device = device

        self.poi_frame_id = list()
        self.poi_local_point = None
        self.poi_radius = None
        self.poi_group = None
        self.parse_poi_info(poi_config)
        self.poi_num = len(self.poi_frame_id)

    @property
    def need_balance(self):
        return False

    def get_distance(self, **kwargs):
        joint_pos = kwargs['joint_pos']
        poi_pos, poi_jac = self.compute_poi_pos_and_J(joint_pos)

        dist = torch.zeros(poi_pos.shape[0], 1, device=self.device)
        normals = torch.zeros(poi_pos.shape[0], 3, device=self.device)
        for i, point in enumerate(poi_pos):
            other_group_ids = torch.where(self.poi_group != self.poi_group[i])[0]
            other_group_points = poi_pos[other_group_ids]
            distances = torch.norm(point - other_group_points, dim=1) - self.poi_radius[i] - self.poi_radius[other_group_ids]
            dist_min_id = torch.argmin(distances)
            dist[i] = torch.abs(distances[dist_min_id])
            normals[i] = (point - other_group_points[dist_min_id]) / dist[i]

        # head_pos, head_dist, head_normals, head_jac = self.compute_repulsive_force_head()
        # poi_pos = torch.vstack([poi_pos, head_pos])
        # dist = torch.vstack([dist, head_dist])
        # normals = torch.vstack([normals, head_normals])
        # poi_jac = torch.vstack([poi_jac, head_jac[None, :]])

        return poi_pos, dist, normals, poi_jac

    def parse_poi_info(self, poi_config):
        poi_local_point = list()
        poi_radius = list()
        poi_group = list()
        for parent_name, points in poi_config.items():
            for point in points:
                frame_id = self.kinematics.pino_model.getFrameId(parent_name)
                if not frame_id < self.kinematics.pino_model.nframes:
                    raise ValueError("Unknown parent ")
                local_point = np.array(point[:3], dtype=float)
                self.poi_frame_id.append(frame_id)
                poi_local_point.append(local_point)
                poi_radius.append(point[3])
                poi_group.append(point[4])

        self.poi_local_point = torch.tensor(np.vstack(poi_local_point), device=self.device).float()
        self.poi_radius = torch.tensor(np.array(poi_radius), device=self.device).float()
        self.poi_group = torch.tensor(np.array(poi_group), device=self.device).int()
