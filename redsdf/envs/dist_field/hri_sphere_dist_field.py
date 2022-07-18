import torch
import numpy as np
from redsdf.envs.dist_field import DistanceField
from redsdf.envs.dist_field.utils import compute_transformation_from_base


class HriDistFieldSphere(DistanceField):
    def __init__(self, kinematics, poi_config, device='cuda'):
        super().__init__(kinematics, device)
        self.kinematics = kinematics
        self.device = device

        self.poi_frame_id = list()
        self.poi_local_point = None
        self.poi_radius = None
        self.table_poi_local_point = None
        self.table_poi_radius = None
        self.hand_radius = None
        self.parse_poi_info(poi_config)
        self.poi_num = len(self.poi_frame_id)

    @property
    def need_balance(self):
        return False

    def get_distance(self, **kwargs):
        robot_base = kwargs['robot_base']
        joint_pos = kwargs['joint_pos']
        hand_pos = kwargs['hand_pos'].reshape(2, 3)
        table_base = kwargs['table_base']

        poi_pos, poi_jac = self.compute_poi_pos_and_J(joint_pos)

        T_robot_table = compute_transformation_from_base(robot_base, table_base, self.device)
        obj_poi_pos = self.table_poi_local_point @ T_robot_table[:3, :3].T + T_robot_table[:3, 3]

        obj_poi_pos = torch.vstack([obj_poi_pos, torch.tensor(hand_pos, device=self.device).float()])
        obj_radius = torch.cat([self.table_poi_radius, self.hand_radius.repeat(2)])

        dist = torch.zeros(poi_pos.shape[0], 1, device=self.device)
        normals = torch.zeros(poi_pos.shape[0], 3, device=self.device)
        for i, point in enumerate(poi_pos):
            distances = torch.norm(point - obj_poi_pos, dim=1) - self.poi_radius[i] - obj_radius
            dist_min_id = torch.argmin(distances)
            dist[i] = torch.abs(distances[dist_min_id])
            normals[i] = (point - obj_poi_pos[dist_min_id]) / dist[i]

        return poi_pos, dist, normals, poi_jac

    def parse_poi_info(self, poi_config):
        poi_local_point = list()
        poi_radius = list()
        table_poi_local_point = list()
        table_poi_radius = list()
        for parent_name, points in poi_config.items():
            if parent_name == "table":
                for point in points:
                    table_poi_local_point.append(np.array(point[:3], dtype=float))
                    table_poi_radius.append(point[3])
            elif parent_name == "hand_radius":
                self.hand_radius = torch.tensor(points, device=self.device).float()
            else:
                for point in points:
                    frame_id = self.kinematics.pino_model.getFrameId(parent_name)
                    if not frame_id < self.kinematics.pino_model.nframes:
                        raise ValueError("Unknown parent ")
                    local_point = np.array(point[:3], dtype=float)
                    self.poi_frame_id.append(frame_id)
                    poi_local_point.append(local_point)
                    poi_radius.append(point[3])

        self.poi_local_point = torch.tensor(np.vstack(poi_local_point), device=self.device).float()
        self.poi_radius = torch.tensor(np.array(poi_radius), device=self.device).float()
        self.table_poi_local_point = torch.tensor(np.array(table_poi_local_point), device=self.device).float()
        self.table_poi_radius = torch.tensor(np.array(table_poi_radius), device=self.device).float()
