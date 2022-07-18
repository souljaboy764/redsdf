import numpy as np
import torch
from redsdf.envs.dist_field import DistanceField
from redsdf.envs.dist_field.utils import compute_transformation_from_base


class TableDistField(DistanceField):
    def __init__(self, kinematics, smpl_manifold_model, poi_config, device='cuda'):
        super().__init__(kinematics, device)
        self.kinematics = kinematics
        self.smpl_manifold_model = smpl_manifold_model
        self.smpl_manifold_model.to(device)

        self.parse_poi_info(poi_config)

    def get_distance(self, **kwargs):
        robot_base = kwargs['robot_base']
        joint_pos = kwargs['joint_pos']
        table_base = kwargs['table_base']

        T_table_robot = compute_transformation_from_base(table_base, robot_base, self.device)

        poi_pos, poi_jac = self.compute_poi_pos_and_J(joint_pos)

        cart_pos_S = poi_pos @ T_table_robot[:3, :3].T + T_table_robot[:3, 3]
        fake_pose = torch.zeros((poi_pos.shape[0], 1), device=self.device)
        dist, normals_S = self.smpl_manifold_model.y_torch_and_J_torch(cart_pos_S, fake_pose)

        dist = dist.detach()
        normals_S = normals_S.squeeze(1).detach()
        normals_R = normals_S @ T_table_robot[:3, :3]

        return poi_pos.detach(), dist, normals_R, poi_jac

    def parse_poi_info(self, poi_config):
        poi_local_point = list()
        for parent_name, points in poi_config.items():
            for local_point in points:
                local_point = np.array(local_point, dtype=float)
                frame_id = self.kinematics.pino_model.getFrameId(parent_name)
                if frame_id < self.kinematics.pino_model.nframes:
                    self.poi_frame_id.append(frame_id)
                    poi_local_point.append(local_point)

        self.poi_local_point = torch.tensor(np.vstack(poi_local_point), device=self.device).float()
        self.poi_num = len(self.poi_frame_id)
