import numpy as np
import torch
import torch.nn.functional

from .apf_base import ApfBase


class CollisionAvoidanceAPF(ApfBase):
    def __init__(self, kinematics, dist_fields, max_action=1.5, dist_effect=0.1, device='cpu'):
        super().__init__(kinematics, max_action)
        self.dist_fields = dist_fields
        self.dist_effect = dist_effect
        self.beta = max_action / (self.dist_effect ** 2)
        self.device = device

    def update(self, **kwargs):
        joint_position = kwargs['joint_pos']
        joint_velocities = np.zeros(self.kinematics.pino_model.nq)
        for dist_field in self.dist_fields:
            # point_pos_R, point_jac_R = dist_field.compute_poi_pos_and_J(joint_position)

            _, dist, normal, point_jac_R = dist_field.get_distance(**kwargs)

            vel_mag = self.velocity_magnitude(dist)
            cart_velocity = vel_mag * normal

            joint_velocity = point_jac_R.transpose(1, 2) @ cart_velocity.unsqueeze(2)
            joint_velocities += joint_velocity.squeeze().sum(0).cpu().numpy()

            if dist_field.need_balance:
                joint_velocities[0] -= 3.0

        return joint_velocities

    def velocity_magnitude(self, dist):
        dist = torch.clip(dist, min=torch.zeros_like(dist), max=torch.ones_like(dist)*self.dist_effect)
        return -self.max_action / self.dist_effect * (dist - self.dist_effect)

