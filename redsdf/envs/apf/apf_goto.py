import numpy as np
from .apf_base import ApfBase


class GoToAPF(ApfBase):
    def __init__(self, kinematics, max_action=5, frame_ids=[72, 102], stiffness=5, damping=0.1, i_gain=3, perturb=False):
        super().__init__(kinematics, max_action)
        self.perturb = perturb
        self.target = np.array([[-0.05444, 1.22507, 0.6865],
                                [-0.05744, -1.22507, 0.6865]])
        self.frame_ids = frame_ids

        self.integral = np.zeros_like(self.frame_ids)

        self.stiffness = stiffness
        self.damping = damping
        self.i_gain = i_gain

        self.error_integral = np.zeros(3)
        self.error_prev = None

    def reset(self):
        self.integral = np.zeros_like(self.frame_ids)

    def set_target(self, target):
        assert len(target) == len(self.frame_ids)
        self.target = np.array(target)
        self.error_prev = None
        self.error_integral = np.zeros(3)

    def update(self, **kwargs):
        joint_position = kwargs['joint_pos']
        self.kinematics.forward_kinematics(joint_position)

        action_list = list()

        for i, f_id, target in zip(np.arange(len(self.frame_ids)), self.frame_ids, self.target):
            link_pos = self.kinematics.get_frame(f_id).translation
            link_jac = self.kinematics.get_jacobian(f_id)[:3]
            error = target - link_pos
            dist_to_target = np.linalg.norm(error)
            goto_vec = error
            if self.error_prev is None:
                self.error_prev = error

            if self.perturb:
                goto_vec = self.perturb_direction(goto_vec, dist_to_target, i)

            apf_action = self.stiffness * error + self.damping * (error - self.error_prev) * 60. + \
                         self.i_gain * self.error_integral
            integral_dim = np.where(np.abs(apf_action) < self.max_action)[0]
            self.error_integral[integral_dim] += error[integral_dim] / 60.
            apf_norm = np.linalg.norm(apf_action)
            apf_action = apf_action * np.minimum(self.max_action / apf_norm, 1.)

            self.error_prev = error
            joint_acc = link_jac.T @ apf_action

            action_list.append(joint_acc)

        self.actions = np.array(action_list)

        return self.actions.sum(axis=0)

    def action_magnitude(self, dist_to_target, link_id):
        action_mag = np.minimum(self.stiffness * dist_to_target, self.max_action)
        if action_mag < self.max_action:
            if dist_to_target > 0.01:
                self.integral[link_id] += 0.001 * self.stiffness
            else:
                self.integral[link_id] -= 0.05 * self.stiffness

        self.integral[link_id] = np.clip(self.integral[link_id], 0., self.max_action)

        action_mag = np.maximum(action_mag, self.integral[link_id])
        return action_mag

    def perturb_direction(self, goto_vec, dist_to_target, i):
        orthog_space = np.linalg.svd(goto_vec[:, np.newaxis])[0][:, 1:]
        goto_vec_perturb = goto_vec + orthog_space @ (np.random.randn(2, 1) * 0.1 * dist_to_target * self.integral[i]).squeeze()
        goto_vec_perturb = goto_vec + orthog_space @ (np.random.randn(2, 1) * 0.01).squeeze()
        goto_vec_perturb = goto_vec_perturb / np.linalg.norm(goto_vec_perturb)
        return goto_vec_perturb
