import numpy as np


class TiagoAPFController:
    def __init__(self, kinematics, apfs, step_size=0.01):
        self.kinematics = kinematics
        self.apfs = apfs
        self.step_size = step_size
        self.action_dim = kinematics.pino_model.nq

        self.velocity_limits = self.kinematics.pino_model.velocityLimit
        self.control_vel = np.zeros(self.action_dim)

    def reset(self):
        for apf in self.apfs:
            apf.reset()

    def update(self, joint_state):
        joint_pos = joint_state[7:28]
        control_cmd = np.zeros(self.action_dim)
        for apf in self.apfs:
            control_cmd += apf.update(joint_pos=joint_pos)

        self.control_vel = self.truncate_velocity(control_cmd)
        des_pos = joint_pos + self.control_vel * self.step_size

        des_pos = np.clip(des_pos,
                          self.kinematics.position_lower_limit,
                          self.kinematics.position_upper_limit)
        return des_pos

    def truncate_velocity(self, joint_velocities):
        joint_velocities = np.clip(joint_velocities, -self.velocity_limits, self.velocity_limits)
        return joint_velocities

    def set_target(self, target):
        self.apfs[0].set_target(target)
