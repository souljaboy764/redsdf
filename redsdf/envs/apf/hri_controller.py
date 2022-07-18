import numpy as np


class HriController:
    def __init__(self, kinematics, apfs, step_size=0.01):
        self.kinematics = kinematics
        self.apfs = apfs
        self.step_size = step_size
        self.action_dim = kinematics.pino_model.nq

        self.velocity_limits = self.kinematics.pino_model.velocityLimit
        self.control_vel = np.zeros(self.action_dim)

    def update(self, state):
        robot_base = state[:7]
        joint_pos = state[7:28]
        smpl_base = state[49:56]
        smpl_pose = state[56:119]
        hand_pos = state[119: 125]
        table_base = state[125:132]
        control_cmd = np.zeros(self.action_dim)
        for apf in self.apfs:
            control_cmd += apf.update(joint_pos=joint_pos, robot_base=robot_base, smpl_base=smpl_base,
                                      smpl_pose=smpl_pose, hand_pos=hand_pos, table_base=table_base)

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