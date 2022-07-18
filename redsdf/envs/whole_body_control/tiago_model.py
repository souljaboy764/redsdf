import os
import numpy as np
import pinocchio as pino
import redsdf


class TiagoModel:
    def __init__(self, client, urdf_file=None, random_init=False, use_head=False, use_gripper=False,
                 control_mode='position'):
        self.client = client
        if urdf_file is None:
            self.urdf_file = os.path.dirname(redsdf.package_dir) + "/object_models/tiago_urdf/tiago_no_wheel.urdf"
        else:
            self.urdf_file = urdf_file
        self.random_init = random_init
        self.use_head = use_head
        self.use_gripper = use_gripper
        self.control_mode = control_mode

        self.model_id = self.client.loadURDF(self.urdf_file, useFixedBase=True,
                                             flags=self.client.URDF_USE_SELF_COLLISION)
        self.kinematics = TiagoKinematics(self.urdf_file)

        self.joint_names = ['torso_lift_joint',
                            'arm_left_1_joint', 'arm_left_2_joint', 'arm_left_3_joint', 'arm_left_4_joint',
                            'arm_left_5_joint', 'arm_left_6_joint', 'arm_left_7_joint',
                            'gripper_left_right_finger_joint', 'gripper_left_left_finger_joint',
                            'arm_right_1_joint', 'arm_right_2_joint', 'arm_right_3_joint', 'arm_right_4_joint',
                            'arm_right_5_joint', 'arm_right_6_joint', 'arm_right_7_joint',
                            'gripper_right_right_finger_joint', 'gripper_right_left_finger_joint',
                            'head_1_joint', 'head_2_joint']
        self.render_link_mask = [0, 1, 9, 11, 20, 21, 22, 23,
                                 31, 32, 33, 34, 35, 36, 42, 43, 44,
                                 46, 47, 48, 49, 50, 51, 57, 58, 59, 61, 62]

        self.pb_joint_ids = np.zeros(len(self.joint_names), dtype=int)
        for idx in range(self.client.getNumJoints(self.model_id)):
            info = self.client.getJointInfo(self.model_id, idx)
            if info[1].decode('ascii') in self.joint_names:
                self.pb_joint_ids[self.joint_names.index(info[1].decode('ascii'))] = int(info[0])
        self.collision_disabled_link_ids = [38, 39, 40, 41, 42, 53, 54, 55, 56, 57]

        self.set_collision_mask()

        self.default_init_position = np.array([0.,
                                               0., 0., np.pi / 2, np.pi/2., 0., 0., 0., 0.0225, 0.0225,
                                               0., 0., np.pi / 2, np.pi/2., 0., 0., 0., 0.0225, 0.0225,
                                               0., 0.])

        self.client.setJointMotorControlArray(self.model_id, self.pb_joint_ids,
                                          self.client.VELOCITY_CONTROL, forces=np.zeros((len(self.pb_joint_ids), 1)))
        self.state = None
        self.init_position = None

    @property
    def num_joints(self):
        return self.kinematics.num_joints

    def set_collision_mask(self):
        # set all links to a default value: Group: 0000001, Mask: 1111000
        for idx in range(self.client.getNumJoints(self.model_id)):
            self.client.setCollisionFilterGroupMask(self.model_id, idx, collisionFilterGroup=int('00000001', 2),
                                                    collisionFilterMask=int('11111000', 2))

        # set upper left arm links to: Group: 0000010, Mask: 1110100
        for idx in self.pb_joint_ids[1:4]:
            self.client.setCollisionFilterGroupMask(self.model_id, idx, collisionFilterGroup=int('00000010', 2),
                                                    collisionFilterMask=int('11110100', 2))

        # set upper right arm links to: Group: 0000100, Mask: 1101010
        for idx in self.pb_joint_ids[10:13]:
            self.client.setCollisionFilterGroupMask(self.model_id, idx, collisionFilterGroup=int('00000100', 2),
                                                    collisionFilterMask=int('11101010', 2))

        # set lower left arm links to: Group: 0001000, Mask: 1010101
        for idx in self.pb_joint_ids[4:8]:
            self.client.setCollisionFilterGroupMask(self.model_id, idx, collisionFilterGroup=int('00001000', 2),
                                                    collisionFilterMask=int('11010101', 2))

        # set lower right arm links to: Group: 0010000, Mask: 0101011
        for idx in self.pb_joint_ids[13:17]:
            self.client.setCollisionFilterGroupMask(self.model_id, idx, collisionFilterGroup=int('00010000', 2),
                                                    collisionFilterMask=int('10101011', 2))

        # set left gripper links to: Group: 0100000, Mask: 1010111
        for idx in self.pb_joint_ids[8:10]:
            self.client.setCollisionFilterGroupMask(self.model_id, idx, collisionFilterGroup=int('00100000', 2),
                                                    collisionFilterMask=int('11010111', 2))

        # set right gripper links to: Group: 1000000, Mask: 0101111
        for idx in self.pb_joint_ids[17:19]:
            self.client.setCollisionFilterGroupMask(self.model_id, idx, collisionFilterGroup=int('01000000', 2),
                                                    collisionFilterMask=int('10101111', 2))

        # set all gripper links to : Group: 0000000, Mask: 0000000
        for idx in self.collision_disabled_link_ids:
            self.client.setCollisionFilterGroupMask(self.model_id, idx, collisionFilterGroup=int('0000000', 2),
                                                    collisionFilterMask=int('00000000', 2))

    def reset(self, state=None):
        self.client.resetBasePositionAndOrientation(self.model_id, [0., 0., 0.], [0., 0., 0., 1.])
        if state is not None:
            self.init_position = state
        elif self.random_init:
            self.init_position = np.random.uniform(self.kinematics.pino_model.lowerPositionLimit,
                                                   self.kinematics.pino_model.upperPositionLimit)
            if not self.use_head:
                self.init_position[19:21] = 0.
            if not self.use_gripper:
                mid_pos = (self.kinematics.pino_model.lowerPositionLimit +
                           self.kinematics.pino_model.upperPositionLimit) / 2
                self.init_position[8:10] = mid_pos[8:10]
                self.init_position[17:19] = mid_pos[17:19]
        else:
            self.init_position = self.default_init_position

        self.client.resetJointStatesMultiDof(self.model_id, self.pb_joint_ids, self.init_position[:, np.newaxis])
        self.state = self.get_state()
        return self.state

    def get_state(self):
        joint_pos = list()
        joint_vel = list()
        for idx in self.pb_joint_ids:
            joint_state = self.client.getJointState(self.model_id, idx)
            joint_pos.append(joint_state[0])
            joint_vel.append(joint_state[1])
        base_state = self.client.getBasePositionAndOrientation(self.model_id)
        return np.concatenate([base_state[0], base_state[1], joint_pos, joint_vel])

    def apply_action(self, action):
        if self.control_mode == 'velocity':
            self.client.setJointMotorControlArray(self.model_id, self.pb_joint_ids,
                                                  self.client.VELOCITY_CONTROL, targetVelocities=action)
        elif self.control_mode == 'position':
            self.client.setJointMotorControlArray(self.model_id, self.pb_joint_ids,
                                                  self.client.POSITION_CONTROL, targetPositions=action)
        else:
            raise NotImplementedError

        if not self.use_gripper:
            self.client.setJointMotorControlArray(self.model_id, self.pb_joint_ids[[8, 9, 17, 18]],
                                                  self.client.POSITION_CONTROL,
                                                  targetPositions=np.array([[0.0225], [0.0225], [0.0225], [0.0225]]))


class TiagoKinematics:
    def __init__(self, urdf_file):
        self.pino_model = pino.buildModelFromUrdf(urdf_file)
        self.pino_data = self.pino_model.createData()

        self.num_joints = self.pino_model.nq
        self.current_joint_pos = None

    def forward_kinematics(self, joint_position):
        if np.any(self.current_joint_pos != joint_position):
            pino.forwardKinematics(self.pino_model, self.pino_data, joint_position)
            pino.computeJointJacobians(self.pino_model, self.pino_data, joint_position)
            pino.updateFramePlacements(self.pino_model, self.pino_data)
            self.current_joint_pos = joint_position.copy()

    def get_frame(self, idx):
        return self.pino_data.oMf[idx]

    def get_jacobian(self, idx, frame=pino.LOCAL_WORLD_ALIGNED):
        return pino.getFrameJacobian(self.pino_model, self.pino_data, idx, frame)

    @property
    def position_lower_limit(self):
        return self.pino_model.lowerPositionLimit

    @property
    def position_upper_limit(self):
        return self.pino_model.upperPositionLimit
