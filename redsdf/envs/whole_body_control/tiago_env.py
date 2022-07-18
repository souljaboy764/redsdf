import time
import numpy as np
import pybullet as p
import pybullet_data
from pybullet_utils.bullet_client import BulletClient
from redsdf.envs.render import Render
from redsdf.envs.whole_body_control.tiago_model import TiagoModel


class TiagoEnv:
    def __init__(self, n_intermediate_steps=1, random_init=False, gui=None, control_mode='position'):
        if gui is None:
            self.client = BulletClient(p.DIRECT)
            self.render = None
        elif gui == 'default':
            self.client = BulletClient(p.GUI)
            self.render = None
            self.client.configureDebugVisualizer(flag=self.client.COV_ENABLE_GUI, enable=False)
            self.client.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=90., cameraPitch=-30.,
                                                   cameraTargetPosition=[0., 0., 0.5])
        elif gui == 'pyrender':
            self.client = BulletClient(p.DIRECT)
            self.render = Render(self.client)

        self.robot = TiagoModel(self.client, random_init=random_init, use_head=False, use_gripper=False,
                                control_mode=control_mode)

        self.client.setPhysicsEngineParameter()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.planeId = p.loadURDF("plane.urdf")
        self.n_intermediate_steps = n_intermediate_steps

        self.state = None

        self.targets = None
        self.target_markers = list()

        if self.render:
            self.render.init_pybullet_model(self.robot.model_id, link_mask=self.robot.render_link_mask)
            self.render.start()

        for i in range(2):
            vis_shape = self.client.createVisualShape(self.client.GEOM_SPHERE, radius=0.02, rgbaColor=[1., 0., 0., 1.])
            target_marker = self.client.createMultiBody(1, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vis_shape)
            self.target_markers.append(target_marker)
            if self.render:
                self.render.init_pybullet_model(target_marker)

    def sample_target(self):
        feasible = False
        while not feasible:
            targets = np.random.uniform([[0.1, -0.3, 0.4], [0.1, 0.0, 0.4]], [[0.6, 0.0, 1.2], [0.6, 0.3, 1.2]])
            targets[:, 2] = np.mean(targets[:, 2]) + np.random.uniform(-0.2, 0.2, (2,))

            if not np.any(np.logical_and(
                    np.logical_and(targets[:, 0] < 0.25,
                                   np.abs(targets[:, 1]) < 0.12), targets[:, 2] > 0.82)):
                self.set_targets(targets)
                feasible = True
        return self.targets

    def set_targets(self, targets):
        self.targets = targets
        for i, target in enumerate(targets):
            self.client.resetBasePositionAndOrientation(self.target_markers[i], target, [0., 0., 0., 1.])

    def reset(self, state=None):
        robot_state = self.robot.reset(state)
        self.state = robot_state
        return self.state

    def step(self, action):
        for _ in range(self.n_intermediate_steps):
            self.robot.apply_action(action)
            p.stepSimulation()
        robot_state = self.robot.get_state()
        self.state = robot_state
        return self.state

    def check_result(self):
        left_hand_pos = self.robot.kinematics.get_frame(72).translation
        right_hand_pos = self.robot.kinematics.get_frame(102).translation
        dist = np.linalg.norm(np.vstack([left_hand_pos, right_hand_pos]) - self.targets, axis=1)
        return dist

    def wrap_control_action(self, torso=None, l_arm=None, r_arm=None,
                            l_gripper=None, r_girpper=None, head=None):
        obs = self.robot.get_state()
        action = obs[7:28].copy()

        if torso is not None:
            action[0:1] = torso
        if l_arm is not None:
            action[1:8] = l_arm
        if l_gripper is not None:
            action[8:10] = l_gripper
        if r_arm is not None:
            action[10:17] = r_arm
        if r_girpper is not None:
            action[17:19] = r_girpper
        if head is not None:
            action[19:21] = head
        return action

    def close(self):
        if self.render:
            self.render.close()


class DebugVisualizer:
    def __init__(self, client_id, tiago_id, joint_indices, control_mode):
        self.client_id = client_id
        self.tiago_id = tiago_id
        self.joint_indices = joint_indices
        self.control_mode = control_mode
        self.num_joints = self.joint_indices.size

        self.debug_param_list = list()
        self.l_gripper_marker = None
        self.l_normal = None
        self.r_gripper_marker = None
        self.r_normal = None

        self.set_debug_slider(np.zeros(self.num_joints))
        self.debug_markers = list()

    def set_debug_slider(self, init_position):
        p.removeAllUserParameters()
        self.debug_param_list = list()
        for i, joint_id in enumerate(self.joint_indices):
            joint_info = p.getJointInfo(self.tiago_id, joint_id)
            id = p.addUserDebugParameter(joint_info[1].decode('utf-8'), joint_info[8], joint_info[9], init_position[i])
            self.debug_param_list.append(id)

    def add_debug_marker(self):
        vis_shape = p.createVisualShape(p.GEOM_SPHERE, radius=0.02, rgbaColor=[1., 0., 0., 1.])
        marker = p.createMultiBody(1, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vis_shape)
        normal = p.addUserDebugLine([0., 0., 0.], [0., 0., 1.], lineColorRGB=[1., 0., 0.], lineWidth=1)
        self.debug_markers.append([marker, normal])

    def get_debug_action(self, state):
        action = list()
        for idx in self.debug_param_list:
            action.append(p.readUserDebugParameter(idx))
        if self.control_mode == 'velocity':
            action = 20 * (action - state[:self.num_joints])
        elif self.control_mode == 'position':
            action = np.array(action)
        return action

    def update(self, point, jac):
        while point.shape[0] > len(self.debug_markers):
            self.add_debug_marker()

        for i, [marker_id, line_id] in enumerate(self.debug_markers):
            p.resetBasePositionAndOrientation(marker_id, point[i], [0., 0., 0., 1.])
            if np.linalg.norm(jac[i]) > 1e-6:
                l_jac = jac[i].flatten() / np.linalg.norm(jac[i])
                self.debug_markers[i][1] = p.addUserDebugLine(point[i], point[i] + l_jac * 0.3,
                                                              lineColorRGB=[1., 0., 0.], lineWidth=1,
                                                              replaceItemUniqueId=line_id)
