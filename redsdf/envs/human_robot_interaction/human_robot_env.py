import time

import numpy as np
import pybullet as p
import pybullet_data
from scipy.spatial.transform import Rotation
from pybullet_utils.bullet_client import BulletClient
from redsdf.envs.whole_body_control import TiagoModel
from redsdf.envs.human_robot_interaction.human_model import HumanModel
from redsdf.envs.render import Render


class HumanRobotEnv:
    def __init__(self, n_intermediate_steps=4, random_init=False, gui=None, control_mode='position',
                 num_target_sequential=3, visualize_smpl=False, visualize_pcl=False):
        if gui is None:
            self.client = BulletClient(p.DIRECT)
            self.render = None
        elif gui == 'default':
            self.client = BulletClient(p.GUI)
            self.render = None
            self.client.configureDebugVisualizer(flag=self.client.COV_ENABLE_GUI, enable=False)
            self.client.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=60., cameraPitch=-45.,
                                                   cameraTargetPosition=[0.5, 0., 0.5])
        elif gui == 'pyrender':
            self.client = BulletClient(p.DIRECT)
            self.render = Render(self.client)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.planeId = p.loadURDF("plane.urdf")
        self.robot = TiagoModel(self.client, random_init=random_init, use_head=False, use_gripper=False,
                                control_mode=control_mode)
        self.human = HumanModel(self.client, camera_pos=[-1.5, 0.55, 1.25],
                                camera_rot=Rotation.from_euler('xyz', (np.pi / 2, 0, -np.pi / 2)).as_quat(),
                                visualize_smpl=visualize_smpl, visualize_pcl=visualize_pcl)

        table_pos = np.array([0.8, -0.15, 0.4])
        self.table_half_dim = [0.4, 0.4, 0.4]
        cube_vis = self.client.createVisualShape(self.client.GEOM_BOX,
                                                 halfExtents=self.table_half_dim, rgbaColor=[0.7, 0.7, 0.7, 1.0])
        cube_col = self.client.createCollisionShape(self.client.GEOM_BOX,
                                                    halfExtents=self.table_half_dim)
        self.table = self.client.createMultiBody(baseCollisionShapeIndex=cube_col, baseVisualShapeIndex=cube_vis,
                                                 basePosition=table_pos)
        self.table_pose = np.concatenate([table_pos, np.array([0., 0., 0., 1.])])

        self.client.setCollisionFilterGroupMask(self.table, -1, int('10000000', 2), int('01111110', 2))
        self.client.setCollisionFilterGroupMask(self.planeId, -1, int('10000000', 2), int('00000000', 2))

        self.n_intermediate_steps = n_intermediate_steps
        self.state = None
        self.step_counter = 0

        self.target_markers = list()
        vis_shape = self.client.createVisualShape(self.client.GEOM_SPHERE, radius=0.02, rgbaColor=[0., 1., 0., 1.])
        self.target_markers.append(self.client.createMultiBody(1, baseCollisionShapeIndex=-1,
                                                               baseVisualShapeIndex=vis_shape))
        self.target = None
        self.num_target_sequential = num_target_sequential

        if self.render:
            self.render.init_pybullet_model(self.robot.model_id, link_mask=self.robot.render_link_mask)
            self.render.init_pybullet_model(self.target_markers[0])
            self.render.init_pybullet_model(self.table)

            if self.human.visualize_smpl:
                quat = Rotation.from_matrix(self.human.smpl_frame[:3, :3]).as_quat()
                self.render.register_smpl_model("smpl_human", self.human.smpl_mesh_current,
                                                self.human.smpl_frame[:3, 3], quat)
            else:
                for human_model_id in self.human.models:
                    self.render.init_pybullet_model(human_model_id)
            if self.human.visualize_pcl:
                quat = Rotation.from_matrix(self.human.smpl_frame[:3, :3]).as_quat()
                self.render.register_pcl_model("pcl_human", self.human.pcl_current,
                                               self.human.smpl_frame[:3, 3], quat)

            self.render.start()

        self.reset()

    def sample_target(self):
        targets = np.random.uniform([[0.4, 0.0, 0.9],
                                     [0.5, -0.1, 1.1],
                                     [0.6, -0.4, 0.85]] * self.num_target_sequential,
                                    [[0.7, 0.3, 1.0],
                                     [0.8, 0.1, 1.3],
                                     [0.9, -0.2, 1.0]] * self.num_target_sequential)
        return targets

    def set_target(self, target):
        self.target = target
        for i, target in enumerate(target):
            self.client.resetBasePositionAndOrientation(self.target_markers[i], target, [0., 0., 0., 1.])

    def reset(self, state=None):
        self.step_counter = 0
        if state is not None:
            self.robot.reset(state[:self.robot.kinematics.num_joints])
        else:
            init_state = np.array([0.3,
                                   0., np.pi / 2, np.pi / 2, np.pi / 2., 0., -np.pi / 2., 0., 0.0225, 0.0225,
                                   0., 0, np.pi / 2, np.pi / 2., np.pi / 2., 0., 0., 0.0225, 0.0225,
                                   0., 0.])
            self.robot.reset(init_state)

        self.human.reset()
        self.update_visualization()
        self.state = self.get_observation()
        return self.state

    def step(self, action):
        if self.step_counter % 2 == 0:
            self.human.update_recorded()
            self.update_visualization()

        for k in range(self.n_intermediate_steps):
            self.robot.apply_action(action)
            p.stepSimulation()

        self.state = self.get_observation()

        self.step_counter += 1
        return self.state

    def get_observation(self):
        robot_state = self.robot.get_state()
        human_state = self.human.get_state()
        return np.concatenate([robot_state, human_state, self.table_pose])

    def close(self):
        self.render.close()

    def check_result(self):
        right_hand_pos = self.robot.kinematics.get_frame(102).translation
        dist = np.linalg.norm(np.vstack([right_hand_pos]) - self.target, axis=1)
        return dist

    def update_visualization(self):
        if self.human.visualize_smpl:
            quat = Rotation.from_matrix(self.human.smpl_frame[:3, :3]).as_quat()
            self.render.update_smpl_model("smpl_human", self.human.smpl_mesh_current, self.human.smpl_frame[:3, 3],
                                          quat)
        if self.human.visualize_pcl:
            quat = Rotation.from_matrix(self.human.smpl_frame[:3, :3]).as_quat()
            self.render.update_pcl_model("pcl_human", self.human.pcl_current, self.human.smpl_frame[:3, 3],
                                         quat)


def main():
    env = HumanRobotEnv(gui='pyrender', visualize_smpl=True, visualize_pcl=True)
    state = env.reset()
    while True:
        action = state[7:28]
        env.step(action)
        time.sleep(1 / 60.)


if __name__ == '__main__':
    main()
