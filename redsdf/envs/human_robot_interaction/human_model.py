import time
import os
import trimesh
import pyrender
import numpy as np
import pybullet as pb
from pybullet_utils.bullet_client import BulletClient
from scipy.spatial.transform import Rotation
import redsdf


class HumanModel:
    def __init__(self, client=None, camera_pos=[0., 0., 0.], camera_rot=[0., 0., 0., 1.],
                 visualize_smpl=False, visualize_pcl=False):
        self.client = client
        self.camera_frame = np.eye(4)
        self.camera_frame[:3, :3] = Rotation.from_quat(np.array(camera_rot)).as_matrix()
        self.camera_frame[:3, 3] = np.array(camera_pos)
        self.visualize_smpl = visualize_smpl
        self.visualize_pcl = visualize_pcl

        # Keypoint Index 1, Keypoint Index 2, radius, length
        self.link_spec = [[0, 1, 0.13, 0.0],            # Right Hip
                          [0, 2, 0.13, 0.],             # Left Hip
                          [13, 14, 0.10, 0.1607],       # Chest
                          [0, 6, 0.15, 0.0],            # Spine 1
                          [3, 12, 0.16, 0.0],           # Spine 2
                          [1, 4, 0.09, 0.3845],         # Right Thigh
                          [2, 5, 0.09, 0.3845],         # Left Thigh
                          [4, 7, 0.07, 0.4005],         # Right Calf
                          [5, 8, 0.07, 0.4005],         # Left Carf
                          [31, 29, 0.04, 0.2348],       # Right Foot
                          [34, 33, 0.04, 0.2348],       # Left Foot
                          [13, 16, 0.08, 0],            # Right Shoulder
                          [14, 17, 0.08, 0],            # Left Shoulder
                          [16, 18, 0.05, 0.26137],      # Right Upper Arm
                          [17, 19, 0.05, 0.26137],      # Left Upper Arm
                          [18, 20, 0.04, 0.25247],      # Right Lower Arm
                          [19, 21, 0.04, 0.25247],      # Left Lower Arm
                          [20, 37, 0.04, 0.05],         # Right Hand
                          [21, 42, 0.04, 0.05],         # Left Hand
                          [12, 15, 0.06, 0.08297],      # Neck
                          [15, 45, 0.12, 0.]            # Head
                          ]

        self.start_index = 100
        human_record_dir = os.path.dirname(redsdf.package_dir) + "/object_models/hri_record"
        self.base_frame_traj, self.skeleton_traj, self.smpl_pose_traj = \
            self.construct_looping_trajectory(os.path.join(human_record_dir, "smpl_trajectory.npz"))

        if visualize_smpl:
            smpl_mesh_dir = os.path.join(human_record_dir, "smpl_mesh")
            print("Loading SMPL models from:", smpl_mesh_dir)
            file_list = sorted(os.listdir(smpl_mesh_dir))[self.start_index:self.start_index + 1000]
            self.smpl_meshes = [pyrender.Mesh.from_trimesh(trimesh.load_mesh(os.path.join(smpl_mesh_dir, mesh_file)))
                                for mesh_file in file_list]
            self.smpl_mesh_current = self.smpl_meshes[0]
        if visualize_pcl:
            pcd_dir = os.path.join(human_record_dir, "safe_blob")
            print("Loading Point Cloud from:", pcd_dir)
            file_list = sorted(os.listdir(pcd_dir))[self.start_index:self.start_index + 1000]
            self.pcls = [np.load(os.path.join(pcd_dir, pcd_file)) for pcd_file in file_list]
            self.pcl_current = self.pcls[0]

        self.loop_count = 0
        self.state = None

        self.default_pose = self.skeleton_traj[0]
        self.models = list()

        self.smpl_frame = np.eye(4)
        self.skeleton_position = self.skeleton_traj

        self.init_human_model()

    def construct_looping_trajectory(self, trajectory_file):
        trajectory = np.load(trajectory_file)

        base_frame = np.tile(np.eye(4), (trajectory['base_pos'].shape[0], 1, 1))
        # Convert the orientation from image frame (y down) to camera frame (y up)
        base_rot_mat = np.array([[1., 0., 0.], [0., -1, 0.], [0., 0., -1]]) @ Rotation.from_quat(
            trajectory['base_ori']).as_matrix()
        base_frame[:, :3, :3] = base_rot_mat
        base_frame[:, :3, 3] = trajectory['base_pos']

        # Extract only [3:66] for smple model
        pose = trajectory['pose'][:, 3:66]

        skeleton = trajectory['skeleton']
        return base_frame[self.start_index:], skeleton[self.start_index:], pose[self.start_index:]

    def init_human_model(self):
        for i, link_i in enumerate(self.link_spec):
            visual_shape = self.client.createVisualShape(self.client.GEOM_CAPSULE, radius=link_i[2], length=link_i[3],
                                                         rgbaColor=np.array([210., 161., 140., 255.]) / 255)
            collision_shape = self.client.createCollisionShape(self.client.GEOM_CAPSULE, radius=link_i[2],
                                                               height=link_i[3])
            pos, quat = self.compute_pos_and_quat(self.default_pose[link_i[0]], self.default_pose[link_i[1]])
            link_model = self.client.createMultiBody(1, collision_shape, visual_shape,
                                                     basePosition=pos,
                                                     baseOrientation=quat)

            self.client.setCollisionFilterGroupMask(link_model, -1, int('10000000', 2), int('01111111', 2))
            self.models.append(link_model)

    def reset(self):
        self.loop_count = 0
        self.update_recorded()
        return self.get_state()

    def update(self, skeleton_position):
        for i, link_i in enumerate(self.link_spec):
            pos, quat = self.compute_pos_and_quat(skeleton_position[link_i[0]], skeleton_position[link_i[1]])
            self.client.resetBasePositionAndOrientation(self.models[i], pos, quat)

    def update_recorded(self):
        self.smpl_frame = self.camera_frame @ self.base_frame_traj[self.loop_count]
        if self.visualize_smpl:
            self.smpl_mesh_current = self.smpl_meshes[self.loop_count]
        if self.visualize_pcl:
            self.pcl_current = self.pcls[self.loop_count]
        self.skeleton_position = self.skeleton_traj[self.loop_count] @ self.smpl_frame[:3, :3].T + \
                                 self.smpl_frame[:3, 3]
        self.update(self.skeleton_position)
        self.loop_count += 1
        if self.loop_count >= self.skeleton_traj.shape[0]:
            self.loop_count = 0

    def get_state(self):
        base_world_state = np.concatenate(
            [self.smpl_frame[:3, 3], Rotation.from_matrix(self.smpl_frame[:3, :3]).as_quat()])
        self.state = np.concatenate([base_world_state, self.smpl_pose_traj[self.loop_count],
                                     self.skeleton_position[[22, 23]].flatten()])
        return self.state

    @staticmethod
    def compute_pos_and_quat(pos_1, pos_2):
        pos = (pos_1 + pos_2) / 2
        vector = pos_2 - pos_1
        vector = vector / np.linalg.norm(vector)
        if np.isclose(vector[0], 1.0):
            sinb = 1
            cosb = 0
            sina = 0
            cosa = 1
        else:
            sinb = vector[0]
            cosb = np.sqrt(1 - sinb ** 2)
            sina = -vector[1] / cosb
            cosa = vector[2] / cosb
        rotation_matrix = np.array([[cosb, 0, sinb],
                                    [sina * sinb, cosa, -sina * cosb],
                                    [-cosa * sinb, sina, cosa * cosb]])
        quat = Rotation.from_matrix(rotation_matrix).as_quat()
        return pos, quat


def main():
    client = BulletClient(pb.GUI)
    import pybullet_data
    client.setAdditionalSearchPath(pybullet_data.getDataPath())
    client.loadURDF("plane.urdf")
    human = HumanModel(client, camera_pos=[-1.5, 0.7, 1.25],
                       camera_rot=Rotation.from_euler('xyz', (np.pi / 2, 0, -np.pi / 2)).as_quat())
    while True:
        human.update_recorded()
        human.get_state()
        time.sleep(0.03)


if __name__ == '__main__':
    main()
