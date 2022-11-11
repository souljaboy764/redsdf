import os
import time
import argparse
import numpy as np
import open3d as o3d
import pybullet as p
from tqdm import tqdm
from multiprocessing import Pool, Value
import redsdf
from redsdf.redsdf_dataset_generator import generate_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--urdf_dir', type=str, default=os.path.dirname(redsdf.package_dir) + "/object_models/tiago_urdf",
                    help="Path of mesh file")
parser.add_argument('--save_dir', type=str, default="./data", help="Path to save generated data")
parser.add_argument('--n_poses', type=int, default=10000, help="Number of poses")
parser.add_argument('--right_arm', action="store_true", default=False, help="Whether to train right arm or left")
parser.add_argument('--debug', action="store_true", default=False, help="Debug the generation process")
parser.add_argument('--append_data', action="store_true", default=False, help="Append to the previous generated data")
args = parser.parse_args()


class TiagoPointCloudGenerator:
    def __init__(self, urdf_file, debug=True):
        if debug:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        self.tiago = p.loadURDF(urdf_file, useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION)

        self.joint_indices = [2, 5, 6, 7, 8, 9, 10, 11]
        self.gripper_indices = [17, 18]
        self.n_joint = len(self.joint_indices)
        self.joint_limits = np.zeros((self.n_joint, 2))
        for i, joint_id in enumerate(self.joint_indices):
            joint_info = p.getJointInfo(self.tiago, joint_id)
            self.joint_limits[i, 0] = joint_info[8]
            self.joint_limits[i, 1] = joint_info[9]

        self.set_collision_mask()
        self.camara_parameters = {'img_width': 256,
                                  'img_height': 256,
                                  'fov': 90,  # degree
                                  'aspect': 1.0,  # ratio
                                  'farVal': 6.0,
                                  'nearVal': 0.1,
                                  'up_axis': 2,
                                  'distance': 1.5,
                                  'target_position': [0.0, 0.0, 0.7]
                                  }
        self.ypr_list = np.array([[0, 0, 0], [90, 0, 0], [180, 0, 0], [-90, 0, 0], [0, 90, 0], [0, -90, 0],
                                  [45, 45, 0], [135, 45, 0], [-45, 45, 0], [-135, 45, 0],
                                  [45, -45, 0], [135, -45, 0], [-45, -45, 0], [-135, -45, 0]])

        self.reset()

    def reset(self):
        p.resetJointStatesMultiDof(self.tiago, self.gripper_indices, [[0.01], [0.01]])

    def set_collision_mask(self, ):
        # set all links to a default value: Group: 0001, Mask: 1100
        for i in range(p.getNumJoints(self.tiago)):
            p.setCollisionFilterGroupMask(self.tiago, i, collisionFilterGroup=1, collisionFilterMask=12)

        # set upper arm links to: Group: 0010, Mask: 1000
        upper_arm_index = [5, 6, 7]
        for idx in upper_arm_index:
            p.setCollisionFilterGroupMask(self.tiago, idx, collisionFilterGroup=2, collisionFilterMask=8)

        # set lower arm links to: Group: 0100, Mask: 0001
        lower_arm_index = [8, 9, 10, 11]
        for idx in lower_arm_index:
            p.setCollisionFilterGroupMask(self.tiago, idx, collisionFilterGroup=4, collisionFilterMask=1)

        # set all gripper links to : Group: 1000, Mask: 0011
        gripper_link = [12, 13, 14, 15, 16, 17, 18]
        for idx in gripper_link:
            p.setCollisionFilterGroupMask(self.tiago, idx, collisionFilterGroup=8, collisionFilterMask=3)

    def start_debugger(self):
        joint_pos_debug = list()
        for i in range(self.n_joint):
            joint_pos_debug.append(
                p.addUserDebugParameter(f"joint_{i}", self.joint_limits[i, 0],
                                        self.joint_limits[i, 1], self.joint_limits[i, 0]))
        while True:
            joint_pos = np.zeros((self.n_joint, 1))
            for i in range(self.n_joint):
                joint_pos[i, 0] = p.readUserDebugParameter(joint_pos_debug[i])
            p.resetJointStatesMultiDof(self.tiago, self.joint_indices, joint_pos.tolist())

            # Set Joint Point
            p.resetJointStatesMultiDof(self.tiago, self.gripper_indices, [[0.01], [0.01]])

            p.performCollisionDetection()
            if len(p.getContactPoints()) > 0:
                print("collision!")
            time.sleep(0.01)

    def sample_configuration(self):
        self_collide = True
        while self_collide:
            joint_pos = np.random.uniform(self.joint_limits[:, 0], self.joint_limits[:, 1])
            # Set Joint Point
            p.resetJointStatesMultiDof(self.tiago, self.joint_indices, targetValues=joint_pos[:, np.newaxis])
            p.performCollisionDetection()
            if len(p.getContactPoints()) == 0:
                return joint_pos

    def generate_point_cloud(self, visualize=False, N_points=8000):
        points = []
        for ypr in self.ypr_list:
            view_matrix = p.computeViewMatrixFromYawPitchRoll(self.camara_parameters['target_position'],
                                                              self.camara_parameters['distance'],
                                                              *ypr,
                                                              self.camara_parameters['up_axis'])
            projection_matrix = p.computeProjectionMatrixFOV(fov=self.camara_parameters['fov'],
                                                             aspect=self.camara_parameters['aspect'],
                                                             nearVal=self.camara_parameters['nearVal'],
                                                             farVal=self.camara_parameters['farVal'])
            image = p.getCameraImage(self.camara_parameters['img_width'],
                                     self.camara_parameters['img_height'],
                                     view_matrix, projection_matrix)
            depth_img = image[3]
            points_i = self.generate_single_view_point_cloud(depth_img, projection_matrix, view_matrix)
            points.append(points_i)

        points = np.vstack(points)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        pcd.normals = o3d.utility.Vector3dVector(points[:, 3:])
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(0.05, 50))
        # pcd = pcd.voxel_down_sample(0.01)
        pcd = pcd.uniform_down_sample(int(np.floor(len(pcd.points) / N_points)))
        if visualize:
            o3d.visualization.draw_geometries([pcd], point_show_normal=True)
        return np.asarray(pcd.points), np.asarray(pcd.normals)

    @staticmethod
    def generate_single_view_point_cloud(depth_img, projection_matrix, view_matrix):
        width, height = depth_img.shape
        projection_matrix = np.array(projection_matrix).reshape((4, 4), order='F')
        view_matrix = np.array(view_matrix).reshape((4, 4), order='F')
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        x = (2 * x - width) / width
        y = -(2 * y - height) / height
        depth_img = depth_img * 2 - 1.
        points = np.stack([x, y, depth_img, np.ones_like(depth_img)], axis=2).reshape((-1, 4))
        point_eye = (np.linalg.inv(projection_matrix) @ points.T).T
        point_eye /= point_eye[:, 3:]
        point_eye = point_eye[np.where(-point_eye[:, 2] < 5)]
        point_world = (np.linalg.inv(view_matrix) @ point_eye.T).T
        point_dir = (np.linalg.inv(view_matrix)[:3, :3] @ point_eye[:, :3].T).T
        return np.concatenate([point_world[:, :3], -point_dir], axis=1)


def main():
    use_left_arm = not args.right_arm
    debug_collision = False
    append_data = args.append_data
    output_dir = args.save_dir
    num_configuration = args.n_poses
    batch_size = 1000

    batch_size = np.minimum(batch_size, num_configuration)

    if use_left_arm:
        urdf_file = os.path.join(args.urdf_dir, "tiago_left_arm_simplified_with_screen.urdf")
    else:
        urdf_file = os.path.join(args.urdf_dir, "tiago_right_arm_simplified_with_screen.urdf")
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    tiago_pcd_generator = TiagoPointCloudGenerator(urdf_file, debug=args.debug)

    # Debug Parameters of Joint Position
    if debug_collision:
        tiago_pcd_generator.start_debugger()
    else:
        if append_data:
            batch_poses = np.load(output_dir + '/poses.npy').tolist()
            count_configuration = len(batch_poses)
        else:
            count_configuration = 0
            batch_poses = list()
        batch_dataset = list()

        pbar = tqdm(total=num_configuration, initial=count_configuration)
        while count_configuration < num_configuration:
            joint_pos = tiago_pcd_generator.sample_configuration()
            points, normals = tiago_pcd_generator.generate_point_cloud(visualize=False, N_points=10000)
            parameters = {'clean_aug_data': True,
                          'aug_clean_thresh': 0.1,
                          'epsilons': [0.0025, 0.0050, 0.0100, 0.0200, 0.0300,
                                       0.0500, 0.0800, 0.1200, 0.1600, 0.2000,
                                       0.2500, 0.3000, 0.4000, 0.5000,
                                       -.0025, -0.005, -0.010, -0.050, -0.1, -0.15],
                          'down_sampling': 0.2,
                          'is_delete_outliers': True,
                          'is_using_point_dependent_weight': True,
                          'verbose': False}
            dataset = generate_dataset(points, normals, **parameters)
            pose_idx = np.repeat(count_configuration, dataset.shape[0])[:, np.newaxis]
            dataset = np.concatenate([dataset, pose_idx], axis=1)
            batch_dataset.append(dataset)
            batch_poses.append(joint_pos)
            if len(batch_dataset) >= batch_size:
                batch_dataset = np.vstack(batch_dataset)
                np.random.shuffle(batch_dataset)

                file_name = f'{count_configuration - batch_size + 1}-{count_configuration}.npy'
                print("File: {}, num configurations: {}, min id: {}, max id: {}"
                      .format(file_name,
                              np.unique(batch_dataset[:, -1]).shape[0],
                              np.min(batch_dataset[:, -1]).astype(int),
                              np.max(batch_dataset[:, -1]).astype(int)))
                assert np.unique(batch_dataset[:, -1]).shape[0] == batch_size and \
                       np.max(batch_dataset[:, -1]) == count_configuration and \
                       np.min(batch_dataset[:, -1]) == count_configuration - batch_size + 1

                np.save(output_dir + '/' + file_name,
                        batch_dataset.astype(np.single))
                np.save(output_dir + '/poses.npy', np.vstack(batch_poses).astype(np.single))
                batch_dataset = list()
            count_configuration += 1
            pbar.update(1)
        print("Total Configuration: ", count_configuration)


if __name__ == '__main__':
    main()
