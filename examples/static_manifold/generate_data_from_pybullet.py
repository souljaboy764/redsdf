import argparse
import pybullet as p
import pybullet_data
import numpy as np
from redsdf.redsdf_dataset_generator import generate_dataset
import open3d as o3d
import os

parser = argparse.ArgumentParser()
parser.add_argument('--mesh_file', type=str, default="../../object_models/sofa.obj", help="path of mesh file")
parser.add_argument('--save_dir', type=str, default="./data_sofa", help="path to save generated data")
args = parser.parse_args()


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

def generate_point_cloud(N_points=10000):
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    mesh_file = args.mesh_file
    assert mesh_file.endswith(".obj") or mesh_file.endswith(".urdf")
    if mesh_file.endswith(".obj"):
        visualShapeId = p.createVisualShape(
           shapeType=p.GEOM_MESH,
           fileName=mesh_file,
           rgbaColor=None,
           meshScale=[1, 1, 1]
        )
        collisionShapeId = p.createCollisionShape(
           shapeType=p.GEOM_MESH,
           fileName=mesh_file,
           meshScale=[1, 1, 1]
        )
        object= p.createMultiBody(
           baseMass=1.0,
           baseCollisionShapeIndex=collisionShapeId,
           baseVisualShapeIndex=visualShapeId,
           basePosition=[0, 0, 0],
           baseOrientation=p.getQuaternionFromEuler([0, 0, 0])
        )
    elif mesh_file.endswith(".urdf"):
        object = p.loadURDF(mesh_file)

    camara_parameters = {'img_width': 256,
                         'img_height': 256,
                         'fov': 90,  # degree
                         'aspect': 1.0,  # ratio
                         'farVal': 6.0,
                         'nearVal': 0.1,
                         'up_axis': 2,
                          'distance': 1.2,
                          'target_position': [0.0, 0.0, 0.0]
                          }
    ypr_list = np.array([[0, 0, 0], [90, 0, 0], [180, 0, 0], [-90, 0, 0], [0, 90, 0], [0, -90, 0],
                        [45, 45, 0], [135, 45, 0], [-45, 45, 0], [-135, 45, 0],
                        [45, -45, 0], [135, -45, 0], [-45, -45, 0], [-135, -45, 0]])
    points = []
    for ypr in ypr_list:
        view_matrix = p.computeViewMatrixFromYawPitchRoll(camara_parameters['target_position'],
                                                          camara_parameters['distance'],
                                                          *ypr,
                                                          camara_parameters['up_axis'])
        projection_matrix = p.computeProjectionMatrixFOV(fov=camara_parameters['fov'],
                                                         aspect=camara_parameters['aspect'],
                                                         nearVal=camara_parameters['nearVal'],
                                                         farVal=camara_parameters['farVal'])
        image = p.getCameraImage(camara_parameters['img_width'],
                                 camara_parameters['img_height'],
                                 view_matrix, projection_matrix)
        depth_img = image[3]
        points_i = generate_single_view_point_cloud(depth_img, projection_matrix, view_matrix)
        points.append(points_i)
    points = np.vstack(points)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    pcd.normals = o3d.utility.Vector3dVector(points[:, 3:])
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(0.05, 50))
    if len(pcd.points) > N_points:
        pcd = pcd.uniform_down_sample(int(np.floor(len(pcd.points) / N_points)))
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=15, std_ratio=5)
    o3d.visualization.draw_geometries([cl], point_show_normal=False)
    return np.asarray(cl.points), np.asarray(cl.normals)

def main():
    data_dir = args.save_dir
    if not os.path.exists(data_dir):
       os.makedirs(data_dir)
    points, normals = generate_point_cloud(N_points=10000)

    parameters = {'clean_aug_data': True,
                  'aug_clean_thresh': 0.1,
                  'epsilons': [0.0025, 0.0050, 0.0100, 0.0200, 0.0300,
                               0.0500, 0.0800, 0.1200, 0.1600, 0.2000,
                               0.2500, 0.3000, 0.4000, 0.5000, 0.0010,
                               -.0025, -0.005, -0.010, -0.050, -0.1, -0.15],
                  'down_sampling': 1.0,
                  'outliers_in_augmented': -0.02,
                  'is_delete_outliers': True,
                  'is_using_point_dependent_weight': True,
                  'verbose': False}
    dataset = generate_dataset(points, normals, **parameters)
    pose_idx = np.repeat(0, dataset.shape[0])[:, np.newaxis]
    dataset = np.concatenate([dataset, pose_idx], axis=1)
    joint_pos = np.zeros([1, 1])
    np.random.shuffle(dataset)
    np.save(os.path.join(data_dir, "0.npy"),
            dataset.astype(np.single))
    np.save(os.path.join(data_dir, 'poses.npy'), joint_pos.astype(np.single))

if __name__ == "__main__":
    main()
