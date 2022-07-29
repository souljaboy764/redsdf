import open3d as o3d
import numpy as np
import os
import argparse
from redsdf.redsdf_dataset_generator import generate_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--mesh_file', type=str, default="../../object_models/human.obj", help="path of mesh file")
parser.add_argument('--save_dir', type=str, default="./data_human", help="path to save generated data")
args = parser.parse_args()

mesh = o3d.io.read_triangle_mesh(args.mesh_file)
data_dir = args.data_dir
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
mesh.compute_vertex_normals()
sample_points = mesh.sample_points_uniformly(number_of_points=10000)
points = np.asarray(sample_points.points)
normals = np.asarray(sample_points.normals)
parameters = {'clean_aug_data': True,
              'aug_clean_thresh': 1e-3,
              'epsilons': [0.0025, 0.0050, 0.0100, 0.0200, 0.0300,
                           0.0500, 0.0800, 0.1200, 0.1600, 0.2000,
                           0.2500, 0.3000, 0.4000, 0.5000, 0.0010,
                           -.0025, -0.005, -0.010, -0.050, -0.1, -0.15],
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