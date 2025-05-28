import open3d as o3d
import numpy as np
import os
import argparse
import csv
from scipy.spatial.transform import Rotation as R
import torch

import pybullet as p
import pybullet_data

from redsdf.redsdf_dataset_generator import generate_dataset
from redsdf.models.pointnet2.pointnet2_cls_msg import PointNet2_CLS_MSG
from examples.static_manifold.generate_data_from_pybullet import generate_point_cloud

parser = argparse.ArgumentParser()
parser.add_argument('--data-root', type=str, default=os.path.expanduser("~/Documents/Datasets/ShapeNetSem"), help="Root directory of ShapeNetSem dataset")
parser.add_argument('--save-dir', type=str, default=os.path.join(os.path.dirname(__file__), "../../data/shapenetsem_pybullet"), help="path to save generated data")
parser.add_argument('--categories', type=str, nargs='+', default=['laptop', 'mug', 'chair'], help="Category or categories of objects to process (e.g., 'mug', 'bottle', etc.)")
parser.add_argument('--num-points', type=int, default=10000, help="Number of points to sample from each mesh")
parser.add_argument('--visualize', action='store_true', help="Whether to visualize the point cloud and mesh after processing")
args = parser.parse_args()

metadata_file = os.path.join(args.data_root, "metadata.csv")
if not os.path.exists(metadata_file):
    raise FileNotFoundError(f"Metadata file not found at {metadata_file}")

# Read the metadata file
metadata = {}
# category_set = set()
category_dict = {}
with open(metadata_file, 'r') as f:
    reader = csv.reader(f, delimiter=',', quotechar='\"')
    for row in reader:
        keys = row
        break
    
    for row in reader:
        key = row[0][4:]
        metadata[key] = {}
        for k,v in zip(keys[1:],row[1:]):
            if k == 'up' or k == 'front' or k == 'aligned.dims':
                # Convert 'up' and 'front' to numpy arrays
                if v != '':
                    metadata[key][k] = np.array([float(x) for x in v.replace('\\','').split(',')])
                else:
                    if k == 'up':
                        metadata[key][k] = np.array([0.0, 0.0, 1.0])
                    elif k == 'front':
                        metadata[key][k] = np.array([1.0, 0.0, 0.0])
                    else:
                        print(f"Warning: 'aligned.dims' for {key} is empty, setting to default [1.0, 1.0, 1.0]")
                if k == 'aligned.dims':
                    metadata[key][k] = metadata[key][k] * 0.01
                    metadata[key][k][1], metadata[key][k][2] = metadata[key][k][2], metadata[key][k][1]  # Swap y and z dimensions
            elif k == 'unit':
                if v != '':
                    metadata[key][k] = float(v)
                else:
                    metadata[key][k] = 1.0
            elif k == 'category' or k == 'wnlemmas' or k == 'name' or k == 'tags':
                # Convert 'category' to a list of categories
                metadata[key][k] = v.split(',')
            else:
                metadata[key][k] = v
            
        for cat in metadata[key]['category']:
            cat = cat.lower()
            if cat not in category_dict:
                category_dict[cat] = []
            category_dict[cat].append(key)

print(f"Loaded metadata for {len(metadata)} objects")

if args.visualize:
    physicsClient = p.connect(p.GUI)
else:
    physicsClient = p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

model = PointNet2_CLS_MSG(40)
model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), "../../redsdf/models/pointnet2/best_model.pth"))['model_state_dict'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

for category in args.categories:  # args.category can be a single category or a list of categories
    keys = category_dict[category]
    print(f"Category: {category}, Number of objects: {len(keys)}")
    category_dir = os.path.join(args.save_dir, category)
    if not os.path.exists(category_dir):
        os.makedirs(category_dir)
    
    for key in keys:
        mesh_file = os.path.join(args.data_root, 'models-OBJ', 'models', key+".obj")
        if not os.path.exists(mesh_file):
            print(f"Mesh file not found for {key}, skipping...")
            continue
        object_dir = os.path.join(category_dir, key)
        if not os.path.exists(object_dir):
            os.makedirs(object_dir)
        print('key: ', key)
        print('name: ', metadata[key]['name'])
        print('tags: ', metadata[key]['tags'])
        print('category: ', metadata[key]['category'])
        print('wnsynset: ', metadata[key]['wnsynset'])
        print('wnlemmas: ', metadata[key]['wnlemmas'])
        
        front = metadata[key]['front']
        front = front / np.linalg.norm(front)
        up = metadata[key]['up']
        up = up / np.linalg.norm(up)
        left = np.cross(up, front) # Note: up × front, not front × up
        left = left / np.linalg.norm(left)
        rotation = np.array([
            [front[0], front[1], front[2]],
            [left[0], left[1], left[2]],
            [up[0], up[1], up[2]],
        ]) 
        
        mesh = o3d.io.read_triangle_mesh(mesh_file)
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()
        
        mesh = mesh.scale(metadata[key]['unit'], center=np.zeros(3))  # Scale mesh to the unit size
        mesh = mesh.rotate(rotation, center=np.zeros(3))  # Rotate mesh to align with the front and up vectors
        
        mesh_center = mesh.get_center()
        min_bound = mesh.get_min_bound()
        max_bound = mesh.get_max_bound()
        range_bound = (max_bound - min_bound)/2.0
        
        mesh = mesh.translate(mesh_center - min_bound - np.array([range_bound[0], range_bound[1], 0.0]), relative=False) # center x and y but keep z at 0.0

        # From Pybullet example:
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
        points, normals = generate_point_cloud(
                mesh_file=mesh_file,
                meshScale=[metadata[key]['unit'], metadata[key]['unit'], metadata[key]['unit']],
                basePosition = -min_bound - np.array([range_bound[0], range_bound[1], 0.0]),
                baseOrientation = R.from_matrix(rotation).as_quat(),
                N_points=args.num_points,
        )
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.normals = o3d.utility.Vector3dVector(normals)
        if args.visualize:
            # Create and visualize bounding box
            bbox = mesh.get_axis_aligned_bounding_box()
            bbox.color = (1, 0, 0)  # Red bounding box
            
            aligned_bbox = o3d.geometry.AxisAlignedBoundingBox(
                min_bound=mesh.get_min_bound(),
                max_bound=mesh.get_max_bound()# + metadata[key]['aligned.dims']
            )
            aligned_bbox.color = (0, 1, 0)  # Green aligned bounding box

            # Create coordinate frame
            coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.5, origin=np.zeros(3)
            )
            o3d.visualization.draw_geometries([pcd, mesh, aligned_bbox, coordinate_frame])
        
        dataset = generate_dataset(points, normals, **parameters)
        pose_idx = np.repeat(0, dataset.shape[0])[:, np.newaxis]
        dataset = np.concatenate([dataset, pose_idx], axis=1)
        joint_pos = np.zeros([1, 1])
        np.random.shuffle(dataset)
        
        np.save(os.path.join(object_dir, "0.npy"),
                dataset.astype(np.single))
        np.save(os.path.join(object_dir, 'poses.npy'), joint_pos.astype(np.single))
        
        with torch.no_grad():
            pcd_embedding = model(torch.concat([torch.Tensor(points), torch.Tensor(normals)], dim=1).T.to(device)[None, ...])[0].detach().cpu().numpy()
        np.save(os.path.join(object_dir, "pointnet2_embedding.npy"), pcd_embedding)
        
        o3d.io.write_triangle_mesh(os.path.join(object_dir, "scaled_mesh.obj"), mesh)
        o3d.io.write_point_cloud(os.path.join(object_dir, "pointcloud.ply"), pcd)

p.disconnect()