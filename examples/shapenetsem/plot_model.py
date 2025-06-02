import torch
import argparse
import numpy as np
import redsdf.utils as utils
import open3d as o3d
import os

from redsdf.models.pointnet2.pointnet2_cls_msg import PointNet2_CLS_MSG
from examples.static_manifold.generate_data_from_pybullet import generate_point_cloud


parser = argparse.ArgumentParser()
parser.add_argument('--model_file', type=str, default="checkpoint.pt", help="path of trained model")
parser.add_argument('--mesh_file', type=str, default="", help="path of ground truth mesh model") # This should be from the preprpcessed ShapeNetSem dataset
parser.add_argument('--pointcloud', type=str, default="", help="path of ground truth pointcloud") # This should be from the preprpcessed ShapeNetSem dataset
parser.add_argument('--num_points', type=int, default=10000, help="number of points to sample from the mesh")
parser.add_argument('--use_cuda', action="store_true", default=False, help="whether to use cuda")
args = parser.parse_args()

device = 'cuda:0' if args.use_cuda and torch.cuda.is_available() else 'cpu'
dis_model = torch.load(args.model_file)

model = PointNet2_CLS_MSG(40)
model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), "../../redsdf/models/pointnet2/best_model.pth"))['model_state_dict'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

gt_pcd = o3d.io.read_point_cloud(args.pointcloud)
gt_pcd.paint_uniform_color([1.0, 0.0, 0.0])  # Red color for ground truth points
gt_points = np.asarray(gt_pcd.points)
gt_normals = np.asarray(gt_pcd.normals)
if len(gt_points) == 0:
    raise ValueError("No points found in the point cloud. Please check the input file.")

with torch.no_grad():
    pcd_embedding = model(torch.concat([torch.Tensor(gt_points), torch.Tensor(gt_normals)], dim=1).T.to(device)[None, ...])[0].detach().cpu().numpy()

pcl, _ = utils.generate_pointcloud_by_raymarching(dis_model, pcd_embedding, device=device, value=0, distance=3, image_ratio=2e-4, max_marching_steps=1000,
                                                  img_size=[320, 320], distance_viewpoint=0.1, target=[0., 0., 0.],
                                                  ypr_list=[[0, 0, 0], [90, 0, 0], [180, 0, 0], [270, 0, 0],
                                                            [0, -90, 0], [0, 90, 0], [45, -45, 0], [135, -45, 0],
                                                            [-45, -45, 0], [-135, -45, 0], [45, 45, 0], [135, 45, 0],
                                                            [-45, 45, 0], [-135, 45, 0]],
                                                  epsilon=0.0005)

print(f"Generated point cloud with {len(gt_points)} points and GT point cloud with {len(rendered_points)} points.")

if args.mesh_file!="":
    mesh = o3d.io.read_triangle_mesh(args.mesh_file)
    mesh.compute_vertex_normals()
else:
    mesh = None
utils.create_vis_animation(pcl, mesh)
# utils.visualization_pointcloud(pcl)
# utils.plot2Dcontour(dis_model, pose_c, axis="z", v=0, device=device)
# utils.plot2Dcontour(dis_model, pose_c, axis="y", v=-0.25, device=device)