import torch
import argparse
import numpy as np
import redsdf.utils as utils
import open3d as o3d

parser = argparse.ArgumentParser()
parser.add_argument('--model_file', type=str, default="checkpoint.pt", help="path of trained model")
parser.add_argument('--mesh_file', type=str, default="", help="path of ground truth mesh model")
parser.add_argument('--use_cuda', action="store_true", default=False, help="whether to use cuda")
args = parser.parse_args()

device = 'cuda:0' if args.use_cuda and torch.cuda.is_available() else 'cpu'
dis_model = torch.load(args.model_file)
pose_c = np.zeros(1)
pcl, _ = utils.generate_pointcloud_by_raymarching(dis_model, pose_c, device=device, value=0, distance=3, image_ratio=2e-4,
                                                  img_size=[320, 320], distance_viewpoint=0.1, target=[0., 0., 0.],
                                                  ypr_list=[[0, 0, 0], [90, 0, 0], [180, 0, 0], [270, 0, 0],
                                                            [0, -90, 0], [0, 90, 0], [45, -45, 0], [135, -45, 0],
                                                            [-45, -45, 0], [-135, -45, 0], [45, 45, 0], [135, 45, 0],
                                                            [-45, 45, 0], [-135, 45, 0]],
                                                  epsilon=0.0005)

if args.mesh_file!="":
    mesh = o3d.io.read_triangle_mesh(args.mesh_file)
    mesh.compute_vertex_normals()
else:
    mesh = None
utils.create_vis_animation(pcl, mesh)
# utils.visualization_pointcloud(pcl)
# utils.plot2Dcontour(dis_model, pose_c, axis="z", v=0, device=device)
# utils.plot2Dcontour(dis_model, pose_c, axis="y", v=-0.25, device=device)