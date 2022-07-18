import torch
import argparse
import numpy as np
from redsdf.utils import visualization_pointcloud, plot2Dcontour, generate_pointcloud_by_raymarching

parser = argparse.ArgumentParser()
parser.add_argument('--model_file', type=str, default="checkpoint.pt", help="file of the. trained model")
parser.add_argument('--poses_file', type=str, default="../../object_models/smpl_meshes/poses_smpl.npy",
                    help="the pose file of human model")
parser.add_argument('--use_cuda', action="store_true", default=False, help="whether to use cuda")
args = parser.parse_args()

device = 'cuda:0' if args.use_cuda and torch.cuda.is_available() else 'cpu'
model = torch.load(args.model_file)
model.eval()
poses = np.load(args.poses_file)
value = 0.0
choice = np.random.choice(np.arange(poses.shape[0]), 5, replace=False)

for i in choice:
    points_model, _ = generate_pointcloud_by_raymarching(model, poses[i], device=device, value=value, distance=2.5,
                                                         img_size=[160, 160], distance_viewpoint=0.1,
                                                         target=[0., 0., 0.],
                                                         ypr_list=[[0, 0, 0], [90, 0, 0], [180, 0, 0], [270, 0, 0],
                                                                   [0, -90, 0], [0, 90, 0], [45, -45, 0], [135, -45, 0],
                                                                   [-45, -45, 0], [-135, -45, 0]])
    visualization_pointcloud([points_model])
