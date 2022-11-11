import torch
import argparse
import numpy as np
from redsdf.utils import visualization_pointcloud, plot2Dcontour, generate_pointcloud_by_raymarching, \
    create_vis_animation

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
    pose = np.array([-0.0605772, 0.22628513, 0.4152068, - 0.46194694, - 0.1678464, - 0.4154218,
                     0.1644592, - 0.03399609, - 0.07086531, 0.6445115, 0.08262138, - 0.15467109,
                     0.74479917, - 0.27376864, 0.11674712, 0.18656241, - 0.05839198, 0.1019961,
                     - 0.5834142, 0.14998358, 0.04388938, - 0.22975257, - 0.07154395, 0.06287636,
                     - 0.10618248, - 0.00177596, - 0.00848303, 0., 0., 0.,
                     0., 0., 0., 0.23948007, - 0.27514211, 0.05680601,
                     0.01632375, - 0.15436637, - 0.12837332, - 0.03252496, 0.14299533, 0.22147159,
                     0.07584253, - 0.43253279, 0.04827495, - 0.04886943, - 1.102587, -0.92444523,
                     - 0.26495959, 0.36815303, 0.62478912, 0.09861467, - 1.62385885, 0.44793796,
                     - 0.32003259, 1.08721426, - 0.72450927, - 0.19913278, - 0.39483011, -0.21942132,
                     - 0.20835243, 0.12807261, 0.12169324])
    points_model, _ = generate_pointcloud_by_raymarching(model, pose, device=device, value=value, distance=2.5,
                                                         img_size=[160, 160], distance_viewpoint=0.1,
                                                         target=[0., 0., 0.],
                                                         ypr_list=[[0, 0, 0], [90, 0, 0], [180, 0, 0], [270, 0, 0],
                                                                   [0, -90, 0], [0, 90, 0], [45, -45, 0], [135, -45, 0],
                                                                   [-45, -45, 0], [-135, -45, 0]])
    create_vis_animation(points_model)
    # visualization_pointcloud([points_model])
