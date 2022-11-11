import numpy as np
import torch
import argparse
from redsdf.utils import visualization_pointcloud, generate_pointcloud_by_raymarching, plot2Dcontour


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', type=str, default="checkpoint.pt", help="path of trained model")
    parser.add_argument('--use_cuda', action="store_true", default=False, help="whether to use cuda")
    parser.add_argument('--random_config', action="store_true", default=False, help="whether to use cuda")
    args = parser.parse_args()
    joint_lower_limits = np.array([0.0, -1.1780972451, -1.1780972451, -0.785398163397,
                                   -0.392699081699, -2.09439510239, -1.41371669412, -2.09439510239])
    joint_upper_limits = np.array([0.35, 1.57079632679, 1.57079632679, 3.92699081699,
                                   2.35619449019, 2.09439510239, 1.41371669412, 2.09439510239])
    if args.random_config:
        joint_pose = np.random.uniform(joint_lower_limits, joint_upper_limits)
    else:
        joint_pose = np.zeros(8)
    joint_pose = np.array([0.1, 0.7, -0.5, 1.0, 1.5, 0., -1.1, 0.])
    device = 'cuda:0' if args.use_cuda and torch.cuda.is_available() else 'cpu'
    model = torch.load(args.model_file, map_location=device)
    model.eval()
    value = 0
    points_model, normals_model = generate_pointcloud_by_raymarching(model, joint_pose, device='cuda:0', value=value,
                                                      img_size=[160, 160], distance_viewpoint=0.1, target=[0., 0., 0.5],
                                                      ypr_list=[[0, 0, 0], [90, 0, 0], [180, 0, 0], [270, 0, 0],
                                                                [0, -90, 0], [0, 90, 0], [45, -45, 0], [135, -45, 0],
                                                                [-45, -45, 0], [-135, -45, 0]])
    create_vis_animation(points_model)
    visualization_pointcloud(points_model)
    plot2Dcontour(model, joint_pose, scope=[[-1., 1.], [-0.2, 1.5]], device='cuda', axis='x', step_length=0.01)


if __name__ == '__main__':
    main()
