import time
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
import torch
import open3d as o3d
from tqdm import tqdm
import pybullet as p


def generate_pointcloud_from_model(model, pose, verbose=True, return_normals=False, **kwargs):
    scopes = kwargs.get('scopes', 'robot')
    value = kwargs.get('value', 0)
    device = kwargs.get('device', 'cpu')
    step_length = kwargs.get('step_length', 0.01)
    batch_size = kwargs.get('batch_size', 100000)
    threshold = kwargs.get('threshold', 0.001)
    model.to(device)
    if scopes == 'robot':
        scopes = [[-0.9, 0.4], [-0.4, 0.8], [-0.1, 2.0]]
    elif scopes == 'human':
        scopes = [[-1, 1], [-1, 1], [-1, 1]]
    xgrid = np.arange(scopes[0][0], scopes[0][1], step_length).astype('float32')
    ygrid = np.arange(scopes[1][0], scopes[1][1], step_length).astype('float32')
    zgrid = np.arange(scopes[2][0], scopes[2][1], step_length).astype('float32')
    xx, yy, zz = np.meshgrid(xgrid, ygrid, zgrid)
    data = torch.from_numpy(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).transpose([1, 0]))
    if verbose:
        print("There are in total {} points".format(data.shape[0]))
    dataloader = DataLoader(data, batch_size=batch_size)
    result = list()
    for loaded_data in tqdm(dataloader, disable=not verbose):
        loaded_data = loaded_data.float().to(device)
        pose_t = torch.tensor(pose.reshape([1, -1]).repeat(repeats=loaded_data.shape[0], axis=0)).float().to(device)
        out = model.y_torch(loaded_data, pose_t).reshape(-1)
        show_idx = torch.where((out - value).abs() < threshold)[0]
        result.append(loaded_data[show_idx])
    result_t = torch.vstack(result)
    result = result_t.to('cpu').detach().numpy()
    if not return_normals:
        return result
    else:
        dataloader = DataLoader(result_t, batch_size=batch_size)
        all_normal = list()
        for loaded_data in tqdm(dataloader, disable=not verbose):
            pose_t = torch.tensor(pose.reshape([1, -1]).repeat(repeats=loaded_data.shape[0], axis=0)).float().to(device)
            normal = model.J(loaded_data, pose_t).reshape([-1, 3])
            normal = normal / np.linalg.norm(normal, axis=1).reshape([-1, 1])
            all_normal.append(normal)
        return result, np.concatenate(all_normal, axis=0)


def visualization_pointcloud(pointclouds, transforms=None, colors=None, normals=None, file=None, library='o3d'):
    if library == 'o3d':
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(visible=True)
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0., 0., 0.])
        vis.add_geometry(origin)
        vis.update_geometry(origin)
        if not isinstance(pointclouds, list):
            pointclouds = [pointclouds]
            if normals is not None:
                normals = [normals]
        for i, pointcloud in enumerate(pointclouds):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pointcloud)
            if not transforms is None:
                pcd.transform(transforms[i])
            if not colors is None:
                pcd.paint_uniform_color(colors[i])
            if not normals is None:
                pcd.normals = o3d.utility.Vector3dVector(normals[i])

            vis.add_geometry(pcd)
            vis.update_geometry(pcd)
        vis.get_render_option().point_show_normal = True
        vis.poll_events()
        vis.update_renderer()
        vis.run()
        if file:
            vis.capture_screen_image(file)
    elif library == 'plt':
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        for pointcloud in pointclouds:
            ax.scatter(pointcloud[:, 0], pointcloud[:, 1], pointcloud[:, 2], marker='.')
        plt.show()
        if file:
            fig.savefig(file)
    else:
        print('no library. please use o3d or plt.')


def plot2Dcontour(model, pose, file=None, scope=[[-2, 2], [-2, 2]], v=0, axis="z",
                  levels=20, device='cpu', step_length=0.005, batch_size=100000, verbose=False):
    model.to(device)
    xgrid = np.arange(scope[0][0], scope[0][1], step_length)
    ygrid = np.arange(scope[1][0], scope[1][1], step_length)
    xx, yy = np.meshgrid(xgrid, ygrid)
    zz = np.ones(xx.shape) * v
    if axis == "x":
        data = np.c_[zz.ravel(), xx.ravel(), yy.ravel()]
    elif axis == "y":
        data = np.c_[xx.ravel(), zz.ravel(), yy.ravel()]
    else:
        data = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
    dataloader = DataLoader(data, batch_size=batch_size)
    result = list()
    for loaded_data in tqdm(dataloader, disable=not verbose):
        pose_t = torch.tensor(pose.reshape([1, -1]).repeat(repeats=loaded_data.shape[0], axis=0)).float().to(device)
        out = model.y_torch(loaded_data.float().to(device), pose_t).to('cpu').detach().numpy()
        result.append(out)
    result = np.concatenate(result, axis=0).reshape(xx.shape)
    fig = plt.figure(figsize=(6, 9))
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    cs = plt.contour(xx, yy, result, levels=levels)
    plt.clabel(cs, inline=1, fontsize=8)
    plt.xticks([])
    plt.yticks([])
    plt.show()
    if file:
        fig.savefig(file)


def plot2Dcontourf(model, pose, file=None, scope=[[-2, 2], [-2, 2]], v=0, axis="z",
                   levels=20, device='cpu', step_length=0.005, batch_size=100000, verbose=False):
    model.to(device)
    xgrid = np.arange(scope[0][0], scope[0][1], step_length)
    ygrid = np.arange(scope[1][0], scope[1][1], step_length)
    xx, yy = np.meshgrid(xgrid, ygrid)
    zz = np.ones(xx.shape) * v
    if axis == "x":
        data = np.c_[zz.ravel(), xx.ravel(), yy.ravel()]
    elif axis == "y":
        data = np.c_[xx.ravel(), zz.ravel(), yy.ravel()]
    else:
        data = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
    dataloader = DataLoader(data, batch_size=batch_size)
    result = list()
    for loaded_data in tqdm(dataloader, disable=not verbose):
        pose_t = torch.tensor(pose.reshape([1, -1]).repeat(repeats=loaded_data.shape[0], axis=0)).float().to(device)
        out = model.y_torch(loaded_data.float().to(device), pose_t).to('cpu').detach().numpy()
        result.append(out)
    result = np.concatenate(result, axis=0).reshape(xx.shape)
    fig = plt.figure()
    cs = plt.contourf(xx, yy, result, levels=levels, cmap=plt.cm.rainbow_r, vmax=result.max(), vmin=result.min())
    plt.colorbar()
    plt.xlabel("x_1")
    plt.ylabel("x_2")
    plt.title("Value of implicit function")
    plt.show()
    if file:
        fig.savefig(file)


def plotcolormap(model, pose, data, file=None, cmap=plt.cm.plasma, scope=[-0.01, 0.01], device='cpu'):
    model.to(device)
    inp = torch.from_numpy(data).float().to(device)
    pose_t = torch.tensor(pose.reshape([1, -1]).repeat(repeats=data.shape[0], axis=0)).float().to(device)
    out = model.y_torch(inp, pose_t).reshape(-1).to('cpu').detach().numpy()
    print("The std of model is: {}".format(np.sqrt((out ** 2).sum() / data.shape[0])))
    colors = ((out - (scope[0] + scope[1]) / 2) / ((scope[1] - scope[0]) / 2) + 1) * (255 / 2)
    colors = cmap(colors.astype("int64"))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0., 0., 0.])
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(visible=True)
    vis.add_geometry(pcd)
    vis.add_geometry(origin)
    vis.update_geometry(pcd)
    vis.update_geometry(origin)
    vis.poll_events()
    vis.update_renderer()
    vis.run()
    if file:
        vis.capture_screen_image(file)


def generate_pointcloud_by_raymarching(model, pose, **kwargs):
    img_size = kwargs.get('img_size', [160, 120])
    distance_viewpoint = kwargs.get('distance_viewpoint', 0.1)
    furthest_distance = kwargs.get('furthest_distance', 4.0)
    image_ratio = kwargs.get('image_ratio', 8e-4)
    max_marching_steps = kwargs.get('max_marching_steps', 30)
    device = kwargs.get('device', 'cpu')
    epsilon = kwargs.get('epsilon', 5e-3)
    target = kwargs.get('target', [0, 0, 0])
    distance = kwargs.get('distance', 2)
    value = kwargs.get('value', 0.0)
    ypr_list = kwargs.get('ypr_list', [[0, 0, 0], [0, 90, 0], [0, 180, 0],
                                       [0, -90, 0], [0, 0, 90], [0, 0, 180],
                                       [0, 0, -90]])
    all_data = list()

    all_normals = list()
    for ypr in ypr_list:
        output_data, normal = raymarching_one_point(model, pose, img_size, distance_viewpoint, furthest_distance,
                                                    image_ratio, target, max_marching_steps, distance, ypr,
                                                    epsilon, value, device)
        all_data.append(output_data)
        all_normals.append(normal)
    return np.concatenate(all_data, axis=0), np.concatenate(all_normals, axis=0)


def raymarching_one_point(model, pose,
                          img_size=[160, 120],
                          distance_viewpoint=0.1,
                          furthest_distance=4.0,
                          image_ratio=8e-4,
                          target=[0, 0, 0],
                          max_marching_steps=30,
                          distance=2,
                          ypr_camera=[0, 0, 0],
                          epsilon=0.005,
                          value=0,
                          device='cpu'):
    img_height = img_size[0]
    img_width = img_size[1]
    pic_matrix = np.zeros([img_height, img_width, 3])
    idxs = np.array(np.meshgrid(np.arange(img_height), np.arange(img_width))).transpose([2, 1, 0])
    view_matrix = p.computeViewMatrixFromYawPitchRoll(target, distance, *ypr_camera, 2)
    view_matrix = np.array(view_matrix).reshape(4, 4).transpose(-1, -2)
    eye_pose = np.linalg.inv(view_matrix)[:3, 3]  # target + distance * view_matrix[:3, 2]
    pic_matrix[:, :, :2] = (idxs - np.array([(img_height - 1) / 2, (img_width - 1) / 2])) * image_ratio
    pic_matrix[:, :, 2] = -distance_viewpoint
    pic_matrix = pic_matrix / np.linalg.norm(pic_matrix, axis=2).reshape(img_height, img_width, 1)
    pic_matrix = (view_matrix[:3, :3].T @ pic_matrix.reshape([img_height, img_width, 3, 1])).reshape(
        [img_height, img_width, 3])
    result = ray_marching(model, pose, eye_pose, pic_matrix, img_width,
                          img_height, max_marching_steps, device, epsilon, furthest_distance, value)
    pose_t = torch.tensor(pose, device=device).float().repeat((result.shape[0], 1))
    data_torch = torch.tensor(result, device=device).float()
    normal = model.J(data_torch, pose_t).reshape([-1, 3])
    normal = normal / np.linalg.norm(normal, axis=1).reshape([-1, 1])
    return result, normal


def ray_marching(model, pose, eye, pic_matrix, w, h, ms, device, e, f, v):
    depth_map = np.zeros(w * h)
    depth_idx = np.ones(w * h)
    re_pic_matrix = pic_matrix.reshape([-1, 3])
    all_data = list()
    model.to(device)
    for i in range(ms):
        input_idx = np.where(depth_idx)[0]
        data = eye + depth_map[input_idx].reshape([-1, 1]) * re_pic_matrix[input_idx]
        data_torch = torch.tensor(data, device=device).float()
        pose_t = torch.tensor(pose, device=device).float().repeat((data.shape[0], 1))
        dist_torch = model.y_torch(data_torch, pose_t) - v
        dist = dist_torch.cpu().detach().numpy().reshape(-1)
        depth_map[input_idx] += (dist / 2)
        del_idx = torch.where(dist_torch < e)[0].cpu().detach().numpy()
        all_data.append(data[del_idx])
        del_idx = input_idx[del_idx]
        depth_idx[del_idx] = 0
        del_idx = np.where(depth_map[input_idx] >= f)[0]
        del_idx = input_idx[del_idx]
        depth_idx[del_idx] = 0
        if not np.any(depth_idx):
            break
    return np.concatenate(all_data, axis=0)

def create_vis_animation(pcl, mesh=None, rot_axis='z'):
    # mesh=None
    print("The point cloud has {} points".format(pcl.shape[0]))
    # pcl = pcl[np.random.choice(np.arange(pcl.shape[0]), np.maximum(15000, pcl.shape[0]))]
    from scipy.spatial.transform import Rotation
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    view_control = vis.get_view_control()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcl)
    T = np.eye(4)

    # T[:3, :3] = Rotation.from_euler('x', np.pi/2).as_matrix()
    # pcd.transform(T)

    vis.add_geometry(pcd)
    vis.update_geometry(pcd)
    
    if mesh is not None:
        mesh.paint_uniform_color([0.8, 0.1, 0.5])
        # mesh = mesh.sample_points_uniformly(number_of_points=15000)
        vis.add_geometry(mesh)
        vis.update_geometry(mesh)

    for i in range(5000):
        # T[:3, :3] = Rotation.from_euler(rot_axis, 0.02).as_matrix()
        # pcd.transform(T)
        # vis.update_geometry(pcd)
        # if mesh is not None:
        #     mesh.transform(T)
        #     vis.update_geometry(mesh)
        vis.update_renderer()
        if not vis.poll_events():
            break
        time.sleep(0.01)
        