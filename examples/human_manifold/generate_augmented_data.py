import os
import argparse
import random
import time
from multiprocessing import Pool, Value
import numpy as np
from redsdf.redsdf_dataset_generator import generate_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--mesh_dir', type=str, default="../../object_models/smpl_meshes", help="path of mesh file")
parser.add_argument('--data_dir', type=str, default="./data", help="path to save generated data")
parser.add_argument('--n_pool', type=int, default=8, help="number of pool")
args = parser.parse_args()


def convert_obj_pcl(filename):
    vertices = list()
    faces = list()
    with open(filename, 'r') as file:
        for line in file:
            data = line.strip().split()
            if data[0] == 'v':
                vertices.append(list(map(float, data[1:])))
            elif data[0] == 'f':
                faces.append(list(map(int, data[1:])))
    vertices = np.array(vertices) - np.array([[-0.000876, -0.211419, 0.027821]])
    faces = np.array(faces)
    points = vertices[faces - 1].mean(axis=-2)
    cross_np = np.cross(vertices[faces[:, 0] - 1] - vertices[faces[:, 1] - 1],
                        vertices[faces[:, 0] - 1] - vertices[faces[:, 2] - 1])
    normal_vector = cross_np / np.linalg.norm(cross_np, axis=1, keepdims=True)
    points = np.array(points)
    normal_vector = np.array(normal_vector)
    return points, normal_vector


def generate_single_file(file):
    start_time = time.process_time()
    idx = int(file[5:10])
    filename = os.path.join(args.mesh_dir, "meshes", file)
    points, normals = convert_obj_pcl(filename)
    result = generate_dataset(points, normals,
                              is_using_point_dependent_weight=True,
                              verbose=False,
                              down_sampling=0.1,
                              epsilons=[-0.1, -0.05, -0.025, -0.01, -0.0075, -0.0050, -0.0025, -0.001,
                                        0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15,
                                        0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
    pose = train_parameters[idx, :]
    with counter.get_lock():
        counter.value += 1
        print("process the data with index {}, complete: {}".format(idx, counter.value))
        print("process duration: {}".format(time.process_time() - start_time))
        return result, pose


def main():
    global train_parameters
    train_parameters = np.load(os.path.join(args.mesh_dir, "poses_smpl.npy"))
    global output_dir
    output_dir = args.data_dir
    n_files_per_chunk = 1000
    global counter
    down_sample_rate = 1

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    poses = list()
    obj_list = sorted(os.listdir(os.path.join(args.mesh_dir, "meshes")))[::down_sample_rate]
    random.shuffle(obj_list)
    for i in range(int(np.ceil(len(obj_list) / n_files_per_chunk))):
        chunk_file_list = obj_list[i * n_files_per_chunk:min((i + 1) * n_files_per_chunk, len(obj_list))]
        counter = Value('i', 0)
        p = Pool(args.n_pool)
        data_pose = p.map(generate_single_file, chunk_file_list)
        data_list = list()
        for data_i, pose_i in data_pose:
            pose_idx = np.repeat(len(poses), data_i.shape[0])[:, np.newaxis]
            data_i = np.concatenate([data_i, pose_idx], axis=1)
            data_list.append(data_i)
            poses.append(pose_i)
        data = np.vstack(data_list)
        np.random.shuffle(data)
        np.save(os.path.join(output_dir, str(i)), data.astype(np.single))
        np.save(os.path.join(output_dir, "poses"), np.vstack(poses).astype(np.single))


if __name__ == "__main__":
    main()
