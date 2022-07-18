import numpy as np
import numpy.linalg as npla
from scipy.spatial.ckdtree import cKDTree
import copy
import open3d as o3d


def generate_dataset(points, normal_vectors, verbose=False, *args, **kwargs):
    clean_aug_data = kwargs.get('clean_aug_data', True)
    is_using_point_dependent_weight = kwargs.get('is_using_point_dependent_weight', False)
    aug_clean_thresh = kwargs.get('aug_clean_thresh', 1e-1)
    epsilons = kwargs.get('epsilons', np.arange(-0.5, 0.5, 0.025))
    is_delete_outliers = kwargs.get("is_delete_outliers", False)
    outliers_in_augmented = kwargs.get("outliers_in_augmented", -0.02)
    down_sampling = kwargs.get("down_sampling", 1)
    valid_idxs = list()
    data = list()
    norm_level_data = list()
    cov_nullspace = list()
    N_data = points.shape[0]
    N_aug_data = N_data
    dim_ambient = points.shape[1]
    if verbose:
        print('N_data = %d' % N_data)
    kd_tree = cKDTree(data=points)
    dim_normal_space = 1

    # delete outliers
    if is_delete_outliers:
        level_mult_eigvec = -abs(outliers_in_augmented) * normal_vectors
        new_data = (points + level_mult_eigvec)
        valid_idx = list(range(N_data))
        if clean_aug_data:
            del_idx = np.where(npla.norm(new_data[kd_tree.query(new_data)[1]] - new_data, axis=1) >
                               (aug_clean_thresh * abs(outliers_in_augmented)))[0]
            valid_idx = np.delete(valid_idx, del_idx)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(new_data[valid_idx])
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=15, std_ratio=1.5)
        del_idx = np.delete(valid_idx, ind)
        aug_idx = np.delete(np.arange(N_data), del_idx)
        points = points[aug_idx]
        normal_vectors = normal_vectors[aug_idx]
        N_aug_data = len(aug_idx)
        kd_tree = cKDTree(data=points)
        if verbose:
            print('Removed outliers = {} / {}'.format(del_idx.shape[0], N_aug_data))
    data.append(points)
    norm_level_data.append(np.zeros((N_aug_data, 1)))
    cov_nullspace.append(copy.deepcopy(normal_vectors))
    normal_vectors = normal_vectors.reshape([N_aug_data, dim_ambient, dim_normal_space])
    valid_idxs.append(list(range(N_aug_data)))

    # append more points based on the Normal Space Eigenvectors and epsilon:
    i = 0
    for epsilon in epsilons:
        i += 1
        if verbose:
            print('Data Augmentation %d/%d' % (i, len(epsilons)))
        if -1e-10 < epsilon < 1e-10:
            if verbose:
                print('epsilon is zero, continue..')
            continue

        level_mult_eigvec = epsilon * normal_vectors[:, :, 0]

        new_data = (points + level_mult_eigvec)
        # delete indices from augmented data if they do not fulfill the neighborhood condition
        valid_idx = list(range(N_aug_data))
        if clean_aug_data:
            del_idx = np.where(npla.norm(new_data[kd_tree.query(new_data)[1]] - new_data, axis=1) >
                               (aug_clean_thresh * abs(epsilon)))[0]
            valid_idx = np.delete(valid_idx, del_idx).tolist()

        N_aug = len(valid_idx)
        if verbose:
            print('Accepted aug points ', N_aug, ' / ', N_data)
        if N_aug == 0:
            continue
        valid_data = new_data[valid_idx]
        valid_idxs.append(valid_idx)
        data.append(valid_data)
        norm_level_data.append(
            (level_mult_eigvec[valid_idx].reshape([N_aug, 1, -1]) @ normal_vectors[valid_idx]).reshape([N_aug, -1]))
        cov_nullspace.append(
            copy.deepcopy(normal_vectors[valid_idx].reshape(N_aug, dim_ambient)))

    # append weight
    norm_level_weight = list()
    if is_using_point_dependent_weight:
        count_valid = np.zeros(N_aug_data)
        number_of_aug = len(valid_idxs)
        for valid_idx in valid_idxs:
            count_valid[valid_idx] += 1
        count_valid = (number_of_aug / count_valid).reshape([N_aug_data, 1])
        for i in range(number_of_aug):
            norm_level_weight.append((np.ones((N_aug_data, 1)) * count_valid)[valid_idxs[i]])
    else:
        for valid_idx in valid_idxs:
            norm_level_weight.append(np.ones((len(valid_idx), 1)))

    data = np.concatenate(data, axis=0)
    norm_level_data = np.concatenate(norm_level_data, axis=0)
    cov_nullspace = np.concatenate(cov_nullspace, axis=0)
    norm_level_weight = np.concatenate(norm_level_weight, axis=0)
    all_aug_data = np.concatenate([data, cov_nullspace, norm_level_data, norm_level_weight], axis=1).astype('float32')

    # down sampling
    if down_sampling != 1:
        assert(0 < down_sampling < 1)
        N_all_data = all_aug_data.shape[0]
        samples = np.arange(N_all_data)
        samples = np.random.choice(samples, int(N_all_data * down_sampling), replace=False)
        all_aug_data = all_aug_data[samples]

    return all_aug_data
