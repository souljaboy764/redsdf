import os
import random
import time
from copy import deepcopy
import numpy as np
import torch
import numpy
from torch.utils.data import Dataset, DataLoader


class ManifoldDataset(Dataset):
    def __init__(self, dataset):
        if isinstance(dataset, dict):
            N_data = None
            for key in dataset:
                if N_data is None:
                    N_data = dataset[key].shape[0]
                else:
                    assert(N_data == dataset[key].shape[0]), ('key %s has data length=%d!=%d=N_data' %
                                                              (key, dataset[key].shape[0], N_data))
        self.data = dataset

    def __getitem__(self, index):
        if isinstance(self.data, dict):
            data_dict = {}
            for key in self.data:
                data_dict[key] = self.data[key][index].type(torch.float)
            return data_dict
        else:
            return self.data[index].type(torch.float)

    def __len__(self):
        if isinstance(self.data, dict):
            for key in self.data:
                return self.data[key].shape[0]
        else:
            return self.data.shape[0]


def get_file_list(data_dir):
    file_list = list()
    for data_file in os.listdir(data_dir):
        if data_file.endswith(".npy") and not data_file.startswith("poses") and not data_file.startswith("pointnet2_embedding"):
            # file_list.append(os.path.join(data_dir, data_file))
            file_list.append(data_file)
    # random.shuffle(file_list)
    return file_list


def load_data_chunk(data_dir, file_list, train_dataset_ratio, validate_dataset_ratio):
    data_chunk = []

    poses = torch.tensor(np.load(data_dir + "/poses.npy")).type(torch.float)
    for i, file in enumerate(file_list):
        data = np.load(data_dir + "/" + file)
        data_chunk.append(data)
    data_chunk = torch.tensor(np.concatenate(data_chunk)).type(torch.float)
    train_idx = int(data_chunk.shape[0] * train_dataset_ratio)
    validate_idx = int(data_chunk.shape[0] * (train_dataset_ratio+validate_dataset_ratio))
    train_data_chunk = data_chunk[:train_idx]
    validate_data_chunk = data_chunk[train_idx:validate_idx]
    test_data_chunk = data_chunk[validate_idx:]
    return train_data_chunk, validate_data_chunk, test_data_chunk, poses


def construct_loader(data_dir, train_dataset_ratio=0.8, validate_dataset_ratio=0.1, **kwargs):
    file_list = get_file_list(data_dir)
    train_dataset, validate_dataset, test_dataset, poses = load_data_chunk(data_dir, file_list,
                                                                           train_dataset_ratio, validate_dataset_ratio)
    train_data_loader = DataLoader(train_dataset, **kwargs)
    validate_data_loader = DataLoader(validate_dataset, **kwargs)
    test_data_loader = DataLoader(test_dataset, **kwargs)
    return train_data_loader, validate_data_loader, test_data_loader, poses

def construct_loader_category(data_dir, train_dataset_ratio=0.8, validate_dataset_ratio=0.1, **kwargs):
    """_summary_

    Args:
        data_dir (str): path to the data directory of a category containing subdirectories with teh corresponding shapenet hexcode that has the data generated for redsdf.
        train_dataset_ratio (float, optional): _description_. Defaults to 0.8.
        validate_dataset_ratio (float, optional): _description_. Defaults to 0.1.

    Returns:
        _type_: _description_
    """

    pointnet2_embeddings = []
    data = []
    for i, subdir in enumerate(os.listdir(data_dir)):
        print(f"Loading data from subdirectory {subdir} ({i+1}/{len(os.listdir(data_dir))})")
        subdir_path = os.path.join(data_dir, subdir)
        data_file = os.path.join(subdir_path, "0.npy")
        data_i = np.load(data_file)
        data_i[:, -1] = i
        data.append(data_i)
        pointnet2_embeddings.append(np.load(os.path.join(subdir_path, "pointnet2_embedding.npy")).astype(np.float32)[0])
        if i>=10:
            print(f"Loaded {i+1} samples, stopping early for testing purposes.")
            break
        
    data = torch.tensor(np.concatenate(data)).type(torch.float)
    pointnet2_embeddings = torch.tensor(np.array(pointnet2_embeddings)).type(torch.float)
    train_idx = int(data.shape[0] * train_dataset_ratio)
    validate_idx = int(data.shape[0] * (train_dataset_ratio+validate_dataset_ratio))
    print(f"Data shape: {data.shape}, Training samples: {train_idx}, Validation samples: {validate_idx-train_idx}, Test samples: {data.shape[0]-validate_idx}")
    train_data_loader = DataLoader(data[:train_idx], **kwargs)
    validate_data_loader = DataLoader(data[train_idx:validate_idx], **kwargs)
    test_data_loader = DataLoader(data[validate_idx:], **kwargs)
    return train_data_loader, validate_data_loader, test_data_loader, pointnet2_embeddings
