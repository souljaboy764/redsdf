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
        if data_file.endswith(".npy") and not data_file.startswith("poses"):
            file_list.append(data_file)
    random.shuffle(file_list)
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


