from torch.utils.data import Dataset, DataLoader
import torch
import os
import json
import h5py
import random
import numpy as np
from cadlib.macro import *

# 定义一个全局函数用于 worker 初始化
def worker_init_fn(worker_id):
    np.random.seed()

def get_dataloader(phase, config, shuffle=None):
    is_shuffle = phase == 'train' if shuffle is None else shuffle
    dataset = CADDataset(phase, config)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=is_shuffle,
        num_workers=config.num_workers,
        worker_init_fn=worker_init_fn  # 使用命名函数

    )
    return dataloader

class CADDataset(Dataset):
    def __init__(self, phase, config):
        super(CADDataset, self).__init__()
        self.raw_data = os.path.join(config.data_root, "cad_vec")
        self.phase = phase
        self.aug = config.augment
        self.path = os.path.join(config.data_root, "train_val_test_split.json")
        print(f"Opening split JSON at: {self.path}")
        with open(self.path, "r") as fp:
            self.all_data = json.load(fp)[phase]

        self.max_total_len = config.max_total_len
        self.size = 256

    def get_data_by_id(self, data_id):
        idx = self.all_data.index(data_id)
        return self.__getitem__(idx)

    def __getitem__(self, index):
        data_id = self.all_data[index]
        h5_path = os.path.join(self.raw_data, data_id + ".h5")
        with h5py.File(h5_path, "r") as fp:
            cad_vec = fp["vec"][:]

        if self.aug and self.phase == "train":
            command = cad_vec[:, 0]
            ext_indices = np.where(command == EXT_IDX)[0]
            if len(ext_indices) > 1 and random.uniform(0, 1) > 0.5:
                ext_vec = np.split(cad_vec, ext_indices + 1, axis=0)[:-1]

                data_id2 = random.choice(self.all_data)
                h5_path2 = os.path.join(self.raw_data, data_id2 + ".h5")
                with h5py.File(h5_path2, "r") as fp:
                    cad_vec2 = fp["vec"][:]
                command2 = cad_vec2[:, 0]
                ext_indices2 = np.where(command2 == EXT_IDX)[0]
                ext_vec2 = np.split(cad_vec2, ext_indices2 + 1, axis=0)[:-1]

                n_replace = random.randint(1, min(len(ext_vec) - 1, len(ext_vec2)))
                old_idx = sorted(random.sample(list(range(len(ext_vec))), n_replace))
                new_idx = sorted(random.sample(list(range(len(ext_vec2))), n_replace))
                for i, (o_idx, n_idx) in enumerate(zip(old_idx, new_idx)):
                    ext_vec[o_idx] = ext_vec2[n_idx]

                sum_len = 0
                new_vec = []
                for vec in ext_vec:
                    sum_len += len(vec)
                    if sum_len > self.max_total_len:
                        break
                    new_vec.append(vec)
                cad_vec = np.concatenate(new_vec, axis=0)

        pad_len = self.max_total_len - cad_vec.shape[0]
        if pad_len > 0:
            cad_vec = np.concatenate([cad_vec, EOS_VEC[np.newaxis].repeat(pad_len, axis=0)], axis=0)
        elif pad_len < 0:
            cad_vec = cad_vec[:self.max_total_len]

        command = cad_vec[:, 0]
        args = cad_vec[:, 1:]
        return {
            "command": torch.tensor(command, dtype=torch.long),
            "args": torch.tensor(args, dtype=torch.long),
            "id": data_id
        }

    def __len__(self):
        return len(self.all_data)