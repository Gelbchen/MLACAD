from torch.utils.data import Dataset, DataLoader
import torch
import os
import json
import h5py
import random
import numpy as np
from cadlib.macro import *


def worker_init_fn(worker_id):
    np.random.seed()

def get_dataloader(phase, config, shuffle=None):
    is_shuffle = phase == 'train' if shuffle is None else shuffle
    try:
        dataset = CADDataset(phase, config)
        dataloader = DataLoader(
            dataset, 
            batch_size=config.batch_size, 
            shuffle=is_shuffle, 
            num_workers=config.num_workers,
            worker_init_fn=worker_init_fn
        )
        return dataloader
    except Exception as e:
        print(f"创建数据加载器时出错: {str(e)}")
        raise


class CADDataset(Dataset):
    def __init__(self, phase, config):
        super(CADDataset, self).__init__()
        # 基本路径和配置
        self.raw_data = os.path.join(config.data_root, "cad_vec")
        self.phase = phase
        self.aug = config.augment
        self.path = os.path.join(config.data_root, "train_val_test_split.json")
        
        # 检查文件是否存在
        if not os.path.exists(self.raw_data):
            raise FileNotFoundError(f"数据目录不存在: {self.raw_data}")
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"分割文件不存在: {self.path}")

        # 加载数据分割信息
        try:
            with open(self.path, "r") as fp:
                self.all_data = json.load(fp)[phase]
        except Exception as e:
            print(f"加载数据分割文件时出错: {str(e)}")
            raise

        # 配置参数，使用getattr提供默认值
        self.max_n_loops = getattr(config, 'max_n_loops', 20)    # Number of paths (N_P)
        self.max_n_curves = getattr(config, 'max_n_curves', 20)  # Number of commands (N_C)
        self.max_total_len = getattr(config, 'max_total_len', 256)
        self.size = 256

    def get_data_by_id(self, data_id):
        try:
            idx = self.all_data.index(data_id)
            return self.__getitem__(idx)
        except ValueError:
            print(f"未找到ID: {data_id}")
            raise

    def __getitem__(self, index):
        try:
            data_id = self.all_data[index]
            h5_path = os.path.join(self.raw_data, data_id + ".h5")
            
            if not os.path.exists(h5_path):
                raise FileNotFoundError(f"H5文件不存在: {h5_path}")

            with h5py.File(h5_path, "r") as fp:
                cad_vec = fp["vec"][:]  # (len, 1 + N_ARGS)

            # 数据增强
            if self.aug and self.phase == "train":
                command1 = cad_vec[:, 0]
                ext_indices1 = np.where(command1 == EXT_IDX)[0]
                if len(ext_indices1) > 1 and random.uniform(0, 1) > 0.5:
                    ext_vec1 = np.split(cad_vec, ext_indices1 + 1, axis=0)[:-1]
            
                    data_id2 = random.choice(self.all_data)
                    h5_path2 = os.path.join(self.raw_data, data_id2 + ".h5")
                    with h5py.File(h5_path2, "r") as fp:
                        cad_vec2 = fp["vec"][:]
                    command2 = cad_vec2[:, 0]
                    ext_indices2 = np.where(command2 == EXT_IDX)[0]
                    ext_vec2 = np.split(cad_vec2, ext_indices2 + 1, axis=0)[:-1]
            
                    n_replace = random.randint(1, min(len(ext_vec1) - 1, len(ext_vec2)))
                    old_idx = sorted(random.sample(list(range(len(ext_vec1))), n_replace))
                    new_idx = sorted(random.sample(list(range(len(ext_vec2))), n_replace))
                    for i in range(len(old_idx)):
                        ext_vec1[old_idx[i]] = ext_vec2[new_idx[i]]
            
                    sum_len = 0
                    new_vec = []
                    for i in range(len(ext_vec1)):
                        sum_len += len(ext_vec1[i])
                        if sum_len > self.max_total_len:
                            break
                        new_vec.append(ext_vec1[i])
                    cad_vec = np.concatenate(new_vec, axis=0)

            # 补零处理
            pad_len = max(0, self.max_total_len - cad_vec.shape[0])  # 确保pad_len不为负
            if pad_len > 0:
                cad_vec = np.concatenate([cad_vec, EOS_VEC[np.newaxis].repeat(pad_len, axis=0)], axis=0)

            # 转换为tensor
            command = torch.tensor(cad_vec[:, 0], dtype=torch.long)
            args = torch.tensor(cad_vec[:, 1:], dtype=torch.long)

            return {"command": command, "args": args, "id": data_id}
            
        except Exception as e:
            print(f"处理索引 {index} (ID: {data_id}) 时出错: {str(e)}")
            raise

    def __len__(self):
        return len(self.all_data)
