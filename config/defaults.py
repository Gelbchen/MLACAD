import torch
from dataclasses import dataclass

@dataclass
class DatasetConfig:
    data_root: str = "data"  # 改为 "data"，而不是 "../data"
    cad_vec_dir: str = "cad_vec"
    train_val_test_split: str = "train_val_test_split.json"
    max_total_len: int = 512
    augment: bool = True
    batch_size: int = 32
    num_workers: int = 4
    vocab_size: int = 1000
    d_model: int = 256
    num_heads: int = 8
    num_layers: int = 4
    d_latent: int = 128

@dataclass
class TrainingConfig:
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 100
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class Config:
    def __init__(self):
        self.dataset = DatasetConfig()
        self.training = TrainingConfig()

    @staticmethod
    def get_config() -> 'Config':
        return Config()

defaults = Config()