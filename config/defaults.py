import torch
from dataclasses import dataclass

@dataclass
class DatasetConfig:
    data_root: str = "data"
    cad_vec_dir: str = "cad_vec"
    train_val_test_split: str = "train_val_test_split.json"
    max_n_ext: int = 20     # MAX_N_EXT
    max_n_loops: int = 10   # MAX_N_LOOPS
    max_n_curves: int = 10  # MAX_N_CURVES
    max_total_len: int = 256  # 减小序列长度，从512降到256
    augment: bool = True
    batch_size: int = 32    # 显著减小批量大小，从128降到32
    num_workers: int = 4    # 减少工作进程数，避免内存占用过大
    vocab_size: int = 1000  
    args_dim: int = 16     
    
    # 调整模型配置以减少显存占用
    d_model: int = 256      # 从512降到256
    num_heads: int = 4      # 从8降到4
    num_layers: int = 4     # 从8降到4
    d_latent: int = 256     # 从512降到256
    dropout: float = 0.1

@dataclass
class TrainingConfig:
    batch_size: int = 32    # 与DatasetConfig保持一致
    learning_rate: float = 5e-5  # 因为批量变小，适当减小学习率
    weight_decay: float = 0.01
    dropout: float = 0.2
    epochs: int = 200
    warmup_steps: int = 5000    # 减少预热步数
    gradient_accumulation_steps: int = 4  # 使用梯度累积来补偿小批量
    grad_clip: float = 0.5
    fp16_training: bool = True  # 启用混合精度训练
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_multi_gpu: bool = False  # 单GPU情况下设为False

class Config:
    def __init__(self):
        self.dataset = DatasetConfig()
        self.training = TrainingConfig()

    @staticmethod
    def get_config() -> 'Config':
        return Config()

defaults = Config()

# 实验组1：基础模型（更小，适合有限GPU内存）
base_config = {
    'd_model': 192,
    'num_heads': 4,
    'num_layers': 4,
    'd_latent': 192
}

# 实验组2：大模型（适当调小）
large_config = {
    'd_model': 256,
    'num_heads': 4,
    'num_layers': 6,
    'd_latent': 256
}

# 实验组3：深层模型（调小宽度，保持深度）
deep_config = {
    'd_model': 192,
    'num_heads': 4,
    'num_layers': 8,
    'd_latent': 192
}

# 实验组7：标准训练
standard_training = {
    'batch_size': 32,
    'learning_rate': 5e-5,
    'weight_decay': 0.01
}

# 实验组8：大批量训练
large_batch = {
    'batch_size': 64,  # 减小大批量的大小
    'learning_rate': 1e-4,
    'weight_decay': 0.01
}

# 实验组9：渐进式学习
progressive = {
    'batch_size': 32,
    'learning_rate': 5e-5,
    'curriculum_learning': True
}
