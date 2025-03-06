import os
import sys
import time
import torch
import argparse
from datetime import datetime
from copy import deepcopy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from train_onGPU import train_model
from config.defaults import DatasetConfig, TrainingConfig, base_config, large_config, deep_config, standard_training, large_batch, progressive
from model_mla import DeepCADMLA
from experiment_manager import ExperimentManager, count_parameters

def create_config_from_dict(base_config, config_dict):
    """根据配置字典更新配置对象"""
    config = deepcopy(base_config)
    for key, value in config_dict.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config

def main():
    parser = argparse.ArgumentParser(description="运行多个实验配置")
    parser.add_argument("--configs", nargs="+", choices=["base", "large", "deep", "all"], default=["base"], help="要运行的配置")
    parser.add_argument("--trainings", nargs="+", choices=["standard", "large_batch", "progressive", "all"], default=["standard"], help="要运行的训练策略")
    parser.add_argument("--epochs", type=int, default=50, help="每个实验的训练轮数")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--batch_size", type=int, default=32, help="批处理大小")
    args = parser.parse_args()
    
    # 解析配置选项
    model_configs = []
    if "all" in args.configs:
        model_configs = ["base", "large", "deep"]
    else:
        model_configs = args.configs
        
    training_configs = []
    if "all" in args.trainings:
        training_configs = ["standard", "large_batch", "progressive"]
    else:
        training_configs = args.trainings
        
    # 实验管理器
    manager = ExperimentManager()
    
    # 基础配置
    base_dataset_cfg = DatasetConfig()
    base_training_cfg = TrainingConfig()
    
    # 降低内存使用
    base_dataset_cfg.batch_size = args.batch_size  # 使用更小的批量大小
    base_dataset_cfg.num_workers = 4  # 减少数据加载线程数
    base_training_cfg.batch_size = args.batch_size
    base_training_cfg.epochs = args.epochs
    base_training_cfg.device = args.device
    base_training_cfg.gradient_accumulation_steps = 4  # 增加梯度累积步数
    
    # 配置字典映射 - 调整模型大小
    model_config_dict = {
        "base": {
            'd_model': 256,
            'num_heads': 4,
            'num_layers': 4,
            'd_latent': 256
        },
        "large": {
            'd_model': 384,
            'num_heads': 8,
            'num_layers': 6,
            'd_latent': 384
        },
        "deep": {
            'd_model': 256,
            'num_heads': 4, 
            'num_layers': 12,
            'd_latent': 256
        }
    }
    
    training_config_dict = {
        "standard": {
            'batch_size': args.batch_size,
            'learning_rate': 1e-4,
            'weight_decay': 0.01
        },
        "large_batch": {
            'batch_size': args.batch_size * 2,  # 适当增加，但不要太大
            'learning_rate': 2e-4,
            'weight_decay': 0.01
        },
        "progressive": {
            'batch_size': args.batch_size,
            'learning_rate': 1e-4,
            'curriculum_learning': True
        }
    }
    
    # 创建所有实验组合
    for model_config_name in model_configs:
        model_config = model_config_dict[model_config_name]
        
        for training_config_name in training_configs:
            training_config = training_config_dict[training_config_name]
            
            # 创建该实验的配置
            dataset_cfg = create_config_from_dict(base_dataset_cfg, model_config)
            training_cfg = create_config_from_dict(base_training_cfg, training_config)
            
            # 实验名称
            exp_name = f"{model_config_name}_{training_config_name}"
            
            # 添加到实验管理器
            manager.add_experiment(exp_name, dataset_cfg, training_cfg)
            
            # 保存实验配置
            manager.save_config(exp_name)
            
            print(f"开始实验: {exp_name}")
            start_time = time.time()
            
            # 创建模型
            model = DeepCADMLA(
                vocab_size=dataset_cfg.vocab_size,
                d_model=dataset_cfg.d_model,
                num_heads=dataset_cfg.num_heads,
                num_layers=dataset_cfg.num_layers,
                d_latent=dataset_cfg.d_latent,
                dropout=dataset_cfg.dropout
            )
            
            # 记录参数量
            param_count = count_parameters(model)
            manager.experiments[exp_name]["results"]["param_count"] = param_count
            print(f"模型参数量: {param_count:,}")
            
            # 训练模型并获取结果
            model, train_losses, val_losses = train_model(model, dataset_cfg, training_cfg, exp_name)
            
            # 记录训练时间
            train_time = (time.time() - start_time) / 3600  # 转换为小时
            manager.experiments[exp_name]["results"]["training_time"] = train_time
            print(f"训练时间: {train_time:.2f} 小时")
            
            # 记录训练和验证损失
            for epoch, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses)):
                manager.log_results(exp_name, epoch, train_loss, val_loss)
                
            # 保存最终模型
            manager.save_model(exp_name, model, args.epochs - 1)
            
            # 保存实验结果
            manager.save_results(exp_name)
            
            print(f"完成实验: {exp_name}")
            
    # 保存所有实验并生成可视化对比
    manager.save_all()
    print(f"所有实验完成，结果保存在 {manager.experiment_dir}")

if __name__ == "__main__":
    main() 