import os
import sys
import time
import torch
import argparse
from datetime import datetime
from copy import deepcopy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from train_onGPU import train_model
from config.defaults import DatasetConfig, TrainingConfig
from model_mla import DeepCADMLA, StandardAttention, LocalAttention, SparseAttention
from experiment_manager import ExperimentManager, count_parameters

class AblationDeepCAD(DeepCADMLA):
    """用于消融实验的模型变体"""
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_latent, 
                 dropout=0.1, attention_type="mla", use_residual=True, 
                 use_layer_norm=True, use_dropout=True):
        super().__init__(vocab_size, d_model, num_heads, num_layers, d_latent, dropout)
        
        # 修改注意力类型
        self.attention_type = attention_type
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm
        self.use_dropout = use_dropout
        
        # 根据配置重新构建编码器和解码器
        self._rebuild_model()
        
    def _rebuild_model(self):
        """根据消融实验配置重建模型"""
        # 实现根据消融配置构建模型的逻辑
        # ...

def main():
    parser = argparse.ArgumentParser(description="运行消融实验")
    parser.add_argument("--attention", choices=["mla", "standard", "local", "sparse"], default="mla", help="注意力机制类型")
    parser.add_argument("--no-residual", action="store_true", help="禁用残差连接")
    parser.add_argument("--no-layernorm", action="store_true", help="禁用层归一化")
    parser.add_argument("--no-dropout", action="store_true", help="禁用Dropout")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="训练设备")
    args = parser.parse_args()
    
    # 实验管理器
    manager = ExperimentManager(base_log_dir="ablation_experiments")
    
    # 基础配置
    dataset_cfg = DatasetConfig()
    training_cfg = TrainingConfig()
    training_cfg.epochs = args.epochs
    training_cfg.device = args.device
    
    # 消融实验配置
    ablation_configs = [
        {
            "name": "完整模型", 
            "attention_type": "mla",
            "use_residual": True,
            "use_layer_norm": True,
            "use_dropout": True
        },
        {
            "name": f"{args.attention}_无残差" if args.no_residual else f"{args.attention}_标准",
            "attention_type": args.attention,
            "use_residual": not args.no_residual,
            "use_layer_norm": not args.no_layernorm,
            "use_dropout": not args.no_dropout
        }
    ]
    
    for config in ablation_configs:
        # 实验名称
        exp_name = config["name"]
        
        # 添加到实验管理器
        manager.add_experiment(exp_name, dataset_cfg, training_cfg, model_type="AblationDeepCAD")
        
        # 保存实验配置
        manager.save_config(exp_name)
        
        print(f"开始消融实验: {exp_name}")
        start_time = time.time()
        
        # 创建模型
        model = AblationDeepCAD(
            vocab_size=dataset_cfg.vocab_size,
            d_model=dataset_cfg.d_model,
            num_heads=dataset_cfg.num_heads,
            num_layers=dataset_cfg.num_layers,
            d_latent=dataset_cfg.d_latent,
            dropout=dataset_cfg.dropout if config["use_dropout"] else 0.0,
            attention_type=config["attention_type"],
            use_residual=config["use_residual"],
            use_layer_norm=config["use_layer_norm"],
            use_dropout=config["use_dropout"]
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
        
        print(f"完成消融实验: {exp_name}")
        
    # 保存所有实验并生成可视化对比
    manager.save_all()
    print(f"所有消融实验完成，结果保存在 {manager.experiment_dir}")

if __name__ == "__main__":
    main() 