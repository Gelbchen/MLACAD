import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import asdict
from datetime import datetime
from config.defaults import DatasetConfig, TrainingConfig
from model_mla import DeepCADMLA

class ExperimentManager:
    def __init__(self, base_log_dir="experiments"):
        self.base_log_dir = base_log_dir
        self.experiments = {}
        self.current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.experiment_dir = os.path.join(base_log_dir, self.current_time)
        os.makedirs(self.experiment_dir, exist_ok=True)
        
    def add_experiment(self, name, dataset_config, training_config, model_type="DeepCADMLA"):
        """添加一个实验配置"""
        self.experiments[name] = {
            "dataset_config": dataset_config,
            "training_config": training_config,
            "model_type": model_type,
            "results": {}
        }
        
    def log_results(self, name, epoch, train_loss, val_loss, metrics=None):
        """记录实验结果"""
        if name not in self.experiments:
            raise ValueError(f"实验 {name} 未注册")
            
        if "log" not in self.experiments[name]["results"]:
            self.experiments[name]["results"]["log"] = []
            
        log_entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss
        }
        
        if metrics:
            log_entry.update(metrics)
            
        self.experiments[name]["results"]["log"].append(log_entry)
        
    def save_model(self, name, model, epoch):
        """保存模型"""
        if name not in self.experiments:
            raise ValueError(f"实验 {name} 未注册")
            
        exp_dir = os.path.join(self.experiment_dir, name)
        os.makedirs(exp_dir, exist_ok=True)
        
        model_path = os.path.join(exp_dir, f"model_epoch_{epoch}.pth")
        torch.save(model.state_dict(), model_path)
        
    def save_config(self, name):
        """保存实验配置"""
        if name not in self.experiments:
            raise ValueError(f"实验 {name} 未注册")
            
        exp_dir = os.path.join(self.experiment_dir, name)
        os.makedirs(exp_dir, exist_ok=True)
        
        config_path = os.path.join(exp_dir, "config.json")
        
        # 转换DatasetConfig和TrainingConfig为字典
        dataset_config = asdict(self.experiments[name]["dataset_config"])
        training_config = asdict(self.experiments[name]["training_config"])
        
        with open(config_path, "w") as f:
            json.dump({
                "dataset_config": dataset_config,
                "training_config": training_config,
                "model_type": self.experiments[name]["model_type"]
            }, f, indent=4)
            
    def save_results(self, name):
        """保存实验结果"""
        if name not in self.experiments:
            raise ValueError(f"实验 {name} 未注册")
            
        exp_dir = os.path.join(self.experiment_dir, name)
        os.makedirs(exp_dir, exist_ok=True)
        
        results_path = os.path.join(exp_dir, "results.json")
        
        with open(results_path, "w") as f:
            json.dump(self.experiments[name]["results"], f, indent=4)
            
    def save_all(self):
        """保存所有实验数据"""
        for name in self.experiments:
            self.save_config(name)
            self.save_results(name)
            
        # 保存实验比较结果
        self.visualize_all()
            
    def visualize_all(self):
        """可视化所有实验结果比较"""
        if not self.experiments:
            return
            
        # 创建可视化目录
        vis_dir = os.path.join(self.experiment_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        # 训练损失对比
        self._plot_loss_comparison("train_loss", "训练损失对比", os.path.join(vis_dir, "train_loss_comparison.png"))
        
        # 验证损失对比
        self._plot_loss_comparison("val_loss", "验证损失对比", os.path.join(vis_dir, "val_loss_comparison.png"))
        
        # 模型架构对比
        self._plot_architecture_comparison(os.path.join(vis_dir, "architecture_comparison.png"))
        
        # 训练时间对比
        if all("training_time" in exp["results"] for exp in self.experiments.values()):
            self._plot_time_comparison(os.path.join(vis_dir, "time_comparison.png"))
            
        # 参数量对比
        if all("param_count" in exp["results"] for exp in self.experiments.values()):
            self._plot_param_comparison(os.path.join(vis_dir, "param_comparison.png"))
            
    def _plot_loss_comparison(self, loss_type, title, save_path):
        """绘制损失对比图"""
        plt.figure(figsize=(12, 8))
        
        for name, exp in self.experiments.items():
            if "log" in exp["results"]:
                epochs = [entry["epoch"] for entry in exp["results"]["log"]]
                losses = [entry[loss_type] for entry in exp["results"]["log"]]
                plt.plot(epochs, losses, label=name)
                
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()
        
    def _plot_architecture_comparison(self, save_path):
        """绘制模型架构对比图"""
        names = list(self.experiments.keys())
        d_models = [exp["dataset_config"].d_model for exp in self.experiments.values()]
        num_heads = [exp["dataset_config"].num_heads for exp in self.experiments.values()]
        num_layers = [exp["dataset_config"].num_layers for exp in self.experiments.values()]
        
        x = np.arange(len(names))
        width = 0.2
        
        plt.figure(figsize=(12, 8))
        plt.bar(x - width, d_models, width, label='隐藏层维度')
        plt.bar(x, num_heads, width, label='注意力头数')
        plt.bar(x + width, num_layers, width, label='层数')
        
        plt.xlabel('实验配置')
        plt.ylabel('数值')
        plt.title('模型架构参数对比')
        plt.xticks(x, names, rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
    def _plot_time_comparison(self, save_path):
        """绘制训练时间对比图"""
        names = list(self.experiments.keys())
        times = [exp["results"]["training_time"] for exp in self.experiments.values()]
        
        plt.figure(figsize=(12, 8))
        bars = plt.bar(names, times)
        
        # 在柱状图上添加具体数值
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}h',
                    ha='center', va='bottom')
        
        plt.xlabel('实验配置')
        plt.ylabel('训练时间 (小时)')
        plt.title('模型训练时间对比')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
    def _plot_param_comparison(self, save_path):
        """绘制参数量对比图"""
        names = list(self.experiments.keys())
        params = [exp["results"]["param_count"] / 1000000 for exp in self.experiments.values()]
        
        plt.figure(figsize=(12, 8))
        bars = plt.bar(names, params)
        
        # 在柱状图上添加具体数值
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}M',
                    ha='center', va='bottom')
        
        plt.xlabel('实验配置')
        plt.ylabel('参数量 (百万)')
        plt.title('模型参数量对比')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

def count_parameters(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) 