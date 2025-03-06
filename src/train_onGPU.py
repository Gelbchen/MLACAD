# -*- coding: utf-8 -*-
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from torch.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
import logging
import csv
import json
from datetime import datetime
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset.cad_dataset import get_dataloader, CADDataset
from model_mla import DeepCADMLA
from config.defaults import DatasetConfig, TrainingConfig

def save_config(config, save_path):
    with open(save_path, "w") as f:
        json.dump(config, f, indent=4)

# 添加 EarlyStopping 类
class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            return False

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def train_model(model, dataset_cfg, train_cfg, exp_name=None):
    # 移除模型创建部分，因为现在模型将作为参数传入
    # model = DeepCADMLA(...)

    data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    dataset_cfg.data_root = data_root

    split_json_path = os.path.join(data_root, dataset_cfg.train_val_test_split)
    cad_vec_path = os.path.join(data_root, dataset_cfg.cad_vec_dir)

    if not os.path.exists(split_json_path):
        raise FileNotFoundError(f"Split JSON file not found at: {split_json_path}")
    if not os.path.exists(cad_vec_path):
        raise FileNotFoundError(f"CAD vector directory not found at: {cad_vec_path}")

    print(f"Data root: {data_root}")
    print(f"Split JSON path: {split_json_path}")
    print(f"CAD vec path: {cad_vec_path}")

    train_loader = get_dataloader(phase="train", config=dataset_cfg, shuffle=True)
    val_loader = get_dataloader(phase="validation", config=dataset_cfg, shuffle=False)

    # 将模型移动到指定设备
    model = model.to(train_cfg.device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=train_cfg.learning_rate,
        weight_decay=0.01,  # 添加权重衰减
        eps=1e-8  # 提高数值稳定性
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=train_cfg.epochs,
        eta_min=1e-6
    )

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join("logs", current_time)
    os.makedirs(log_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=os.path.join(log_dir, "tensorboard"))
    log_file = os.path.join(log_dir, "training.log")
    logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(message)s")

    config_data = {"DatasetConfig": dataset_cfg.__dict__, "TrainingConfig": train_cfg.__dict__}
    save_config(config_data, os.path.join(log_dir, "config.json"))

    csv_file = os.path.join(log_dir, "training_metrics.csv")
    with open(csv_file, "w", newline="") as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(["Epoch", "Train Loss", "Val Loss"])

    print(f"Log files will be saved in: {log_dir}")

    scaler = GradScaler()
    args_embed_layer = nn.Sequential(
        nn.Linear(dataset_cfg.args_dim, dataset_cfg.d_model),
        nn.ReLU(),
        nn.Dropout(dataset_cfg.dropout)
    ).to(train_cfg.device)

    # 在模型创建后添加自定义初始化
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    model.apply(init_weights)

    # 收集训练和验证损失
    train_losses = []
    val_losses = []

    for epoch in range(train_cfg.epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad()  # 每个epoch开始时清零梯度
        
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch} Training")):
            command = batch["command"].to(train_cfg.device)
            args = batch["args"].to(train_cfg.device).float()
            
            # 处理参数嵌入
            batch_size, seq_len, args_dim = args.shape
            args_reshaped = args.view(-1, args_dim)
            args_embed = args_embed_layer(args_reshaped)
            args_embed = args_embed.view(batch_size, seq_len, -1)
            
            # 兼容 DataParallel 的属性访问
            embedding = model.module.embedding if isinstance(model, nn.DataParallel) else model.embedding
            command_embed = embedding(command)
            input_embed = command_embed + args_embed
            
            # 输入序列 (嵌入后的)
            input_seq = input_embed[:, :-1]
            
            # 目标序列 (嵌入后的，用于模型输入)
            target_embed = command_embed[:, 1:]
            
            # 目标索引 (原始命令，用于损失计算)
            target_idx = command[:, 1:]
            
            with autocast(enabled=train_cfg.fp16_training):
                outputs = model(input_seq, target_embed)
                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    print("Outputs contain NaN or Inf")
                loss = criterion(outputs.view(-1, dataset_cfg.vocab_size), target_idx.contiguous().view(-1))
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    print("Loss contains NaN or Inf")
                loss = loss.mean()
                # 梯度累积 - 将损失除以累积步数
                loss = loss / train_cfg.gradient_accumulation_steps
            
            # 使用梯度缩放器缩放损失并反向传播
            scaler.scale(loss).backward()
            
            # 在指定的步数上累积梯度后更新模型
            if (i + 1) % train_cfg.gradient_accumulation_steps == 0:
                # 取消缩放以便进行梯度裁剪
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            total_loss += loss.item() * train_cfg.gradient_accumulation_steps  # 恢复原始损失大小用于记录
            
            # 调试信息
            if i == 0:  # 只打印第一个批次
                print(f"Command dtype: {command.dtype}, shape: {command.shape}")
                print(f"Args dtype: {args.dtype}, shape: {args.shape}")
                print(f"Command range: [{command.min()}, {command.max()}]")
                print(f"Args range: [{args.min()}, {args.max()}]")
                
                # 检查数据是否包含NaN或Inf
                if torch.isnan(command).any() or torch.isinf(command).any():
                    print("Command contains NaN or Inf")
                if torch.isnan(args).any() or torch.isinf(args).any():
                    print("Args contain NaN or Inf")
        
        # 确保最后一个批次的梯度也得到应用
        if len(train_loader) % train_cfg.gradient_accumulation_steps != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
        scheduler.step()
        
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        logging.info(f"Epoch {epoch}, Train Loss: {avg_train_loss}")
        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        with open(csv_file, "a", newline="") as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow([epoch, avg_train_loss])

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} Validation"):
                command = batch["command"].to(train_cfg.device)
                args = batch["args"].to(train_cfg.device).float()

                print(f"Command dtype: {command.dtype}, shape: {command.shape}")
                print(f"Args dtype: {args.dtype}, shape: {args.shape}")
                print(f"Command range: [{command.min()}, {command.max()}]")
                print(f"Args range: [{args.min()}, {args.max()}]")

                # 处理参数嵌入
                batch_size, seq_len, args_dim = args.shape
                args_reshaped = args.view(-1, args_dim)
                args_embed = args_embed_layer(args_reshaped)
                args_embed = args_embed.view(batch_size, seq_len, -1)

                embedding = model.module.embedding if isinstance(model, nn.DataParallel) else model.embedding
                command_embed = embedding(command)
                input_embed = command_embed + args_embed

                input_seq = input_embed[:, :-1]
                target_embed = command_embed[:, 1:]
                target_idx = command[:, 1:]

                outputs = model(input_seq, target_embed)
                loss = criterion(outputs.view(-1, dataset_cfg.vocab_size), target_idx.contiguous().view(-1))
                loss = loss.mean()
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        logging.info(f"Epoch {epoch}, Val Loss: {avg_val_loss}")
        writer.add_scalar("Loss/validation", avg_val_loss, epoch)
        with open(csv_file, "a", newline="") as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow([epoch, None, avg_val_loss])

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(log_dir, f"model_epoch_{epoch}.pth"))

        # 增加学习率检查
        print(f"Current learning rate: {optimizer.param_groups[0]['lr']}")

        # 增加模型参数梯度检查
        max_grad_norm = 0.0
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_norm = param.grad.norm().item()
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    print(f"Gradient contains NaN or Inf in {name}")
                max_grad_norm = max(max_grad_norm, grad_norm)
                if grad_norm > 10:
                    print(f"大梯度警告: {name} - grad_norm: {grad_norm}")

    torch.save(model.state_dict(), os.path.join(log_dir, "model_final.pth"))
    with open(os.path.join(log_dir, "training_metrics.json"), "w") as f:
        json.dump({"train_loss": train_losses, "val_loss": val_losses}, f, indent=4)

    print(f"Training completed. Results saved in: {log_dir}")

    # 返回模型和损失
    return model, train_losses, val_losses

# 添加新的main函数调用train_model
def main():
    dataset_cfg = DatasetConfig()
    train_cfg = TrainingConfig()

    # 降低资源需求（可选，根据硬件调整）
    dataset_cfg.batch_size = 32
    dataset_cfg.num_workers = 0
    train_cfg.batch_size = 32

    # 创建模型
    model = DeepCADMLA(
        vocab_size=dataset_cfg.vocab_size,
        d_model=dataset_cfg.d_model,
        num_heads=dataset_cfg.num_heads,
        num_layers=dataset_cfg.num_layers,
        d_latent=dataset_cfg.d_latent,
        dropout=dataset_cfg.dropout
    )

    # 调用train_model函数
    model, train_losses, val_losses = train_model(model, dataset_cfg, train_cfg)

    print("训练完成")

if __name__ == '__main__':
    main()