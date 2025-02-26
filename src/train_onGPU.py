# -*- coding: utf-8 -*-
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import logging
import csv
import json
from datetime import datetime
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset.cad_dataset import get_dataloader, CADDataset
from model import DeepCADMLA
from config.defaults import DatasetConfig, TrainingConfig


def save_config(config, save_path):
    """
    保存配置信息到 JSON 文件
    """
    with open(save_path, "w") as f:
        json.dump(config, f, indent=4)


def main():
    dataset_cfg = DatasetConfig()
    train_cfg = TrainingConfig()

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

    # 获取数据加载器
    train_loader = get_dataloader(phase="train", config=dataset_cfg, shuffle=True)
    val_loader = get_dataloader(phase="validation", config=dataset_cfg, shuffle=False)

    model = DeepCADMLA(
        vocab_size=dataset_cfg.vocab_size,
        d_model=dataset_cfg.d_model,
        num_heads=dataset_cfg.num_heads,
        num_layers=dataset_cfg.num_layers,
        d_latent=dataset_cfg.d_latent
    ).to(train_cfg.device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.learning_rate)

    # 创建日志目录
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join("logs", current_time)
    os.makedirs(log_dir, exist_ok=True)

    # 初始化 TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(log_dir, "tensorboard"))

    # 配置日志文件
    log_file = os.path.join(log_dir, "training.log")
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format="%(asctime)s - %(message)s")

    # 保存配置
    config_data = {
        "DatasetConfig": dataset_cfg.__dict__,
        "TrainingConfig": train_cfg.__dict__
    }
    save_config(config_data, os.path.join(log_dir, "config.json"))  # 保存配置信息

    # 保存训练指标到 CSV 文件
    csv_file = os.path.join(log_dir, "training_metrics.csv")
    with open(csv_file, "w", newline="") as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(["Epoch", "Train Loss", "Val Loss"])

    print(f"Log files will be saved in: {log_dir}")

    scaler = GradScaler()

    for epoch in range(train_cfg.epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch} Training"):
            command = batch["command"].to(train_cfg.device)  # (batch_size, seq_len)
            args = batch["args"].to(train_cfg.device)  # (batch_size, seq_len, 16), torch.long

            # 调试：打印输入形状和类型
            print(f"Command shape: {command.shape}, dtype: {command.dtype}")
            print(f"Args shape: {args.shape}, dtype: {args.dtype}")

            # 将 args 转换为浮点类型并映射到词汇表
            args_float = args.float()  # 从 torch.long 转换为 torch.float
            args_to_vocab = nn.Linear(args.shape[-1], dataset_cfg.vocab_size).to(train_cfg.device)
            args_logits = args_to_vocab(args_float)  # (batch_size, seq_len, vocab_size)
            args_indices = torch.argmax(args_logits, dim=-1)  # (batch_size, seq_len)

            input_seq = command
            target_seq = command

            # 使用自动混合精度
            with autocast():  # 自动混合精度
                outputs = model(input_seq, target_seq)  # (batch_size, seq_len-1, vocab_size)
                loss = criterion(outputs.view(-1, dataset_cfg.vocab_size), target_seq[:, 1:].contiguous().view(-1))
                loss = loss.mean()

            scaler.scale(loss).backward()  # 缩放损失并反向传播
            scaler.step(optimizer)  # 更新梯度
            scaler.update()  # 更新缩放器

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        logging.info(f"Epoch {epoch}, Train Loss: {avg_train_loss}")
        writer.add_scalar("Loss/train", avg_train_loss, epoch)

        # 保存到 CSV
        with open(csv_file, "a", newline="") as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow([epoch, avg_train_loss])

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} Validation"):
                command = batch["command"].to(train_cfg.device)
                args = batch["args"].to(train_cfg.device)

                args_float = args.float()
                args_to_vocab = nn.Linear(args.shape[-1], dataset_cfg.vocab_size).to(train_cfg.device)
                args_logits = args_to_vocab(args_float)
                args_indices = torch.argmax(args_logits, dim=-1)

                input_seq = command
                target_seq = command

                outputs = model(input_seq, target_seq)
                loss = criterion(outputs.view(-1, dataset_cfg.vocab_size), target_seq[:, 1:].contiguous().view(-1))
                loss = loss.mean()
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        logging.info(f"Epoch {epoch}, Val Loss: {avg_val_loss}")
        writer.add_scalar("Loss/validation", avg_val_loss, epoch)

        # 保存到 CSV
        with open(csv_file, "a", newline="") as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow([epoch, None, avg_val_loss])

        # 保存模型权重
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(log_dir, f"model_epoch_{epoch}.pth"))

    # 保存最终模型权重
    torch.save(model.state_dict(), os.path.join(log_dir, "model_final.pth"))

    # 保存训练指标到 JSON
    with open(os.path.join(log_dir, "training_metrics.json"), "w") as f:
        json.dump({
            "train_loss": [avg_train_loss],
            "val_loss": [avg_val_loss]
        }, f, indent=4)

    print(f"Training completed. Results saved in: {log_dir}")


if __name__ == '__main__':
    main()
