import os
import sys
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset.cad_dataset import get_dataloader, CADDataset
from model import DeepCADMLA
from config.defaults import DatasetConfig, TrainingConfig
from tqdm import tqdm


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

    train_loader = get_dataloader(phase="train", config=dataset_cfg, shuffle=True)
    val_loader = get_dataloader(phase="validation", config=dataset_cfg, shuffle=False)

    model = DeepCADMLA(
        vocab_size=dataset_cfg.vocab_size,
        d_model=dataset_cfg.d_model,
        num_heads=dataset_cfg.num_heads,
        num_layers=dataset_cfg.num_layers,
        d_latent=dataset_cfg.d_latent
    ).to(train_cfg.device)

    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.learning_rate)

    for epoch in range(train_cfg.epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch} Training"):
            command = batch["command"].to(train_cfg.device)  # (batch_size, seq_len)
            args = batch["args"].to(train_cfg.device)  # (batch_size, seq_len, 16), torch.long

            # 调试：打印输入形状和类型
            print(f"Command shape: {command.shape}, dtype: {command.dtype}")
            print(f"Args shape: {args.shape}, dtype: {args.dtype}")

            # 将 args 转换为浮点类型
            args_float = args.float()  # 从 torch.long 转换为 torch.float
            args_to_vocab = nn.Linear(args.shape[-1], dataset_cfg.vocab_size).to(train_cfg.device)
            args_logits = args_to_vocab(args_float)  # (batch_size, seq_len, vocab_size)
            args_indices = torch.argmax(args_logits, dim=-1)  # (batch_size, seq_len)

            input_seq = command
            target_seq = command

            outputs = model(input_seq, target_seq)  # (batch_size, seq_len-1, vocab_size)
            loss = criterion(outputs.view(-1, dataset_cfg.vocab_size), target_seq[:, 1:].contiguous().view(-1))
            loss = loss.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}, Train Loss: {avg_train_loss}")

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
        print(f"Epoch {epoch}, Val Loss: {avg_val_loss}")


if __name__ == '__main__':
    main()