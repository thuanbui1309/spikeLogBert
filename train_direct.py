"""
Direct training of SpikeLogBERT (without knowledge distillation).

Alternative to the 2-stage distillation pipeline.
Simpler but may achieve lower accuracy.

Usage:
    python train_direct.py --dataset_dir data/hdfs --label_num 29
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer
from spikingjelly.activation_based import functional

from model import SpikeLogBERT
from data.dataset import LogParsingDataset
from utils import set_seed, to_device, check_and_create_path


def parse_args():
    parser = argparse.ArgumentParser(description="Direct training of SpikeLogBERT")

    # Data
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--label_num", type=int, required=True)

    # Model
    parser.add_argument("--depths", type=int, default=6)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--num_step", type=int, default=16)
    parser.add_argument("--tau", type=float, default=10.0)
    parser.add_argument("--common_thr", type=float, default=1.0)
    parser.add_argument("--tokenizer_path", type=str, default="bert-base-cased")

    # Training
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)

    # Save
    parser.add_argument("--save_dir", type=str, default="saved_models/direct")

    return parser.parse_args()


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path)

    model = SpikeLogBERT(
        depths=args.depths, length=args.max_length, T=args.num_step,
        tau=args.tau, common_thr=args.common_thr,
        vocab_size=len(tokenizer), dim=args.dim,
        num_classes=args.label_num, mode="train",
    )

    # Multi-GPU
    device_ids = list(range(torch.cuda.device_count()))
    if len(device_ids) > 1:
        model = nn.DataParallel(model, device_ids=device_ids)
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    scaler = torch.cuda.amp.GradScaler()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=5e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)

    # Data
    train_dataset = LogParsingDataset(os.path.join(args.dataset_dir, "train.txt"))
    test_dataset = LogParsingDataset(os.path.join(args.dataset_dir, "test.txt"))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")

    # Training loop
    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        losses = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            messages, labels = batch
            labels = labels.to(device)

            inputs = tokenizer(
                list(messages), padding="max_length", truncation=True,
                return_tensors="pt", max_length=args.max_length
            )
            to_device(inputs, device)

            _, outputs = model(inputs['input_ids'])

            # outputs: could be (B*T, C) or (B, T, C) depending on DataParallel
            outputs = outputs.reshape(-1, args.num_step, args.label_num)
            outputs = outputs.transpose(0, 1)  # T B C
            logits = torch.mean(outputs, dim=0)  # B C

            loss = F.cross_entropy(logits, labels)
            losses.append(loss.item())

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            functional.reset_net(model)

        scheduler.step()

        # Evaluate
        acc = _evaluate(model, test_loader, tokenizer, args, device)
        print(f"Epoch {epoch+1}: avg_loss={np.mean(losses):.4f}, test_acc={acc:.4f}")

        if acc >= best_acc:
            best_acc = acc
            save_path = os.path.join(
                args.save_dir,
                f"spikelogbert_direct_epoch{epoch+1}_acc{acc:.4f}.pth"
            )
            check_and_create_path(args.save_dir)
            torch.save(model.state_dict(), save_path)
            print(f"  → Saved best model to {save_path}")

    print(f"\nBest test accuracy: {best_acc:.4f}")


def _evaluate(model, data_loader, tokenizer, args, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in data_loader:
            messages, labels = batch
            inputs = tokenizer(
                list(messages), padding="max_length", truncation=True,
                return_tensors="pt", max_length=args.max_length
            )
            to_device(inputs, device)

            _, outputs = model(inputs['input_ids'])
            outputs = outputs.to("cpu")
            outputs = outputs.reshape(-1, args.num_step, args.label_num)
            outputs = outputs.transpose(0, 1)
            logits = torch.mean(outputs, dim=0)

            preds = torch.argmax(logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += len(labels)
            functional.reset_net(model)

    return correct / total if total > 0 else 0.0


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    train(args)
