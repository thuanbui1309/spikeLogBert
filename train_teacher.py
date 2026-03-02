"""
Stage 1: Fine-tune BERT teacher model on log parsing (multi-class classification).

The fine-tuned BERT serves as the teacher for knowledge distillation into SpikeLogBERT.

Usage:
    python train_teacher.py --dataset_dir data/hdfs --label_num 29 --epochs 4
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification

from data.dataset import TokenizedLogParsingDataset
from utils import set_seed, check_and_create_path


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune BERT for log parsing")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Directory with train.txt/val.txt/test.txt")
    parser.add_argument("--teacher_model_name", type=str, default="bert-base-cased")
    parser.add_argument("--label_num", type=int, required=True, help="Number of log template classes")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--save_dir", type=str, default="saved_models/teacher")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    return parser.parse_args()


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(args.teacher_model_name)
    model = BertForSequenceClassification.from_pretrained(
        args.teacher_model_name, num_labels=args.label_num
    )

    # Multi-GPU support
    device_ids = list(range(torch.cuda.device_count()))
    if len(device_ids) > 1:
        model = nn.DataParallel(model, device_ids=device_ids)
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr)

    # Load pre-tokenized datasets
    train_dataset = TokenizedLogParsingDataset(
        os.path.join(args.dataset_dir, "train.txt"), tokenizer, args.max_length
    )
    val_dataset = TokenizedLogParsingDataset(
        os.path.join(args.dataset_dir, "val.txt"), tokenizer, args.max_length
    )
    test_dataset = TokenizedLogParsingDataset(
        os.path.join(args.dataset_dir, "test.txt"), tokenizer, args.max_length
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    best_acc = 0.0
    for epoch in range(args.epochs):
        # ---- Train ----
        model.train()
        train_losses = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]"):
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = F.cross_entropy(outputs.logits, labels)
            train_losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_train_loss = sum(train_losses) / len(train_losses)

        # ---- Evaluate ----
        val_acc = _evaluate(model, val_loader, device)
        test_acc = _evaluate(model, test_loader, device)

        print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, val_acc={val_acc:.4f}, test_acc={test_acc:.4f}")

        # Save best model
        if val_acc >= best_acc:
            best_acc = val_acc
            save_path = os.path.join(args.save_dir, f"best_teacher_{args.label_num}classes")
            check_and_create_path(save_path)

            if len(device_ids) > 1:
                model.module.save_pretrained(save_path)
            else:
                model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"  → Saved best model (val_acc={best_acc:.4f}) to {save_path}")

    print(f"\nBest validation accuracy: {best_acc:.4f}")
    return save_path


def _evaluate(model, data_loader, device):
    """Evaluate classification accuracy (pre-tokenized data)."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in data_loader:
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += len(labels)
    model.train()
    return correct / total if total > 0 else 0.0


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    train(args)
