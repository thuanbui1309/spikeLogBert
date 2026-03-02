"""
Stage 2: Knowledge Distillation — Distill fine-tuned BERT into SpikeLogBERT (Spikformer).

Losses:
    1. Embedding loss: MSE(student.emb, teacher.emb)
    2. Representation loss: MSE(student hidden states, teacher hidden states)
    3. Logit loss: KL-div(student logits, teacher logits)
    4. CE loss: CrossEntropy(student logits, labels)

Usage:
    python distill.py \
        --dataset_dir data/hdfs \
        --teacher_model_path saved_models/teacher/best_teacher_29classes \
        --label_num 29
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
from transformers import BertTokenizer, BertForSequenceClassification
from spikingjelly.activation_based import functional

from model import SpikeLogBERT
from data.dataset import TokenizedLogParsingDataset
from utils import set_seed, check_and_create_path


def parse_args():
    parser = argparse.ArgumentParser(description="Distill BERT into SpikeLogBERT")

    # Data
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--label_num", type=int, required=True)

    # Teacher
    parser.add_argument("--teacher_model_path", type=str, required=True)

    # Student architecture
    parser.add_argument("--depths", type=int, default=6)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--num_step", type=int, default=4, help="SNN timesteps (T)")
    parser.add_argument("--tau", type=float, default=10.0)
    parser.add_argument("--common_thr", type=float, default=1.0)

    # Pre-distilled weights (optional)
    parser.add_argument("--predistill_model_path", type=str, default="")

    # Training
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)

    # Loss weights
    parser.add_argument("--ce_weight", type=float, default=0.1)
    parser.add_argument("--emb_weight", type=float, default=0.1)
    parser.add_argument("--logit_weight", type=float, default=1.0)
    parser.add_argument("--rep_weight", type=float, default=0.1)
    parser.add_argument("--ignored_layers", type=int, default=0)

    # Save
    parser.add_argument("--save_dir", type=str, default="saved_models/distilled")

    return parser.parse_args()


def distill(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Teacher ----
    tokenizer = BertTokenizer.from_pretrained(args.teacher_model_path)
    teacher_model = BertForSequenceClassification.from_pretrained(
        args.teacher_model_path, num_labels=args.label_num, output_hidden_states=True
    ).to(device)
    for param in teacher_model.parameters():
        param.requires_grad = False
    teacher_model.eval()
    print(f"Loaded teacher from {args.teacher_model_path}")

    # ---- Student ----
    student_model = SpikeLogBERT(
        depths=args.depths, length=args.max_length, T=args.num_step,
        tau=args.tau, common_thr=args.common_thr,
        vocab_size=len(tokenizer), dim=args.dim,
        num_classes=args.label_num, mode="distill",
    )

    if args.predistill_model_path:
        weights = torch.load(args.predistill_model_path, map_location='cpu')
        student_model.load_state_dict(weights, strict=False)
        print(f"Loaded pre-distilled weights from {args.predistill_model_path}")

    # Multi-GPU
    device_ids = list(range(torch.cuda.device_count()))
    if len(device_ids) > 1:
        student_model = nn.DataParallel(student_model, device_ids=device_ids)
    student_model = student_model.to(device)

    # ---- Optimizer ----
    scaler = torch.amp.GradScaler('cuda')
    optimizer = optim.AdamW(student_model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=5e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)

    # ---- Data (pre-tokenized) ----
    train_dataset = TokenizedLogParsingDataset(
        os.path.join(args.dataset_dir, "train.txt"), tokenizer, args.max_length
    )
    test_dataset = TokenizedLogParsingDataset(
        os.path.join(args.dataset_dir, "test.txt"), tokenizer, args.max_length
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False
    )

    print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")

    # ---- Training loop ----
    best_acc = 0.0

    for epoch in range(args.epochs):
        student_model.train()
        loss_logs = {"total": [], "ce": [], "emb": [], "logit": [], "rep": []}

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            batch_size = input_ids.size(0)

            # Teacher forward (no tokenization needed — already have input_ids)
            with torch.no_grad():
                teacher_outputs = teacher_model(
                    input_ids=input_ids, attention_mask=attention_mask
                )

            # Embedding loss
            tea_embeddings = teacher_model.bert.embeddings.word_embeddings.weight
            if len(device_ids) > 1:
                stu_embeddings = student_model.module.emb.weight
            else:
                stu_embeddings = student_model.emb.weight
            emb_loss = F.mse_loss(stu_embeddings, tea_embeddings)

            # Student forward
            tea_rep = teacher_outputs.hidden_states[1:][::int(12 / args.depths)]
            stu_rep, student_outputs = student_model(input_ids)

            # Reshape student outputs: (B, T, C) → mean over T
            student_logits = torch.mean(student_outputs, dim=1)  # B C

            # CE loss
            ce_loss = F.cross_entropy(student_logits, labels)

            # Logit distillation loss (KL divergence)
            logit_loss = F.kl_div(
                F.log_softmax(student_logits, dim=1),
                F.softmax(teacher_outputs.logits, dim=1),
                reduction='batchmean'
            )

            # Representation distillation loss (torch.stack on GPU, no numpy!)
            tea_rep_tensor = torch.stack(list(tea_rep))  # (num_layers, B, L, D)

            rep_loss = 0
            tea_rep_subset = tea_rep_tensor[args.ignored_layers:]
            stu_rep_subset = stu_rep[args.ignored_layers:]
            for i in range(len(stu_rep_subset)):
                rep_loss += F.mse_loss(stu_rep_subset[i], tea_rep_subset[i])
            rep_loss = rep_loss / batch_size

            # Total loss
            total_loss = (
                args.emb_weight * emb_loss
                + args.ce_weight * ce_loss
                + args.logit_weight * logit_loss
                + args.rep_weight * rep_loss
            )

            # Backward
            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            functional.reset_net(student_model)

            # Log
            loss_logs["total"].append(total_loss.item())
            loss_logs["ce"].append(ce_loss.item())
            loss_logs["emb"].append(emb_loss.item())
            loss_logs["logit"].append(logit_loss.item())
            loss_logs["rep"].append(rep_loss.item() if isinstance(rep_loss, torch.Tensor) else rep_loss)

        scheduler.step()

        # ---- Evaluate ----
        acc = _evaluate_snn(student_model, test_loader, args, device)
        print(
            f"Epoch {epoch+1}: "
            f"total={np.mean(loss_logs['total']):.4f}, "
            f"ce={np.mean(loss_logs['ce']):.4f}, "
            f"emb={np.mean(loss_logs['emb']):.4f}, "
            f"logit={np.mean(loss_logs['logit']):.4f}, "
            f"rep={np.mean(loss_logs['rep']):.4f}, "
            f"test_acc={acc:.4f}"
        )

        if acc >= best_acc:
            best_acc = acc
            save_path = os.path.join(
                args.save_dir,
                f"spikelogbert_epoch{epoch+1}_acc{acc:.4f}_T{args.num_step}.pth"
            )
            check_and_create_path(args.save_dir)
            torch.save(student_model.state_dict(), save_path)
            print(f"  → Saved best model to {save_path}")

    print(f"\nBest test accuracy: {best_acc:.4f}")


def _evaluate_snn(model, data_loader, args, device):
    """Evaluate SNN model accuracy (pre-tokenized data)."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in data_loader:
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)

            _, outputs = model(input_ids)
            logits = torch.mean(outputs, dim=1)  # B C

            preds = torch.argmax(logits, dim=-1)
            correct += (preds == labels.to(device)).sum().item()
            total += len(labels)

            functional.reset_net(model)

    return correct / total if total > 0 else 0.0


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    distill(args)
