"""
Evaluation script for log parsing accuracy.

Metrics:
    - Parsing Accuracy (PA): % of log messages correctly assigned template
    - Per-class precision, recall, F1
    - Confusion matrix (optional)

Usage:
    python evaluate.py \
        --model_path saved_models/distilled/best.pth \
        --dataset_dir data/hdfs \
        --label_num 29
"""

import os
import argparse
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer
from sklearn.metrics import classification_report, accuracy_score
from spikingjelly.activation_based import functional

from model import SpikeLogBERT
from data.dataset import TokenizedLogParsingDataset
from utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SpikeLogBERT")

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--label_num", type=int, required=True)

    # Model config (must match saved model)
    parser.add_argument("--depths", type=int, default=6)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--num_step", type=int, default=4)
    parser.add_argument("--tau", type=float, default=10.0)
    parser.add_argument("--common_thr", type=float, default=1.0)
    parser.add_argument("--tokenizer_path", type=str, default="bert-base-cased")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--split", type=str, default="test", choices=["test", "val", "train"])
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)

    return parser.parse_args()


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path)

    # Load model
    model = SpikeLogBERT(
        depths=args.depths, length=args.max_length, T=args.num_step,
        tau=args.tau, common_thr=args.common_thr,
        vocab_size=len(tokenizer), dim=args.dim,
        num_classes=args.label_num, mode="train",
    )
    state_dict = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    print(f"Loaded model from {args.model_path}")

    # Load pre-tokenized data
    split_file = os.path.join(args.dataset_dir, f"{args.split}.txt")
    dataset = TokenizedLogParsingDataset(split_file, tokenizer, args.max_length)
    data_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    print(f"Evaluating on {args.split} split: {len(dataset)} samples")

    # Load label mapping (if exists)
    label_names = None
    mapping_path = os.path.join(args.dataset_dir, "label_mapping.txt")
    if os.path.exists(mapping_path):
        label_names = {}
        with open(mapping_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    label_names[int(parts[1])] = parts[0]

    # Inference
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)

            _, outputs = model(input_ids)
            logits = torch.mean(outputs, dim=1)  # B C

            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())

            functional.reset_net(model)

    # ---- Metrics ----
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Parsing Accuracy (PA)
    pa = accuracy_score(all_labels, all_preds)

    # Classification report
    target_names = None
    if label_names:
        unique_labels = sorted(set(all_labels) | set(all_preds))
        target_names = [label_names.get(i, f"class_{i}") for i in unique_labels]

    report = classification_report(
        all_labels, all_preds,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    report_str = classification_report(
        all_labels, all_preds,
        target_names=target_names,
        zero_division=0,
    )

    print(f"\n{'='*60}")
    print(f"Parsing Accuracy (PA): {pa:.4f} ({pa*100:.2f}%)")
    print(f"{'='*60}")
    print(f"\nClassification Report:")
    print(report_str)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results = {
        "model_path": args.model_path,
        "dataset_dir": args.dataset_dir,
        "split": args.split,
        "num_samples": len(dataset),
        "parsing_accuracy": pa,
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "classification_report": report,
    }

    results_path = os.path.join(args.output_dir, f"eval_{args.split}.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return results


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    evaluate(args)
