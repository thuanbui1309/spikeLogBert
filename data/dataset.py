"""
Dataset classes for log parsing.

Data format (train.txt / test.txt):
    <raw log message>\t<template_id>
    Receiving block blk_-123 src: /10.0.0.1 dest: /10.0.0.2\t5
    PacketResponder 1 for block blk_-456 terminating\t11
"""

import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


def _load_samples(data_path: str):
    """Load tab-separated (message, label) pairs from file."""
    samples = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 2:
                samples.append((parts[0], int(parts[1])))
    return samples


class LogParsingDataset(Dataset):
    """
    Dataset for log parsing as text classification.
    Returns raw (message_string, label_int) pairs.
    """

    def __init__(self, data_path: str):
        super().__init__()
        self.samples = _load_samples(data_path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        message, label = self.samples[index]
        return message, label


class TokenizedLogParsingDataset(Dataset):
    """
    Pre-tokenized dataset — tokenizes ALL messages once at init time.
    Returns (input_ids, attention_mask, label) tensors directly.

    This eliminates the CPU tokenizer bottleneck from the training loop.
    """

    def __init__(self, data_path: str, tokenizer, max_length: int = 128):
        super().__init__()
        samples = _load_samples(data_path)
        messages = [s[0] for s in samples]
        self.labels = torch.tensor([s[1] for s in samples], dtype=torch.long)

        print(f"Pre-tokenizing {len(messages)} messages (max_length={max_length})...")
        encoded = tokenizer(
            messages, padding="max_length", truncation=True,
            return_tensors="pt", max_length=max_length
        )
        self.input_ids = encoded["input_ids"]       # (N, L)
        self.attention_mask = encoded["attention_mask"]  # (N, L)
        print(f"  Done. Shape: {self.input_ids.shape}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.input_ids[index], self.attention_mask[index], self.labels[index]


def create_log_parsing_data(
    structured_csv_path: str,
    output_dir: str,
    content_col: str = "Content",
    event_col: str = "EventId",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    max_samples: int = None,
):
    """
    Create train/val/test splits from Drain-parsed structured CSV.

    The structured CSV is expected to have at least:
        - Content: raw log message
        - EventId: template identifier (string or int)

    Output format (tab-separated):
        <log message>\t<template_id_int>

    Args:
        structured_csv_path: path to *_structured.csv from Drain parser
        output_dir: directory to write train.txt, val.txt, test.txt
        content_col: column name for raw log message
        event_col: column name for event/template ID
        train_ratio: fraction for training
        val_ratio: fraction for validation
        test_ratio: fraction for testing
        seed: random seed
        max_samples: if set, subsample to this many rows
    """
    print(f"Loading structured CSV: {structured_csv_path}")
    df = pd.read_csv(structured_csv_path, engine='c', na_filter=False, memory_map=True)

    print(f"Total log messages: {len(df)}")
    print(f"Unique templates: {df[event_col].nunique()}")

    # Subsample if needed (large datasets)
    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=seed)
        print(f"Subsampled to {max_samples} messages")

    # Create integer label mapping
    unique_events = sorted(df[event_col].unique())
    event_to_id = {event: idx for idx, event in enumerate(unique_events)}
    df['label'] = df[event_col].map(event_to_id)

    print(f"\nTemplate mapping:")
    for event, idx in event_to_id.items():
        count = (df[event_col] == event).sum()
        print(f"  {event} → {idx} ({count:,} samples)")

    # Split: train / val / test
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    # Handle rare classes: move samples with few occurrences to train directly
    # Threshold: need at least ceil(1/test_ratio) * 2 samples for stratified 2-split
    min_samples = max(10, int(2 / min(val_ratio, test_ratio)))
    label_counts = df['label'].value_counts()
    rare_labels = label_counts[label_counts < min_samples].index.tolist()
    if rare_labels:
        print(f"\nRare classes (< {min_samples} samples, forced to train): {rare_labels}")
        rare_df = df[df['label'].isin(rare_labels)]
        main_df = df[~df['label'].isin(rare_labels)]
    else:
        rare_df = pd.DataFrame()
        main_df = df

    train_df, temp_df = train_test_split(
        main_df, train_size=train_ratio, random_state=seed, stratify=main_df['label']
    )
    relative_val = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df, train_size=relative_val, random_state=seed, stratify=temp_df['label']
    )

    # Add rare samples to train
    if not rare_df.empty:
        train_df = pd.concat([train_df, rare_df], ignore_index=True)

    print(f"\nSplit sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    _save_split(train_df, content_col, os.path.join(output_dir, 'train.txt'))
    _save_split(val_df, content_col, os.path.join(output_dir, 'val.txt'))
    _save_split(test_df, content_col, os.path.join(output_dir, 'test.txt'))

    # Save label mapping
    mapping_path = os.path.join(output_dir, 'label_mapping.txt')
    with open(mapping_path, 'w') as f:
        for event, idx in event_to_id.items():
            f.write(f"{event}\t{idx}\n")
    print(f"Saved label mapping to {mapping_path}")

    return event_to_id


def _save_split(df, content_col, output_path):
    """Save a data split as tab-separated file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            message = str(row[content_col]).replace('\t', ' ').replace('\n', ' ')
            label = int(row['label'])
            f.write(f"{message}\t{label}\n")
    print(f"Saved {len(df)} samples to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create log parsing dataset from Drain output")
    parser.add_argument("--input", required=True, help="Path to *_structured.csv")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--content_col", default="Content", help="Column name for log message")
    parser.add_argument("--event_col", default="EventId", help="Column name for template ID")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    create_log_parsing_data(
        structured_csv_path=args.input,
        output_dir=args.output,
        content_col=args.content_col,
        event_col=args.event_col,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        max_samples=args.max_samples,
        seed=args.seed,
    )
