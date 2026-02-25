"""
Utility functions for SpikeLogBERT
"""

import os
import random
import logging
import torch
import numpy as np


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    if seed is not None:
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def check_and_create_path(path: str):
    """Create directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def save_model(path, model):
    """Save model state dict to file."""
    check_and_create_path(os.path.dirname(path))
    torch.save(model.state_dict(), path)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Saved model to {path}")
    print(f"  Total parameters: {n_params:,}")
    print(f"  Trainable parameters: {n_trainable:,}")


def load_model(path, model, strict=False):
    """Load model state dict from file."""
    state_dict = torch.load(path, map_location='cpu')
    model.load_state_dict(state_dict, strict=strict)
    print(f"Loaded model from {path}")
    return model


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def to_device(x, device):
    """Move dict of tensors to device."""
    for key in x:
        x[key] = x[key].to(device)
