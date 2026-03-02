#!/bin/bash
#
# SpikeLogBERT — HDFS Full Training Pipeline
# RTX 4090 24GB optimized
#
# Usage:
#   cd spikeLogBert
#   bash train/hdfs/run.sh
#
set -euo pipefail

# ============================================================
# Config
# ============================================================
DATASET="HDFS"
DATASET_DIR="data/hdfs"
RAW_DIR="data/raw"
RESULTS_DIR="results/hdfs"
TEACHER_EPOCHS=4
TEACHER_BATCH=64
DISTILL_EPOCHS=30
DISTILL_BATCH=32
DISTILL_LR="5e-4"
NUM_STEP=4
DEPTHS=6
DIM=768

# Set HuggingFace cache local to avoid permission issues
export HF_HOME=./.hf_cache

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

echo "============================================"
echo " SpikeLogBERT — HDFS Training Pipeline"
echo " Working dir: $PROJECT_DIR"
echo " Started: $(date)"
echo "============================================"

# ============================================================
# 1. Environment check
# ============================================================
echo ""
echo "[1/7] Checking environment..."

# Check Python
if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found"
    exit 1
fi
echo "  Python: $(python3 --version)"

# Check/install uv
if ! command -v uv &>/dev/null; then
    echo "  uv not found, installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "  uv: $(uv --version)"

# Sync dependencies
echo "  Installing dependencies..."
uv sync --quiet
echo "  ✅ Dependencies ready"

# Check GPU
uv run python3 -c "
import torch
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f'  GPU: {name} ({vram:.1f} GB)')
else:
    print('  ⚠️  No GPU detected — training will be slow')
"

# ============================================================
# 2. Download data
# ============================================================
echo ""
echo "[2/7] Downloading HDFS dataset..."

if [ -f "$RAW_DIR/HDFS_2k.log_structured.csv" ]; then
    echo "  ✅ HDFS data already exists"
else
    uv run python3 data/download.py --dataset HDFS --output "$RAW_DIR"
fi

# Detect full vs 2k
if [ -f "$RAW_DIR/HDFS.log_structured.csv" ]; then
    INPUT_CSV="$RAW_DIR/HDFS.log_structured.csv"
    echo "  Using FULL HDFS dataset"
else
    INPUT_CSV="$RAW_DIR/HDFS_2k.log_structured.csv"
    echo "  Using HDFS 2k (benchmark subset)"
fi

# ============================================================
# 3. Preprocess
# ============================================================
echo ""
echo "[3/7] Preprocessing data..."

if [ -f "$DATASET_DIR/train.txt" ] && [ -f "$DATASET_DIR/label_mapping.txt" ]; then
    echo "  ✅ Preprocessed data already exists"
else
    uv run python3 data/dataset.py \
        --input "$INPUT_CSV" \
        --output "$DATASET_DIR"
fi

# Get number of labels
NUM_LABELS=$(wc -l < "$DATASET_DIR/label_mapping.txt")
echo "  Templates: $NUM_LABELS"
echo "  Train: $(wc -l < "$DATASET_DIR/train.txt") samples"
echo "  Val:   $(wc -l < "$DATASET_DIR/val.txt") samples"
echo "  Test:  $(wc -l < "$DATASET_DIR/test.txt") samples"

# ============================================================
# 4. Stage 1: Train BERT Teacher
# ============================================================
TEACHER_PATH="saved_models/teacher/best_teacher_${NUM_LABELS}classes"

echo ""
echo "[4/7] Stage 1: Training BERT teacher..."

if [ -d "$TEACHER_PATH" ]; then
    echo "  ✅ Teacher model already exists at $TEACHER_PATH"
else
    uv run python3 train_teacher.py \
        --dataset_dir "$DATASET_DIR" \
        --label_num "$NUM_LABELS" \
        --epochs "$TEACHER_EPOCHS" \
        --batch_size "$TEACHER_BATCH" \
        --lr 5e-5
fi

# ============================================================
# 5. Stage 2: Knowledge Distillation
# ============================================================
echo ""
echo "[5/7] Stage 2: Knowledge distillation (BERT → SpikeLogBERT)..."

uv run python3 distill.py \
    --dataset_dir "$DATASET_DIR" \
    --teacher_model_path "$TEACHER_PATH" \
    --label_num "$NUM_LABELS" \
    --depths "$DEPTHS" \
    --dim "$DIM" \
    --num_step "$NUM_STEP" \
    --epochs "$DISTILL_EPOCHS" \
    --batch_size "$DISTILL_BATCH" \
    --lr "$DISTILL_LR" \
    --save_dir "saved_models/distilled/hdfs"

# ============================================================
# 6. Find best model
# ============================================================
echo ""
echo "[6/7] Finding best distilled model..."

BEST_MODEL=$(ls -t saved_models/distilled/hdfs/spikelogbert_*.pth 2>/dev/null | head -1)
if [ -z "$BEST_MODEL" ]; then
    echo "  ERROR: No distilled model found!"
    exit 1
fi
echo "  Best model: $BEST_MODEL"

# ============================================================
# 7. Evaluate
# ============================================================
echo ""
echo "[7/7] Stage 3: Evaluating..."

mkdir -p "$RESULTS_DIR"

uv run python3 evaluate.py \
    --model_path "$BEST_MODEL" \
    --dataset_dir "$DATASET_DIR" \
    --label_num "$NUM_LABELS" \
    --num_step "$NUM_STEP" \
    --output_dir "$RESULTS_DIR" \
    --batch_size 64

# ============================================================
# Summary
# ============================================================
echo ""
echo "============================================"
echo " Training Complete!"
echo " Finished: $(date)"
echo "============================================"
echo ""
echo "Results:"
uv run python3 -c "
import json
with open('$RESULTS_DIR/eval_test.json') as f:
    r = json.load(f)
print(f'  Parsing Accuracy: {r[\"parsing_accuracy\"]:.4f} ({r[\"parsing_accuracy\"]*100:.2f}%)')
print(f'  Macro F1:         {r[\"macro_f1\"]:.4f}')
print(f'  Weighted F1:      {r[\"weighted_f1\"]:.4f}')
"
echo ""
echo "Saved:"
echo "  Model:   $BEST_MODEL"
echo "  Results: $RESULTS_DIR/eval_test.json"
