# SpikeLogBERT

Spiking Neural Network-based log parser using Spikformer architecture.  
Replaces traditional rule-based parsers (Drain, Spell) with an energy-efficient SNN classifier.

## Pipeline

```
Raw Log Message → BERTTokenizer → Spikformer (SNN) → Template ID → Anomaly Detection (downstream)
```

## Quick Start

### 1. Setup Environment

```bash
pip install torch transformers spikingjelly scikit-learn pandas tqdm pyyaml
```

### 2. Download Data

```bash
# Download HDFS 2k benchmark dataset
python data/download.py --dataset HDFS --output data/raw

# Preprocess into train/val/test splits
python data/dataset.py \
    --input data/raw/HDFS_2k.log_structured.csv \
    --output data/hdfs
```

### 3. Train

**Option A: Knowledge Distillation (recommended)**

```bash
# Stage 1: Fine-tune BERT teacher
python train_teacher.py \
    --dataset_dir data/hdfs \
    --label_num 29 \
    --epochs 4

# Stage 2: Distill into SpikeLogBERT
python distill.py \
    --dataset_dir data/hdfs \
    --teacher_model_path saved_models/teacher/best_teacher_29classes \
    --label_num 29
```

**Option B: Direct Training**

```bash
python train_direct.py --dataset_dir data/hdfs --label_num 29
```

### 4. Evaluate

```bash
python evaluate.py \
    --model_path saved_models/distilled/hdfs/best.pth \
    --dataset_dir data/hdfs \
    --label_num 29
```

## Project Structure

```
spikeLogBert/
├── model/
│   ├── __init__.py
│   └── spikformer.py          # SpikeLogBERT model (Spikformer architecture)
├── data/
│   ├── download.py             # Download LogHub datasets
│   └── dataset.py              # Dataset classes & preprocessing
├── configs/
│   ├── hdfs.yaml               # HDFS configuration
│   ├── bgl.yaml                # BGL configuration (placeholder)
│   └── thunderbird.yaml        # Thunderbird configuration (placeholder)
├── train_teacher.py            # Stage 1: Fine-tune BERT teacher
├── distill.py                  # Stage 2: Knowledge distillation
├── train_direct.py             # Alternative: direct training
├── evaluate.py                 # Evaluation script
├── utils.py                    # Utilities
└── README.md
```

## Architecture

Based on SpikeBERT's Spikformer with LIF (Leaky Integrate-and-Fire) neurons.

| Component | Detail |
|-----------|--------|
| Backbone | Spikformer (Spiking Transformer) |
| Neuron Model | LIF with ATan surrogate gradient |
| Embedding | nn.Embedding (shared with BERT tokenizer) |
| SNN Timesteps | 16 (configurable) |
| Model Dim | 768 |
| Layers | 6 |

## References

- **LogBERT**: Guo et al., "LogBERT: Log Anomaly Detection via BERT", ASE 2021
- **SpikeBERT**: Lv et al., "SpikeBERT: A Language Spikformer Trained with Two-Stage Knowledge Distillation", ACL 2024 Findings
- **LogHub-2.0**: https://github.com/logpai/loghub-2.0
