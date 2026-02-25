"""
SpikeLogBERT - Spikformer model for log parsing
Adapted from SpikeBERT's new_spikformer.py

Architecture:
    Input (token IDs) → Embedding → Repeat T timesteps → ATan
    → [Spiking Block × N] → LayerNorm → Mean pooling → Classifier

Each Spiking Block:
    x → spiking_self_attention (+residual) → spiking_mlp (+residual) → out
"""

import torch
import torch.nn as nn
from spikingjelly.activation_based import surrogate, neuron, functional

# Global config
BACKEND = "torch"
DETACH_RESET = True


class SpikingSelfAttention(nn.Module):
    """Spiking Self-Attention with LIF neurons for Q, K, V projections."""

    def __init__(self, length, tau, common_thr, dim, heads=8, qk_scale=0.25):
        super().__init__()
        assert dim % heads == 0, f"dim {dim} should be divided by num_heads {heads}."

        self.dim = dim
        self.heads = heads
        self.qk_scale = qk_scale

        # Q projection
        self.q_m = nn.Linear(dim, dim)
        self.q_ln = nn.LayerNorm(dim)
        self.q_lif = neuron.LIFNode(
            tau=tau, step_mode='m', detach_reset=DETACH_RESET,
            surrogate_function=surrogate.ATan(), v_threshold=common_thr, backend=BACKEND
        )

        # K projection
        self.k_m = nn.Linear(dim, dim)
        self.k_ln = nn.LayerNorm(dim)
        self.k_lif = neuron.LIFNode(
            tau=tau, step_mode='m', detach_reset=DETACH_RESET,
            surrogate_function=surrogate.ATan(), v_threshold=common_thr, backend=BACKEND
        )

        # V projection
        self.v_m = nn.Linear(dim, dim)
        self.v_ln = nn.LayerNorm(dim)
        self.v_lif = neuron.LIFNode(
            tau=tau, step_mode='m', detach_reset=DETACH_RESET,
            surrogate_function=surrogate.ATan(), v_threshold=common_thr, backend=BACKEND
        )

        # Attention output
        self.attn_lif = neuron.LIFNode(
            tau=tau, step_mode='m', detach_reset=DETACH_RESET,
            surrogate_function=surrogate.ATan(), v_threshold=common_thr / 2, backend=BACKEND
        )

        # Output projection
        self.last_m = nn.Linear(dim, dim)
        self.last_ln = nn.LayerNorm(dim)
        self.last_lif = neuron.LIFNode(
            tau=tau, step_mode='m', detach_reset=DETACH_RESET,
            surrogate_function=surrogate.ATan(), v_threshold=common_thr, backend=BACKEND
        )

    def forward(self, x):
        # x: B T L D
        x = x.transpose(0, 1)  # T B L D
        T, B, L, D = x.shape
        x_for_qkv = x.flatten(0, 1)  # TB L D

        # Q
        q_m_out = self.q_ln(self.q_m(x_for_qkv)).reshape(T, B, L, D).contiguous()
        q_m_out = self.q_lif(q_m_out)
        q = q_m_out.reshape(T, B, L, self.heads, D // self.heads).permute(0, 1, 3, 2, 4).contiguous()

        # K
        k_m_out = self.k_ln(self.k_m(x_for_qkv)).reshape(T, B, L, D).contiguous()
        k_m_out = self.k_lif(k_m_out)
        k = k_m_out.reshape(T, B, L, self.heads, D // self.heads).permute(0, 1, 3, 2, 4).contiguous()

        # V
        v_m_out = self.v_ln(self.v_m(x_for_qkv)).reshape(T, B, L, D).contiguous()
        v_m_out = self.v_lif(v_m_out)
        v = v_m_out.reshape(T, B, L, self.heads, D // self.heads).permute(0, 1, 3, 2, 4).contiguous()

        # Attention
        attn = (q @ k.transpose(-2, -1))
        x = (attn @ v) * self.qk_scale  # T B heads L D//heads

        x = x.transpose(2, 3).reshape(T, B, L, D).contiguous()
        x = self.attn_lif(x)

        # Output projection
        x = x.flatten(0, 1)
        x = self.last_ln(self.last_m(x))
        x = self.last_lif(x.reshape(T, B, L, D).contiguous())

        x = x.transpose(0, 1)  # B T L D
        return x


class SpikingMLP(nn.Module):
    """Spiking MLP with LIF neurons and 4x hidden expansion."""

    def __init__(self, length, tau, common_thr, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.ln1 = nn.LayerNorm(hidden_features)
        self.lif1 = neuron.LIFNode(
            tau=tau, step_mode='m', detach_reset=DETACH_RESET,
            surrogate_function=surrogate.ATan(), v_threshold=common_thr, backend=BACKEND
        )

        self.fc2 = nn.Linear(hidden_features, out_features)
        self.ln2 = nn.LayerNorm(out_features)
        self.lif2 = neuron.LIFNode(
            tau=tau, step_mode='m', detach_reset=DETACH_RESET,
            surrogate_function=surrogate.ATan(), v_threshold=common_thr, backend=BACKEND
        )

    def forward(self, x):
        # x: B T L D
        x = x.transpose(0, 1)  # T B L D
        T, B, L, D = x.shape

        x = x.flatten(0, 1)
        x = self.lif1(self.ln1(self.fc1(x)).reshape(T, B, L, -1).contiguous())
        x = x.flatten(0, 1)
        x = self.lif2(self.ln2(self.fc2(x)).reshape(T, B, L, -1).contiguous())

        x = x.transpose(0, 1)  # B T L D
        return x


class SpikingBlock(nn.Module):
    """Single Spiking Transformer block with residual connections."""

    def __init__(self, length, tau, common_thr, dim, heads=8, qk_scale=0.125):
        super().__init__()
        self.attn = SpikingSelfAttention(
            length=length, tau=tau, common_thr=common_thr,
            dim=dim, heads=heads, qk_scale=qk_scale
        )
        self.mlp = SpikingMLP(
            length=length, tau=tau, common_thr=common_thr,
            in_features=dim, hidden_features=dim * 4, out_features=dim
        )

    def forward(self, x):
        # x: B T L D
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class RepresentationTransform(nn.Module):
    """Transform layer for intermediate representations (used in distillation)."""

    def __init__(self, dim, length):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        return self.ln(self.fc(x))


class SpikeLogBERT(nn.Module):
    """
    Spikformer-based log parser.

    Input: token IDs from BERTTokenizer
    Output: (representations, logits)
        - representations: list of per-layer hidden states (for distillation)
        - logits: classification logits (B, T, num_classes) or (B, T, dim) for pre_distill

    Args:
        depths: number of Spiking Transformer blocks
        length: max sequence length
        tau: LIF neuron time constant
        common_thr: LIF neuron firing threshold
        dim: model dimension
        T: number of SNN timesteps
        vocab_size: tokenizer vocabulary size
        num_classes: number of log template classes
        heads: number of attention heads
        qk_scale: attention scale factor
        mode: "train" | "distill" | "pre_distill"
    """

    def __init__(
        self,
        depths=6,
        length=128,
        tau=10.0,
        common_thr=1.0,
        dim=768,
        T=16,
        vocab_size=28996,
        num_classes=29,
        heads=8,
        qk_scale=0.125,
        mode="train",
    ):
        super().__init__()
        self.mode = mode
        self.atan = surrogate.ATan()
        self.T = T

        # Embedding
        self.emb = nn.Embedding(vocab_size, dim)

        # Spiking Transformer blocks
        self.blocks = nn.ModuleList([
            SpikingBlock(
                length=length, tau=tau, common_thr=common_thr,
                dim=dim, heads=heads, qk_scale=qk_scale
            )
            for _ in range(depths)
        ])

        self.last_ln = nn.LayerNorm(dim)

        # Representation transforms (for distillation)
        self.transforms = nn.ModuleList([
            RepresentationTransform(dim, length) for _ in range(depths)
        ])

        # Classifier head
        if mode != "pre_distill":
            self.classifier = nn.Linear(dim, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        """
        Args:
            x: input token IDs, shape (B, L)

        Returns:
            representations: list of per-layer hidden states, each (B, L, D)
            logits: (B, T, num_classes) or (B, T, D) if pre_distill
        """
        # Embedding: B L → B L D
        x = self.emb(x)

        # Repeat T timesteps: B L D → T B L D → B T L D
        x = x.repeat(tuple([self.T] + torch.ones(len(x.size()), dtype=int).tolist()))
        x = x.transpose(0, 1)  # B T L D

        # Apply surrogate gradient activation
        x = self.atan(x)

        # Pass through spiking blocks
        representations = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)  # B T L D
            # Mean over time dimension for representation
            representations.append(self.transforms[i](x.mean(1)))  # B L D

        # Final LayerNorm
        x = self.last_ln(x)  # B T L D

        # Mean over sequence length
        x = x.mean(2)  # B T D

        # Classify
        if self.mode != "pre_distill":
            x = self.classifier(x)  # B T num_classes

        return representations, x

    def predict(self, x):
        """
        Convenience method for inference.
        Returns predicted template IDs.

        Args:
            x: input token IDs, shape (B, L)

        Returns:
            predictions: (B,) tensor of predicted template IDs
        """
        self.eval()
        with torch.no_grad():
            _, outputs = self.forward(x)
            # outputs: B T num_classes
            logits = torch.mean(outputs, dim=1)  # B num_classes
            predictions = torch.argmax(logits, dim=-1)  # B
        return predictions
