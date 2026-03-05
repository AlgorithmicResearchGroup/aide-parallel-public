"""Baseline attention module for the Attention-Agent task.

The evaluation harness expects a ``build_attention(config)`` factory that returns an ``nn.Module`` compatible with
nanoGPT's transformer blocks. The module receives tensors shaped ``(batch, sequence, embed_dim)`` and must emit tensors
with the same shape. This baseline mirrors the standard scaled dot-product causal multi-head attention used in GPT-2.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalMultiheadAttention(nn.Module):
    """Standard causal multi-head self-attention."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.embed_dim = int(config.n_embd)
        self.num_heads = int(config.n_head)
        self.dropout = float(getattr(config, "dropout", 0.0))
        self.bias = bool(getattr(config, "bias", True))
        self.block_size = int(config.block_size)

        if self.embed_dim % self.num_heads != 0:
            raise ValueError("n_embd must be divisible by n_head")

        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=self.bias)
        self.proj = nn.Linear(self.embed_dim, self.embed_dim, bias=self.bias)
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)

        mask = torch.tril(torch.ones(self.block_size, self.block_size, dtype=torch.bool))
        self.register_buffer("attn_mask", mask)

    def _reshape(self, tensor: torch.Tensor) -> torch.Tensor:
        B, T, _ = tensor.shape
        tensor = tensor.view(B, T, self.num_heads, self.head_dim)
        return tensor.transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = self._reshape(q)
        k = self._reshape(k)
        v = self._reshape(v)

        att = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        mask = self.attn_mask[:T, :T]
        att = att.masked_fill(~mask, float("-inf"))
        probs = F.softmax(att, dim=-1)
        probs = self.attn_dropout(probs)

        y = torch.matmul(probs, v)
        y = y.transpose(1, 2).contiguous().view(B, T, self.embed_dim)
        y = self.resid_dropout(self.proj(y))
        return y


def build_attention(config: Any) -> nn.Module:
    """Factory used by the evaluation harness to create the attention block."""

    return CausalMultiheadAttention(config)
