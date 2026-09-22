"""Trainable and adaptive temporal-compression research candidates."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as functional
from torch import nn


class LearnedTemporalDownsampler(nn.Module):
    """Lightweight depthwise/pointwise downsampler initialized as mean pooling."""

    def __init__(self, hidden_size: int, factor: int) -> None:
        super().__init__()
        if hidden_size < 1 or factor < 2:
            raise ValueError("hidden_size must be positive and factor must be at least 2.")
        self.factor = factor
        self.depthwise = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size=factor,
            stride=factor,
            groups=hidden_size,
            bias=False,
        )
        self.projection = nn.Linear(hidden_size, hidden_size)
        nn.init.constant_(self.depthwise.weight, 1.0 / factor)
        nn.init.eye_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pooled = self.depthwise(hidden_states.transpose(1, 2)).transpose(1, 2)
        return self.projection(pooled)


def adaptive_merge_hidden_states(
    hidden_states: torch.Tensor, reduction_ratio: float
) -> torch.Tensor:
    """Merge similar adjacent tokens while retaining high-change boundaries.

    This reference implementation is intended for quality screening and teacher
    distillation. An ONNX-friendly fixed-shape implementation is required before
    deployment profiling.
    """
    if hidden_states.ndim != 3 or hidden_states.shape[0] != 1:
        raise ValueError("Expected hidden states shaped [1, sequence, hidden].")
    if not 0.0 < reduction_ratio < 1.0:
        raise ValueError("reduction_ratio must be within (0, 1).")
    sequence_length = hidden_states.shape[1]
    target_length = max(1, math.ceil(sequence_length * (1.0 - reduction_ratio)))
    if target_length >= sequence_length:
        return hidden_states
    similarity = functional.cosine_similarity(
        hidden_states[:, :-1], hidden_states[:, 1:], dim=-1
    )[0]
    boundary_count = target_length - 1
    boundaries = torch.topk(1.0 - similarity, boundary_count).indices + 1
    boundaries = torch.sort(boundaries).values
    positions = torch.arange(sequence_length, device=hidden_states.device)
    segment_ids = torch.bucketize(positions, boundaries)
    merged = hidden_states.new_zeros((target_length, hidden_states.shape[2]))
    merged.index_add_(0, segment_ids, hidden_states[0])
    counts = torch.bincount(segment_ids, minlength=target_length).clamp_min(1)
    return (merged / counts[:, None]).unsqueeze(0)
