"""Training module for transformer-based language models."""

from .train import get_model
from .data import CoTDataset

__all__ = ["get_model", "CoTDataset"]