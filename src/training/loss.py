"""
Role: Defines the training loss criteria, including label smoothing cross entropy
as used in the MT3 / T5 architectures.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class MT3Loss(nn.Module):
    """
    Cross Entropy Loss with label smoothing, ignoring pad tokens.
    """
    def __init__(self, pad_idx: int, label_smoothing: float = 0.1):
        super().__init__()
        self.pad_idx = pad_idx
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes the loss over the sequence.
        logits: [batch_size, seq_len, vocab_size]
        targets: [batch_size, seq_len]
        """
        pass
