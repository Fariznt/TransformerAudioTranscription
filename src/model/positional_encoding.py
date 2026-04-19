"""
Implements the Relative Position Bias used in T5 architectures,
as required by the MT3 paper for audio-to-MIDI transcription.

Unlike standard absolute sinusoidal embeddings, T5 computes a scalar bias
added directly to the attention logits based on the relative distance
between query and key positions.
"""
import torch
import torch.nn as nn
import math

class T5RelativePositionBias(nn.Module):
    """
    Computes a relative positional bias matrix to be added to the attention logits.
    Follows the T5 formulation: maps distances to a fixed number of buckets.
    """
    def __init__(self, num_heads: int, num_buckets: int = 32, max_distance: int = 128, bidirectional: bool = True):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.bidirectional = bidirectional
        
        # The embedding lookup for the relative position buckets
        self.relative_attention_bias = nn.Embedding(self.num_buckets, num_heads)

    def compute_bias(self, query_length: int, key_length: int, device: torch.device) -> torch.Tensor:
        """
        Creates the (query_length, key_length) bias matrix.
        Returns tensor of shape [1, num_heads, query_length, key_length].
        """
        pass

    def forward(self, query_length: int, key_length: int) -> torch.Tensor:
        """
        Provides the computed bias matrix during the forward pass.
        """
        device = self.relative_attention_bias.weight.device
        return self.compute_bias(query_length, key_length, device)
