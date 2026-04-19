"""
Stacked Transformer Encoder for MT3.

Uses native PyTorch nn.TransformerEncoderLayer blocks. The T5-style relative 
position bias is injected into each layer's self-attention through the additive 
`attn_mask` argument, removing the need for a custom attention implementation.
"""
import torch
import torch.nn as nn
from src.model.positional_encoding import T5RelativePositionBias

class MT3Encoder(nn.Module):
    """
    Stack of encoder layers operating on projected Log-Mel spectrogram features.
    """
    def __init__(self, num_layers: int, d_model: int, num_heads: int, d_ff: int, dropout: float):
        super().__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Shared bidirectional relative position bias applied at every layer
        self.relative_position_bias = T5RelativePositionBias(
            num_heads=num_heads, bidirectional=True
        )

    def forward(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, src_seq_len, d_model]
            src_key_padding_mask: [batch_size, src_seq_len] bool tensor, True for padding.
        
        Returns:
            encoded: [batch_size, src_seq_len, d_model]
        
        The relative position bias (shape [1, num_heads, L, L]) is reshaped to 
        [batch * num_heads, L, L] and passed as the additive `mask` argument to the 
        underlying nn.TransformerEncoder.
        """
        pass
