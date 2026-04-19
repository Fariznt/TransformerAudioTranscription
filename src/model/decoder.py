"""
Stacked Transformer Decoder for MT3.

Uses native PyTorch nn.TransformerDecoderLayer blocks. The T5-style causal relative 
position bias is combined with the causal mask and injected via the additive 
`tgt_mask` argument. Cross-attention uses only standard masking.
"""
import torch
import torch.nn as nn
from src.model.positional_encoding import T5RelativePositionBias

class MT3Decoder(nn.Module):
    """
    Stack of decoder layers that autoregressively attend to encoded audio features.
    """
    def __init__(self, num_layers: int, d_model: int, num_heads: int, d_ff: int, dropout: float):
        super().__init__()
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Shared unidirectional (causal) relative position bias
        self.relative_position_bias = T5RelativePositionBias(
            num_heads=num_heads, bidirectional=False
        )

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor = None,
        memory_key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            tgt: [batch_size, tgt_seq_len, d_model] embedded decoder inputs
            memory: [batch_size, src_seq_len, d_model] encoder output
            tgt_key_padding_mask: bool tensor marking padded target tokens
            memory_key_padding_mask: bool tensor marking padded source frames
        
        Returns:
            decoded: [batch_size, tgt_seq_len, d_model]
        
        The causal mask and relative position bias are summed into a single float 
        additive mask of shape [batch * num_heads, L, L] and passed as `tgt_mask`.
        """
        pass
