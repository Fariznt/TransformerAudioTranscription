"""
The core sequence-to-sequence MT3 model, combining the custom Encoder and Decoder.
Provides clear interfaces for the overall training workflow.
"""
import torch
import torch.nn as nn
from src.model.encoder import MT3Encoder
from src.model.decoder import MT3Decoder

class MT3Model(nn.Module):
    """
    The top-level model connecting the input audio features to the token vocabulary output.
    """
    def __init__(self, vocab_size: int, n_mels: int = 256, d_model: int = 512, 
                 num_heads: int = 8, num_layers: int = 6, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        
        # Projects the Log-Mel Spectrogram features (n_mels) up to the transformer dimension (d_model)
        self.audio_projection = nn.Linear(n_mels, d_model)
        
        # Embeds the discrete target MIDI tokens
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        self.encoder = MT3Encoder(num_layers, d_model, num_heads, d_ff, dropout)
        self.decoder = MT3Decoder(num_layers, d_model, num_heads, d_ff, dropout)
        
        # Maps the decoder d_model outputs back to the token vocabulary logits
        self.output_projection = nn.Linear(d_model, vocab_size)

    def forward(self, audio_features: torch.Tensor, decoder_input_tokens: torch.Tensor, 
                src_mask: torch.Tensor = None, tgt_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass connecting audio to logits (used during training).
        
        Args:
            audio_features: [batch_size, src_seq_len, n_mels]
            decoder_input_tokens: [batch_size, tgt_seq_len] (shifted input for teacher forcing)
            src_mask: boolean mask for padded audio frames
            tgt_mask: causal + pad mask for decoder inputs
            
        Returns:
            logits: [batch_size, tgt_seq_len, vocab_size]
        """
        pass
    
    def generate(self, audio_features: torch.Tensor, max_len: int, start_token_id: int) -> torch.Tensor:
        """
        Autoregressive generation of token predictions.
        (Called during evaluation/inference)
        
        Args:
            audio_features: [batch_size, src_seq_len, n_mels]
            max_len: maximum number of tokens to generate
            start_token_id: vocabulary index representing <SOS>
            
        Returns:
            predicted_tokens: [batch_size, generated_seq_len]
        """
        pass
