import torch
import torch.nn as nn
from typing import Optional
from .attention import MultiHeadAttention
from .feedforward import FeedForwardLayer, DropoutLayer
from .normalization import LayerNormalization
from .residual import ResidualConnection


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.ff = FeedForwardLayer(d_model, d_ff)
        self.norm1 = LayerNormalization(d_model)
        self.norm2 = LayerNormalization(d_model)
        self.norm3 = LayerNormalization(d_model)
        self.dropout1 = DropoutLayer(dropout)
        self.dropout2 = DropoutLayer(dropout)
        self.dropout3 = DropoutLayer(dropout)
        self.residual1 = ResidualConnection()
        self.residual2 = ResidualConnection()
        self.residual3 = ResidualConnection()
    
    def forward(self, x: torch.Tensor, memory: torch.Tensor, src_mask: Optional[torch.Tensor] = None, tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_output = self.residual1(x, lambda x: self.dropout1(self.self_attn(x, x, x, tgt_mask)))
        attn_output = self.norm1(attn_output)
        cross_output = self.residual2(attn_output, lambda x: self.dropout2(self.cross_attn(x, memory, memory, src_mask)))
        cross_output = self.norm2(cross_output)
        ff_output = self.residual3(cross_output, lambda x: self.dropout3(self.ff(x)))
        return self.norm3(ff_output)