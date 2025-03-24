import torch.nn as nn
from torch import Tensor
from typing import Optional
from .attention import MultiHeadAttention
from .feedforward import FeedForwardLayer, DropoutLayer
from .normalization import LayerNormalization
from .residual import ResidualConnection

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.ff = FeedForwardLayer(d_model, d_ff)
        self.norm1 = LayerNormalization(d_model)
        self.norm2 = LayerNormalization(d_model)
        self.dropout1 = DropoutLayer(dropout)
        self.dropout2 = DropoutLayer(dropout)
        self.residual1 = ResidualConnection()
        self.residual2 = ResidualConnection()
    
    def forward(self, x: Tensor) -> Tensor:
        attn_output = self.residual1(x, lambda x: self.dropout1(self.self_attn(x, x, x)))
        attn_output = self.norm1(attn_output)
        ff_output = self.residual2(attn_output, lambda x: self.dropout2(self.ff(x)))
        return self.norm2(ff_output)
