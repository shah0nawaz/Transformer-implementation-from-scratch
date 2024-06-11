import torch
import torch.nn as nn

from layers.attention_head import SelfAttention
from layers.multihead_attention import MultiHeadAttention
from layers.layernorm import LayerNormalization
from layers.positionwiseffnn import PositionwiseFeedForward


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_dif, n_head, dropout):
        super(EncoderLayer, self).__init__()
        self.m_headatt = MultiHeadAttention(d_model, n_head)
        self.lnorm1 = LayerNormalization(d_model)
        self.lnorm2 = LayerNormalization(d_model)
        self.pwisefnn = PositionwiseFeedForward(d_model, d_dif)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x , mask):
        out_matt = self.m_headatt(x,x,x,mask=mask)
        x = self.lnorm1(x + self.dropout(out_matt))
        output_ffn = self.pwisefnn(x)
        x = self.lnorm2(x + self.dropout(output_ffn))
        return x

