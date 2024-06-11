import torch
import torch.nn as nn

from layers.positionwiseffnn import PositionwiseFeedForward
from layers.layernorm import LayerNormalization
from layers.multihead_attention import MultiHeadAttention

class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_dif, n_head, dropout):
        super(DecoderLayer,self).__init__()
        self.m_masked_att = MultiHeadAttention(d_model, n_head)
        self.m_cross_att = MultiHeadAttention(d_model, n_head)
        self.lnorm1 = LayerNormalization(d_model)
        self.lnorm2 = LayerNormalization(d_model)
        self.lnorm3 = LayerNormalization(d_model)

        self.pwffn = PositionwiseFeedForward(d_model, d_dif)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, enc_out, en_out, src_mask, trg_mask):

        out_m_matt = self.m_masked_att(x,x,x,trg_mask)
        x = self.lnorm1(x + self.dropout(out_m_matt))

        out_m_catt = self.m_cross_att(x, enc_out,en_out, src_mask)
        x = self.lnorm2(x+ self.dropout(out_m_catt))

        pwffn_out = self.pwffn(x)
        x = self.lnorm3(x + self.dropout(pwffn_out))

        return x


