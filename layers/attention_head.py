import torch.nn as nn
import torch
import math
import numpy as np

class SelfAttention(nn.Module):
    def __init__(self):
        super(SelfAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e = 1e-12):
        batch_size, head, length, d_model = k.size()
        k_t = k.transpose(2,3)
        s = (q @ k_t)/math.sqrt(d_model)
        if mask is not None:
            s = s.masked_fill(mask==0, -10000)
        score = self.softmax(s)
        v = score @ v
        return v, score

if __name__=='__main__':
    b,h,l,d = 10,2,512,128
    q = torch.zeros((b,h,l,d))
    k = torch.zeros((b,h,l,d))
    v = torch.zeros((b,h,l,d))
    selfattention = SelfAttention()
    selfattention(q,k,v)






