import torch
import torch.nn as nn
import math
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len, device='cpu'):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_seq_len, d_model, device=device)
        self.pos = torch.arange(0,max_seq_len, device=device)
        self.pos = self.pos.float().unsqueeze(dim=-1)

        _2i = torch.arange(0, d_model, step = 2 , device=device).float()

        self.encoding[:, 0::2] = torch.sin(self.pos / (10000** (_2i/d_model)))
        self.encoding[:, 1::2] = torch.sin(self.pos / (10000** (_2i/d_model)))


    def forward(self,x):
        batch_size, seq_len,_ = x.size()
        x = x + self.encoding[:seq_len,:]
        return x

if __name__=='__main__':
    pe = PositionalEncoding(50000, 512)
    x = torch.ones([10,5000])
    print(pe(x)[0:1:3])




