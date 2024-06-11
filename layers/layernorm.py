import torch
import torch.nn as nn



class LayerNormalization(nn.Module):
    def __init__(self, d_model, eps = 1e-12):
        super(LayerNormalization, self).__init__()
        self.gama = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self,x):
        self.mean = x.mean(-1, keepdim=True)
        self.var = x.var(-1, unbiased=False, keepdim=True)

        out = (self.mean - self.var)/torch.sqrt(self.var)
        return self.gama*out + self.beta

