import torch.nn as nn
import torch
from layers.attention_head import SelfAttention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.attention = SelfAttention()
        self.n_head = n_head
        self.WQ = nn.Linear(d_model, d_model)
        self.WK = nn.Linear(d_model, d_model)
        self.WV = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self,q,k,v, mask=None):
        Q = self.WQ(q)
        K = self.WK(k)
        V = self.WV(v)
        #print(f'{Q.size()} {K.size()} {V.size()}')
        Q,K,V = self.tensor_split(Q), self.tensor_split(K), self.tensor_split(V)
        #print(f'{Q.size()} {K.size()} {V.size()}')
        out, attention = self.attention(Q,K,V, mask=mask)
        #print(f'{out.size()} {attention.size()}')
        out = self.tensor_concat(out)
        #print(out.size())
        out = self.w_concat(out)
        return out


    def tensor_split(self, T):
        batch_size, length, d_model = T.size()
        d_tensor = d_model//self.n_head
        return T.view(batch_size, length, self.n_head, d_tensor).transpose(1,2)

    def tensor_concat(self, T):
        batch_size, head, length, d_tensor = T.size()
        d_model = head*d_tensor
        return T.transpose(1,2).contiguous().view(batch_size, length, d_model)


if __name__=='__main__':
    x = torch.ones(1,100,512)
    mult_head_attention = MultiHeadAttention(512, 8)
    print(mult_head_attention(x).size())

