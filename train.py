import os
import torch
import torch.nn as nn
import random
import  torch.optim as optim


from model.transformer import Transformer


src_vocab_size = 5000
trg_vocab_size = 5000
max_seq_len = 100
d_model = 512
n_heads = 8
n_blocks = 6
d_ff = 2048
batch_size = 64


dropout = 0.3
n_epochs = 1000

trans = Transformer(src_vocab_size, trg_vocab_size, d_model,max_seq_len, d_ff, n_heads, n_heads, dropout)



criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(trans.parameters(), lr = 0.0001, betas = (0.9,0.98), eps=1e-9)

src_data = torch.randint(1, src_vocab_size, (batch_size, max_seq_len ))
trg_data = torch.randint(1, trg_vocab_size, (batch_size, max_seq_len))

#print(src_data.shape)
#print(trg_data.shape)

trans.train()

for epoch in range(n_epochs):
    optimizer.zero_grad()
    output = trans(src_data, trg_data[:,:-1])
    loss = criterion(output.contiguous().view(-1, trg_vocab_size), trg_data[:,1:].contiguous().view(-1))
    loss.backward()
    optimizer.step()

    print(f'loss at {epoch} is {loss}')
