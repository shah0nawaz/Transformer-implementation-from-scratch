import torch
import torch.nn as nn


from layers.positionalencoding import PositionalEncoding
from blocks.encoder_block import EncoderLayer
from blocks.decoder_block import DecoderLayer


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, d_model, max_seq_len, d_dif, n_heads,n_blocks, dropout):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(trg_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)

        # self.encoderblock = EncoderLayer(d_model, d_dif, n_heads, dropout)
        # self.decoderblock = DecoderLayer(d_model, d_dif, n_heads, dropout)

        self.encoderblocks = nn.ModuleList([EncoderLayer(d_model, d_dif, n_heads, dropout) for _ in range(n_blocks)])
        self.decoderblocks = nn.ModuleList([DecoderLayer(d_model, d_dif, n_heads, dropout) for _ in range(n_blocks)])

        self.fc = nn.Linear(d_model, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)


    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, trg):
        encoder_wembeddings = self.encoder_embedding(src)
        decoder_wembeddings = self.decoder_embedding(trg)
        #print(encoder_wembeddings.size())
        positioned_en_wembeddings = self.positional_encoding(encoder_wembeddings)
        positioned_de_wembeddings = self.positional_encoding(decoder_wembeddings)

        src_embedded = self.dropout(positioned_en_wembeddings)
        trg_embedded = self.dropout(positioned_de_wembeddings)

        src_mask, trg_mask = self.generate_mask(src, trg)

        enc_out = src_embedded
        for enc_block in self.encoderblocks:
            enc_out = enc_block(enc_out, src_mask)

        dec_out = trg_embedded
        for dec_block in self.decoderblocks:
            dec_out = dec_block(dec_out, enc_out, enc_out, src_mask, trg_mask)


        fc_out = self.fc(dec_out)
        prob = self.softmax(fc_out)
        return prob


