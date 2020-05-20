import torch.nn as nn
from transformer_layers import *
import copy
from config import ARGS


class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
        super(Transformer, self).__init__()
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.N = N
        self.d_model = d_model
        self.d_ff = d_ff
        self.h = h
        self.dropout = dropout
        self.max_len = ARGS.max_len + 2  # consider <s>, </s>

        c = copy.deepcopy
        enc_self_attn = MultiHeadedAttention(h, d_model, dropout, attn_type=ARGS.enc_self_attn, attn_norm=ARGS.attn_norm, max_len=self.max_len, sparsity_mode=ARGS.sparsity_mode, sparsity_top_k=ARGS.sparsity_top_k)
        dec_self_attn = MultiHeadedAttention(h, d_model, dropout, attn_type=ARGS.dec_self_attn, attn_norm=ARGS.attn_norm, max_len=self.max_len, sparsity_mode=ARGS.sparsity_mode, sparsity_top_k=ARGS.sparsity_top_k)
        cross_attn = MultiHeadedAttention(h, d_model, dropout)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout, self.max_len)

        self.src_emb = nn.Sequential(Embeddings(d_model, src_vocab), c(position))
        self.tgt_emb = nn.Sequential(Embeddings(d_model, tgt_vocab), c(position))
        self.encoder = Encoder(EncoderLayer(d_model, c(enc_self_attn), c(ff), dropout), N)
        self.decoder = Decoder(DecoderLayer(d_model, c(dec_self_attn), c(cross_attn), c(ff), dropout), N)
        self.generator = Generator(d_model, tgt_vocab)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, src_mask, tgt_mask):
        embedded_src = self.src_emb(src)
        memory = self.encoder(embedded_src, src_mask)
        embedded_tgt = self.tgt_emb(tgt)
        decoder_output = self.decoder(embedded_tgt, memory, src_mask, tgt_mask)
        model_output = self.generator(decoder_output)

        return model_output


def get_model(pt_path):
    return torch.load(pt_path)
