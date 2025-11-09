import math
import copy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def clones(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, q, k, v, mask: Optional[torch.Tensor] = None):
        # q, k, v: (batch, heads, seq_len, d_k)
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # mask shape should be broadcastable to scores
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)
        return output, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # in_proj: project input to Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        # out projection
        self.w_o = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention(dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask: Optional[torch.Tensor] = None):
        batch_size = query.size(0)

        # linear projections
        q = self.w_q(query)  # (B, Lq, d_model)
        k = self.w_k(key)
        v = self.w_v(value)

        # split into heads and transpose
        def split_heads(x):
            # x: (B, L, d_model) -> (B, heads, L, d_k)
            return x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        # adapt mask shape if provided: expected (B, 1, Lq, Lk) or broadcastable
        if mask is not None:
            # ensure mask has shape (B, 1, Lq, Lk) for broadcasting over heads
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)

        x, attn = self.attention(q, k, v, mask=mask)
        # x: (B, heads, Lq, d_k) -> concat
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        x = self.w_o(x)
        x = self.dropout(x)
        return x, attn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (B, L, d_model)
        L = x.size(1)
        x = x + self.pe[:, :L]
        return x


class SublayerConnection(nn.Module):
    """Residual connection followed by layer normalization."""

    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """Apply residual connection to any sublayer with the same size."""
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, heads, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout=dropout)
        self.sublayer = clones(SublayerConnection(d_model, dropout=dropout), 2)
        self.d_model = d_model

    def forward(self, x, src_mask=None):
        # self-attention sublayer
        x = self.sublayer[0](x, lambda _x: self.self_attn(_x, _x, _x, mask=src_mask)[0])
        # feed-forward sublayer
        x = self.sublayer[1](x, self.feed_forward)
        return x


class Encoder(nn.Module):
    def __init__(self, layer: EncoderLayer, N: int):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.d_model)

    def forward(self, x, src_mask=None):
        for layer in self.layers:
            x = layer(x, src_mask=src_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, heads, dropout=dropout)
        self.src_attn = MultiHeadAttention(d_model, heads, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout=dropout)
        self.sublayer = clones(SublayerConnection(d_model, dropout=dropout), 3)
        self.d_model = d_model

    def forward(self, x, memory, src_mask=None, tgt_mask=None):
        # masked self-attention
        x = self.sublayer[0](x, lambda _x: self.self_attn(_x, _x, _x, mask=tgt_mask)[0])
        # src-target attention
        x = self.sublayer[1](x, lambda _x: self.src_attn(_x, memory, memory, mask=src_mask)[0])
        # feed-forward
        x = self.sublayer[2](x, self.feed_forward)
        return x


class Decoder(nn.Module):
    def __init__(self, layer: DecoderLayer, N: int):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.d_model)

    def forward(self, x, memory, src_mask=None, tgt_mask=None):
        for layer in self.layers:
            x = layer(x, memory, src_mask=src_mask, tgt_mask=tgt_mask)
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(self,
                 src_vocab: int,
                 tgt_vocab: int,
                 d_model: int = 512,
                 N: int = 6,
                 heads: int = 8,
                 d_ff: int = 2048,
                 dropout: float = 0.1,
                 max_len: int = 5000):
        super().__init__()
        self.src_embed = nn.Embedding(src_vocab, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)
        #self.pos_encoder = nn.Identity()

        encoder_layer = EncoderLayer(d_model, heads, d_ff, dropout)
        decoder_layer = DecoderLayer(d_model, heads, d_ff, dropout)
        self.encoder = Encoder(encoder_layer, N)
        self.decoder = Decoder(decoder_layer, N)

        self.d_model = d_model
        self.generator = nn.Linear(d_model, tgt_vocab)  # output projection

        self._init_parameters()

    def _init_parameters(self):
        # initialize weights following original transformer paper recommendations
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src, src_mask=None):
        # src: (B, L_src)
        x = self.src_embed(src) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        return self.encoder(x, src_mask=src_mask)

    def decode(self, tgt, memory, src_mask=None, tgt_mask=None):
        x = self.tgt_embed(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        return self.decoder(x, memory, src_mask=src_mask, tgt_mask=tgt_mask)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """Take in and process masked src and target sequences."""
        memory = self.encode(src, src_mask=src_mask)
        out = self.decode(tgt, memory, src_mask=src_mask, tgt_mask=tgt_mask)
        logits = self.generator(out)
        return logits


# Utility mask functions

def make_src_mask(src, pad_idx=0):
    # src: (B, L)
    return (src != pad_idx).unsqueeze(-2)  # (B, 1, L)


def make_tgt_mask(tgt, pad_idx=0):
    # tgt: (B, L)
    tgt_mask = (tgt != pad_idx).unsqueeze(-2)  # (B, 1, L)
    L = tgt.size(1)
    subsequent_mask = torch.triu(torch.ones((1, L, L), device=tgt.device), diagonal=1).bool()
    # allow positions where subsequent_mask == 0
    tgt_mask = tgt_mask & ~subsequent_mask
    return tgt_mask  # (B, L, L) with broadcasting where needed


if __name__ == "__main__":
    # Quick sanity check
    B, S, T = 2, 7, 6  # batch, src len, tgt len
    SRC_VOCAB = 11
    TGT_VOCAB = 13
    model = Transformer(SRC_VOCAB, TGT_VOCAB, d_model=128, N=2, heads=8, d_ff=512)

    src = torch.randint(1, SRC_VOCAB, (B, S))
    tgt = torch.randint(1, TGT_VOCAB, (B, T))
    src_mask = make_src_mask(src, pad_idx=0)  # (B,1,S)
    tgt_mask = make_tgt_mask(tgt, pad_idx=0)  # (B, T, T)

    logits = model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
    print('logits.shape =', logits.shape)  # expected (B, T, tgt_vocab)
