import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

class PositionalEncoding(nn.Module):
    def __init__(self, context_size, d_model):
        super().__init__()
        self.encoding = torch.zeros(context_size, d_model)
        pos = torch.arange(0, context_size).unsqueeze(dim=1)
        dim = torch.arange(0, d_model, 2)

        # dim_2 = torch.arange(1, d_model, 2)
        # import ipdb; ipdb.set_trace()
        self.encoding[:, 0::2] = torch.sin(pos / (10000**(2 * dim / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000**(2 * dim / d_model)))

    def forward(self, x):
        seq_len = x.size(1)
        return self.encoding[:seq_len, :]

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout_rate = 0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        hidden_states, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(hidden_states))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class Decoder(nn.Module):
    def __init__(self, context_size,
                 d_model, d_ff, num_heads, n_blocks, use_pos_embedding=False):
        super().__init__()
        self.pos_embedding = PositionalEncoding(context_size, d_model)
        self.use_pos_embedding = use_pos_embedding

        self.blocks = nn.ModuleList([
            DecoderBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
            )
            for _ in range(n_blocks)
        ])

    def forward(self, x):
        if self.use_pos_embedding:
            x = x + self.pos_embedding(x).to(x.device)
        for block in self.blocks:
            x = block(x)
        return x

class Transformer(nn.Module):
    def __init__(self, context_size,
                 d_model, d_ff, num_heads, n_blocks, n_classes, use_pos_embedding=False):
        super().__init__()

        self.decoder = Decoder(
            context_size,
            d_model,
            d_ff,
            num_heads,
            n_blocks,
            use_pos_embedding=use_pos_embedding
        )
        self.out = nn.Linear(d_model, n_classes)

    def forward(self, x):
        x = self.decoder(x)
        output = self.out(x[:, -1, :])
        return output
