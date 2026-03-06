# src/models/transformer_model.py

import torch
import torch.nn as nn
import math
import yaml


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_len=512):
        super(PositionalEncoding, self).__init__()
        pe       = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() *
                             (-math.log(10000.0) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class MusicTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size,
                 num_layers, dropout, nhead=4):
        super(MusicTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_enc   = PositionalEncoding(embedding_dim)
        encoder_layer  = nn.TransformerEncoderLayer(
            d_model    = embedding_dim,
            nhead      = nhead,
            dim_feedforward = hidden_size,
            dropout    = dropout,
            batch_first = True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout     = nn.Dropout(dropout)
        self.fc          = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        embedded    = self.pos_enc(self.embedding(x))
        embedded    = self.dropout(embedded)
        transformer_out = self.transformer(embedded)
        output      = self.fc(transformer_out[:, -1, :])
        return output


def build_transformer(vocab_size):
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    m = config["model"]
    return MusicTransformer(
        vocab_size    = vocab_size,
        embedding_dim = m["embedding_dim"],
        hidden_size   = m["hidden_size"],
        num_layers    = m["num_layers"],
        dropout       = m["dropout"],
        nhead         = 4
    )


if __name__ == "__main__":
    vocab_size   = 577
    model        = build_transformer(vocab_size)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(model)
    print(f"\nTotal parameters: {total_params:,}")
    fake_input   = torch.randint(0, vocab_size, (4, 64))
    output       = model(fake_input)
    print(f"Input shape  : {fake_input.shape}")
    print(f"Output shape : {output.shape}")
    print(f"\n✅ Transformer works!")