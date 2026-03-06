# src/models/gru_model.py

import torch
import torch.nn as nn
import yaml


class MusicGRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, dropout):
        super(MusicGRU, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru       = nn.GRU(
            input_size  = embedding_dim,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            dropout     = dropout,
            batch_first = True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        embedded        = self.dropout(self.embedding(x))
        gru_out, hidden = self.gru(embedded)
        output          = self.fc(self.dropout(hidden[-1]))
        return output


def build_gru(vocab_size):
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    m = config["model"]
    return MusicGRU(
        vocab_size    = vocab_size,
        embedding_dim = m["embedding_dim"],
        hidden_size   = m["hidden_size"],
        num_layers    = m["num_layers"],
        dropout       = m["dropout"]
    )


if __name__ == "__main__":
    vocab_size   = 577
    model        = build_gru(vocab_size)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(model)
    print(f"\nTotal parameters: {total_params:,}")
    fake_input   = torch.randint(0, vocab_size, (4, 64))
    output       = model(fake_input)
    print(f"Input shape  : {fake_input.shape}")
    print(f"Output shape : {output.shape}")
    print(f"\n✅ GRU works!")