# src/models/rnn_model.py

import torch
import torch.nn as nn
import yaml


class MusicRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, dropout):
        super(MusicRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn       = nn.RNN(
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
        rnn_out, hidden = self.rnn(embedded)
        output          = self.fc(self.dropout(hidden[-1]))
        return output


def build_rnn(vocab_size):
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    m = config["model"]
    return MusicRNN(
        vocab_size    = vocab_size,
        embedding_dim = m["embedding_dim"],
        hidden_size   = m["hidden_size"],
        num_layers    = m["num_layers"],
        dropout       = m["dropout"]
    )


if __name__ == "__main__":
    vocab_size   = 577
    model        = build_rnn(vocab_size)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(model)
    print(f"\nTotal parameters: {total_params:,}")
    fake_input   = torch.randint(0, vocab_size, (4, 64))
    output       = model(fake_input)
    print(f"Input shape  : {fake_input.shape}")
    print(f"Output shape : {output.shape}")
    print(f"\n✅ RNN works!")