
import torch
import torch.nn as nn
import yaml


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.W_query = nn.Linear(hidden_size, hidden_size)
        self.W_key   = nn.Linear(hidden_size, hidden_size)
        self.V       = nn.Linear(hidden_size, 1)

    def forward(self, hidden, encoder_outputs):
        query   = self.W_query(hidden.unsqueeze(1))
        keys    = self.W_key(encoder_outputs)
        scores  = self.V(torch.tanh(query + keys))
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(weights * encoder_outputs, dim=1)
        return context, weights


class MusicLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size,
                 num_layers, dropout, use_attention=True):
        super(MusicLSTM, self).__init__()
        self.hidden_size   = hidden_size
        self.num_layers    = num_layers
        self.use_attention = use_attention

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm      = nn.LSTM(
            input_size  = embedding_dim,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            dropout     = dropout,
            batch_first = True
        )
        self.attention = BahdanauAttention(hidden_size)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        embedded                 = self.dropout(self.embedding(x))
        lstm_out, (hidden, cell) = self.lstm(embedded)
        if self.use_attention:
            context, weights = self.attention(hidden[-1], lstm_out)
        else:
            context = hidden[-1]
        output = self.fc(self.dropout(context))
        return output


def build_lstm(vocab_size):
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    m = config["model"]
    return MusicLSTM(
        vocab_size    = vocab_size,
        embedding_dim = m["embedding_dim"],
        hidden_size   = m["hidden_size"],
        num_layers    = m["num_layers"],
        dropout       = m["dropout"],
        use_attention = m["use_attention"]
    )


if __name__ == "__main__":
    vocab_size   = 577
    model        = build_lstm(vocab_size)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(model)
    print(f"\nTotal parameters: {total_params:,}")
    fake_input = torch.randint(0, vocab_size, (4, 64))
    output     = model(fake_input)
    print(f"Input shape  : {fake_input.shape}")
    print(f"Output shape : {output.shape}")
    print(f"\n✅ LSTM works!")