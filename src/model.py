# src/model.py
from models.lstm_model import MusicLSTM, BahdanauAttention, build_lstm

def build_model(vocab_size):
    return build_lstm(vocab_size)

if __name__ == "__main__":
    model        = build_model(577)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print("✅ model.py import works!")