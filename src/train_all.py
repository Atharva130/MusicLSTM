# src/train_all.py

import torch
import torch.nn as nn
import numpy as np
import pickle
import yaml
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm

from models.rnn_model         import build_rnn
from models.gru_model         import build_gru
from models.lstm_model        import build_lstm
from models.transformer_model import build_transformer


class MusicDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs  = torch.tensor(inputs, dtype=torch.long)
        self.targets = torch.tensor(targets, dtype=torch.long)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def load_data():
    inputs  = np.load("data/inputs.npy")
    targets = np.load("data/targets.npy")
    with open("data/note_to_int.pkl", "rb") as f:
        note_to_int = pickle.load(f)
    return inputs, targets, len(note_to_int)


def train_model(model, dataloader, epochs, device, model_name):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    loss_history = []
    acc_history  = []
    Path("checkpoints").mkdir(exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss    = 0
        total_correct = 0
        total_samples = 0

        for inputs_batch, targets_batch in tqdm(dataloader, desc=f"{model_name} Epoch {epoch}/{epochs}"):
            inputs_batch  = inputs_batch.to(device)
            targets_batch = targets_batch.to(device)

            optimizer.zero_grad()
            output = model(inputs_batch)
            loss   = criterion(output, targets_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()

            # Accuracy — did we predict the right note?
            predicted      = torch.argmax(output, dim=1)
            total_correct += (predicted == targets_batch).sum().item()
            total_samples += targets_batch.size(0)

        avg_loss = total_loss / len(dataloader)
        accuracy = (total_correct / total_samples) * 100

        loss_history.append(avg_loss)
        acc_history.append(accuracy)
        scheduler.step(avg_loss)

        print(f"{model_name} | Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}%")

    torch.save(model.state_dict(), f"checkpoints/{model_name}_final.pt")
    print(f"✅ Saved: checkpoints/{model_name}_final.pt\n")
    return loss_history, acc_history


def plot_comparison(results):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for model_name, (losses, accs) in results.items():
        ax1.plot(losses, label=model_name)
        ax2.plot(accs,   label=model_name)

    ax1.set_title("Loss Over Epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    ax2.set_title("Accuracy Over Epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("ablation_comparison.png", dpi=150)
    plt.show()
    print("✅ Saved: ablation_comparison.png")

def print_comparison_table(results):
    print("\n" + "=" * 70)
    print(f"{'Model':<20} {'Final Loss':<15} {'Best Loss':<15} {'Best Acc':<15}")
    print("=" * 70)
    for model_name, (losses, accs) in results.items():
        print(f"{model_name:<20} {losses[-1]:<15.4f} {min(losses):<15.4f} {max(accs):<15.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}\n")

    inputs, targets, vocab_size = load_data()
    dataset    = MusicDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    models = {
        "Transformer" : build_transformer(vocab_size).to(device),
    }

    results = {}

    # Add LSTM results manually from our existing training
    results["LSTM"] = (
        [1.89],   # loss
        [None]    # acc not recorded during LSTM training
    )

    for model_name, model in models.items():
        print(f"\n{'='*50}")
        print(f"  Training {model_name}")
        print(f"{'='*50}")
        losses, accs        = train_model(model, dataloader, epochs, device, model_name)
        results[model_name] = (losses, accs)

    plot_comparison(results)
    print_comparison_table(results)
    print("\n🔥 Ablation study complete!")


    