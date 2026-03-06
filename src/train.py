
import torch
import torch.nn as nn
import numpy as np
import pickle
import yaml
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
from model import build_model


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
    vocab_size = len(note_to_int)
    return inputs, targets, vocab_size


def train(resume_from=None):
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    t = config["training"]

    device = torch.device(t["device"] if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    inputs, targets, vocab_size = load_data()
    dataset    = MusicDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=t["batch_size"], shuffle=True)

    model     = build_model(vocab_size).to(device)
    criterion = nn.CrossEntropyLoss()

    # Resume from checkpoint if given
    if resume_from:
        model.load_state_dict(torch.load(resume_from, map_location=device))
        print(f"✅ Resumed from: {resume_from}")

    # Lower learning rate for fine tuning
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Reduces LR by 50% when loss stops improving for 3 epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode     = "min",
        factor   = 0.5,
        patience = 3
    )

    Path("checkpoints").mkdir(exist_ok=True)

    for epoch in range(1, 51):
        model.train()
        total_loss = 0

        for inputs_batch, targets_batch in tqdm(dataloader, desc=f"Epoch {epoch}/50"):
            inputs_batch  = inputs_batch.to(device)
            targets_batch = targets_batch.to(device)

            optimizer.zero_grad()
            output = model(inputs_batch)
            loss   = criterion(output, targets_batch)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), t["clip_grad_norm"])
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch} | Loss: {avg_loss:.4f}")
        scheduler.step(avg_loss)

        if epoch % 10 == 0:
            path = f"checkpoints/finetune_epoch_{epoch}.pt"
            torch.save(model.state_dict(), path)
            print(f"✅ Saved: {path}")

    torch.save(model.state_dict(), "checkpoints/model_final.pt")
    print("\n🎵 Fine tuning complete!")


if __name__ == "__main__":
    train(resume_from="checkpoints/finetune_epoch_20.pt")


if __name__ == "__main__":
    train()