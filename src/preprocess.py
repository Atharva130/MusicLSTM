# src/preprocess.py

import pickle
from pathlib import Path
from music21 import converter, note, chord
import numpy as np
from tqdm import tqdm
from data_loader import get_midi_files, load_config


def extract_notes_from_file(file_path):
    notes = []
    try:
        score     = converter.parse(str(file_path))
        all_notes = score.flatten().notes

        for element in all_notes:
            duration = round(element.quarterLength * 2) / 2
            duration = max(0.25, min(duration, 2.0))

            if isinstance(element, note.Note):
                token = f"{element.pitch}_{duration}"
                notes.append(token)

            elif isinstance(element, chord.Chord):
                token = f"{element.pitches[0]}_{duration}"
                notes.append(token)

    except Exception as e:
        print(f"Could not parse {file_path}: {e}")
    return notes


def extract_all_notes(composer=None):
    all_notes = []
    files     = get_midi_files(composer)
    print(f"\nExtracting notes from {len(files)} files...")
    for file_path in tqdm(files):
        notes = extract_notes_from_file(file_path)
        all_notes.extend(notes)
    print(f"Total tokens extracted: {len(all_notes)}")
    return all_notes


def build_vocabulary(all_notes):
    unique_notes = sorted(set(all_notes))
    vocab_size   = len(unique_notes)
    note_to_int  = {n: i for i, n in enumerate(unique_notes)}
    int_to_note  = {i: n for i, n in enumerate(unique_notes)}
    print(f"Vocabulary size: {vocab_size} unique tokens")
    return note_to_int, int_to_note, vocab_size


def create_sequences(all_notes, note_to_int, sequence_length=64):
    all_ints = [note_to_int[n] for n in all_notes]
    inputs   = []
    targets  = []

    for i in range(len(all_ints) - sequence_length):
        inputs.append(all_ints[i : i + sequence_length])
        targets.append(all_ints[i + sequence_length])

    inputs  = np.array(inputs)
    targets = np.array(targets)

    print(f"Total sequences: {len(inputs)}")
    print(f"Input shape:     {inputs.shape}")
    print(f"Target shape:    {targets.shape}")
    return inputs, targets


def save_preprocessed_data(inputs, targets, note_to_int, int_to_note):
    np.save("data/inputs.npy", inputs)
    np.save("data/targets.npy", targets)
    with open("data/note_to_int.pkl", "wb") as f:
        pickle.dump(note_to_int, f)
    with open("data/int_to_note.pkl", "wb") as f:
        pickle.dump(int_to_note, f)
    print("\n✅ Saved: inputs.npy, targets.npy")
    print("✅ Saved: note_to_int.pkl, int_to_note.pkl")


if __name__ == "__main__":
    config          = load_config()
    sequence_length = config["data"]["sequence_length"]
    all_notes       = extract_all_notes()
    note_to_int, int_to_note, vocab_size = build_vocabulary(all_notes)
    inputs, targets = create_sequences(all_notes, note_to_int, sequence_length)
    save_preprocessed_data(inputs, targets, note_to_int, int_to_note)
    print(f"\n🎵 Preprocessing complete!")
    print(f"   Vocab size : {vocab_size}")
    print(f"   Sequences  : {len(inputs)}")