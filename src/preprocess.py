# src/preprocess.py

import pickle
from pathlib import Path
from music21 import corpus, converter, note, chord, stream
import numpy as np
from tqdm import tqdm
from data_loader import get_midi_files, load_config

# ─────────────────────────────────────────────────────
# JOB 1 — Read one MIDI file and extract notes
# ─────────────────────────────────────────────────────
def extract_notes_from_file(file_path):
    """
    Opens one MIDI/MXL file and extracts all notes.
    Returns a list of strings like ["C4", "E4", "G4", "rest"]
    
    We convert chords to their lowest note — keeps it simple
    for the LSTM to learn from.
    """
    notes = []

    try:
        # Parse the file using music21
        score = converter.parse(str(file_path))

        # Flatten means: ignore instruments/parts, 
        # just give me ALL notes in one flat list
        all_notes = score.flatten().notes

        for element in all_notes:
            if isinstance(element, note.Note):
                # Single note → just take its name
                # e.g. "C4", "E5", "G3"
                notes.append(str(element.pitch))

            elif isinstance(element, chord.Chord):
                # Chord = multiple notes at once
                # We take the LOWEST note to keep it simple
                notes.append(str(element.pitches[0]))

    except Exception as e:
        print(f"Could not parse {file_path}: {e}")

    return notes


# ─────────────────────────────────────────────────────
# JOB 2 — Extract notes from ALL files
# ─────────────────────────────────────────────────────
def extract_all_notes(composer=None):
    """
    Loops through every MIDI file and extracts notes.
    Returns one big flat list of all notes from all files.
    """
    all_notes = []
    files = get_midi_files(composer)

    print(f"\nExtracting notes from {len(files)} files...")

    for file_path in tqdm(files):
        notes = extract_notes_from_file(file_path)
        all_notes.extend(notes)

    print(f"Total notes extracted: {len(all_notes)}")
    return all_notes


# ─────────────────────────────────────────────────────
# JOB 3 — Build vocabulary
# ─────────────────────────────────────────────────────
def build_vocabulary(all_notes):
    """
    Finds every UNIQUE note and assigns it a number.
    
    Example:
    notes = ["C4", "E4", "G4", "C4", "E4"]
    unique = ["C4", "E4", "G4"]
    vocab  = {"C4": 0, "E4": 1, "G4": 2}
    """
    # set() removes duplicates
    unique_notes = sorted(set(all_notes))
    vocab_size = len(unique_notes)

    # note → number dictionary
    note_to_int = {note: i for i, note in enumerate(unique_notes)}

    # number → note dictionary (for converting back after generation)
    int_to_note = {i: note for i, note in enumerate(unique_notes)}

    print(f"Vocabulary size: {vocab_size} unique notes")
    return note_to_int, int_to_note, vocab_size


# ─────────────────────────────────────────────────────
# JOB 4 — Create sequences for LSTM
# ─────────────────────────────────────────────────────
def create_sequences(all_notes, note_to_int, sequence_length=64):
    """
    Converts notes to numbers, then creates input/output pairs.
    
    Example with sequence_length=4 (we use 64 in real training):
    notes:   [C4, E4, G4, B4, A4, F4]
    numbers: [0,  1,  2,  3,  4,  5 ]
    
    Input sequence 1: [0, 1, 2, 3] → Target: 4
    Input sequence 2: [1, 2, 3, 4] → Target: 5
    
    This is called a SLIDING WINDOW
    """
    # Convert all notes to numbers
    all_ints = [note_to_int[n] for n in all_notes]

    inputs = []
    targets = []

    # Slide a window of size sequence_length across all notes
    for i in range(len(all_ints) - sequence_length):
        # Input: 64 notes
        input_seq = all_ints[i : i + sequence_length]
        # Target: the note that comes AFTER those 64
        target = all_ints[i + sequence_length]

        inputs.append(input_seq)
        targets.append(target)

    # Convert to numpy arrays for PyTorch
    inputs = np.array(inputs)
    targets = np.array(targets)

    print(f"Total sequences created: {len(inputs)}")
    print(f"Input shape: {inputs.shape}")
    print(f"Target shape: {targets.shape}")

    return inputs, targets


# ─────────────────────────────────────────────────────
# JOB 5 — Save everything to disk
# ─────────────────────────────────────────────────────
def save_preprocessed_data(inputs, targets, note_to_int, int_to_note):
    """
    Saves our processed data so we don't have to 
    reprocess every time we train.
    
    pickle = Python's way of saving any object to a file
    """
    save_dir = Path("data")
    save_dir.mkdir(exist_ok=True)

    np.save("data/inputs.npy", inputs)
    np.save("data/targets.npy", targets)

    with open("data/note_to_int.pkl", "wb") as f:
        pickle.dump(note_to_int, f)

    with open("data/int_to_note.pkl", "wb") as f:
        pickle.dump(int_to_note, f)

    print("\n✅ Saved: inputs.npy, targets.npy")
    print("✅ Saved: note_to_int.pkl, int_to_note.pkl")


# ─────────────────────────────────────────────────────
# RUN EVERYTHING
# ─────────────────────────────────────────────────────
if __name__ == "__main__":
    config = load_config()
    sequence_length = config["data"]["sequence_length"]

    # Step 1 - Extract all notes
    all_notes = extract_all_notes()

    # Step 2 - Build vocabulary
    note_to_int, int_to_note, vocab_size = build_vocabulary(all_notes)

    # Step 3 - Create sequences
    inputs, targets = create_sequences(all_notes, note_to_int, sequence_length)

    # Step 4 - Save to disk
    save_preprocessed_data(inputs, targets, note_to_int, int_to_note)

    print(f"\n🎵 Preprocessing complete!")
    print(f"   Vocab size : {vocab_size}")
    print(f"   Sequences  : {len(inputs)}")