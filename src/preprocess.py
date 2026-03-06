import pickle
from pathlib import Path
from music21 import corpus, converter, note, chord, stream
import numpy as np
from tqdm import tqdm
from data_loader import get_midi_files, load_config

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
                notes.append(str(element.pitch))

            elif isinstance(element, chord.Chord):
                notes.append(str(element.pitches[0]))

    except Exception as e:
        print(f"Could not parse {file_path}: {e}")

    return notes


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


def build_vocabulary(all_notes):
    
    # set() removes duplicates
    unique_notes = sorted(set(all_notes))
    vocab_size = len(unique_notes)

    # note → number dictionary
    note_to_int = {note: i for i, note in enumerate(unique_notes)}

    # number → note dictionary (for converting back after generation)
    int_to_note = {i: note for i, note in enumerate(unique_notes)}

    print(f"Vocabulary size: {vocab_size} unique notes")
    return note_to_int, int_to_note, vocab_size


def create_sequences(all_notes, note_to_int, sequence_length=64):
    
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


def save_preprocessed_data(inputs, targets, note_to_int, int_to_note):
    
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