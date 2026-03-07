# src/generate.py

import torch
import numpy as np
import pickle
import yaml
import random
from pathlib import Path
from music21 import stream, note, chord, tempo, pitch
from model import build_model

CHORD_INTERVALS = {
    "major": [0, 4, 7],
    "minor": [0, 3, 7]
}

def load_assets():
    with open("data/note_to_int.pkl", "rb") as f:
        note_to_int = pickle.load(f)
    with open("data/int_to_note.pkl", "rb") as f:
        int_to_note = pickle.load(f)
    return note_to_int, int_to_note


def load_model(vocab_size, checkpoint="checkpoints/finetune_epoch_30.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = build_model(vocab_size).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    return model, device


def get_seed(note_to_int):
    inputs = np.load("data/inputs.npy")
    idx    = np.random.randint(0, len(inputs))
    return inputs[idx].tolist()


def get_chord_notes(note_name, key_type="major"):
    try:
        p           = pitch.Pitch(note_name)
        intervals   = CHORD_INTERVALS[key_type]
        chord_notes = []
        for interval in intervals:
            new_pitch = pitch.Pitch(midi=p.midi + interval)
            chord_notes.append(str(new_pitch.nameWithOctave))
        return chord_notes
    except:
        return [note_name]


def generate(mood="calm", length=300, temperature=0.8):
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    note_to_int, int_to_note = load_assets()
    vocab_size               = len(note_to_int)
    model, device            = load_model(vocab_size)
    sequence                 = get_seed(note_to_int)[-config["data"]["sequence_length"]:]
    generated                = []

    print(f"Generating {length} notes | Mood: {mood} | Temp: {temperature}")

    with torch.no_grad():
        for _ in range(length):
            input_tensor = torch.tensor([sequence], dtype=torch.long).to(device)
            output       = model(input_tensor) / temperature
            probs        = torch.softmax(output, dim=1).squeeze()
            predicted    = torch.multinomial(probs, 1).item()
            generated.append(int_to_note[predicted])
            sequence = sequence[1:] + [predicted]

    return generated


def save_midi(generated_tokens, mood="calm", output_path="data/audio/output.mid"):
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    Path("data/audio").mkdir(parents=True, exist_ok=True)

    key_type = config["moods"][mood]["key"]
    bpm      = config["moods"][mood]["tempo"]
    music    = stream.Stream()
    music.append(tempo.MetronomeMark(number=bpm))

    # Dynamic curve: builds up then relaxes (like real music)
    total = len(generated_tokens)
    for i, token in enumerate(generated_tokens):
        try:
            parts     = token.rsplit("_", 1)
            note_name = parts[0]
            duration  = float(parts[1]) if len(parts) > 1 else 0.5

            # Velocity: soft start, louder middle, soft end
            progress = i / total
            if progress < 0.2:
                velocity = random.randint(40, 60)    # soft intro
            elif progress < 0.7:
                velocity = random.randint(65, 95)    # expressive middle
            else:
                velocity = random.randint(35, 55)    # gentle ending

            # Slight duration humanization (±5%)
            duration = duration * random.uniform(0.95, 1.05)

            if i % 4 == 0:
                chord_notes = get_chord_notes(note_name, key_type)
                new_chord               = chord.Chord(chord_notes)
                new_chord.quarterLength = duration
                new_chord.volume.velocity = velocity - 10  # chords slightly softer
                music.append(new_chord)
            else:
                new_note               = note.Note(note_name)
                new_note.quarterLength = duration
                new_note.volume.velocity = velocity
                music.append(new_note)

        except:
            rest               = note.Rest()
            rest.quarterLength = 0.5
            music.append(rest)

    music.write("midi", fp=output_path)
    print(f"✅ MIDI saved: {output_path}")
    return output_path


if __name__ == "__main__":
    mood      = "calm"
    tokens    = generate(mood=mood, length=300, temperature=0.8)
    midi_path = save_midi(tokens, mood=mood)
    print(f"\n🎵 Generated {len(tokens)} notes")
    print(f"📁 MIDI file: {midi_path}")