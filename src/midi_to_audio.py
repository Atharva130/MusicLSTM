# src/midi_to_audio.py

import subprocess
from pathlib import Path

SOUNDFONT  = r"C:\ProgramData\soundfonts\default.sf2"
FLUIDSYNTH = r"C:\tools\fluidsynth\bin\fluidsynth.exe"

def midi_to_mp3(midi_path, output_path=None):
    midi_path   = Path(midi_path).resolve()
    output_path = Path(output_path).resolve() if output_path else midi_path.with_suffix(".wav")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        FLUIDSYNTH,
        "-ni",
        "-g", "2.5",          
        "-T", "wav",
        "-F", str(output_path),
        SOUNDFONT,
        str(midi_path),
    ]
    subprocess.call(cmd)
    print(f"✅ Audio saved: {output_path}")
    return str(output_path)


if __name__ == "__main__":
    midi_to_mp3("data/audio/output.mid", "data/audio/output.wav")