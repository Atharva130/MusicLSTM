import os
import yaml
from pathlib import Path

def load_config(config_path="config.yaml"):
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_midi_files(composer=None):
    
    config = load_config()
    midi_dir = Path(config["data"]["midi_dir"])

    midi_files = []

    # If user asks for specific composer
    if composer:
        composer_path = midi_dir / composer
        # glob means "find all files matching this pattern"
        files = [f for f in composer_path.iterdir() if f.suffix in ['.mid', '.midi', '.xml', '.mxl']]
        midi_files.extend(files)

    # If no composer specified, get ALL composers
    else:
        composers = config["data"]["composers"]
        for comp in composers:
            composer_path = midi_dir / comp
            files = [f for f in composer_path.iterdir() if f.suffix in ['.mid', '.midi', '.xml', '.mxl']]
            midi_files.extend(files)

    return midi_files


def dataset_summary():
    
    config = load_config()
    composers = config["data"]["composers"]

    print("=" * 40)
    print("       DATASET SUMMARY")
    print("=" * 40)

    total = 0
    for composer in composers:
        files = get_midi_files(composer)
        count = len(files)
        total += count
        print(f"  {composer.capitalize():<15} {count} files")

    print("-" * 40)
    print(f"  {'Total':<15} {total} files")
    print("=" * 40)


if __name__ == "__main__":
    dataset_summary()