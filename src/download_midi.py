
from music21 import corpus
import shutil
from pathlib import Path

def download_bach():
    """
    music21 has Bach built in -- no internet needed!
    We just copy them into our data/midi/bach/ folder.
    """
    output_dir = Path("data/midi/bach")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all Bach pieces from music21's built-in library
    bach_paths = corpus.getComposer('bach')

    count = 0
    for path in bach_paths:
        # Only take .mid or .xml files
        if str(path).endswith(('.mid', '.midi', '.xml', '.mxl')):
            dest = output_dir / Path(path).name
            shutil.copy(str(path), str(dest))
            count += 1
            print(f"Copied: {Path(path).name}")

    print(f"\n✅ Bach: {count} files downloaded")


if __name__ == "__main__":
    download_bach()