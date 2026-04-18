# config.py

DATASET_PATH = "Dataset"
WAVS_PATH = f"{DATASET_PATH}/wavs"
METADATA_PATH = f"{DATASET_PATH}/metadata.csv"

# reference speaker file
SPEAKER_WAV = f"{WAVS_PATH}/LJ001-0001.wav"

# output directory
OUTPUT_DIR = "outputs"

# dummy sentences
TEXT_SAMPLES = [
    "Hello, this is a cloned voice test, and this is used to test me cloner system.",
    "This system demonstrates voice cloning using XTTS.",
    "We are evaluating similarity between real and generated speech.",
    "Deep learning enables realistic speech synthesis."
]

DEVICE = "cuda"  # change to "cpu" if no GPU