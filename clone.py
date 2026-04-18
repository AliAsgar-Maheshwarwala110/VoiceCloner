# clone.py

import os
import numpy as np
import librosa
import soundfile as sf
import torch
from TTS.api import TTS

# XTTS requires 22050 Hz
XTTS_SR = 22050
TOP_DB = 25
PROCESSED_DIR = "samples/processed"

# Load model once at import time
tts = TTS(
    model_name="tts_models/multilingual/multi-dataset/xtts_v2",
    gpu=torch.cuda.is_available()
)


def preprocess_audio(input_path: str, output_path: str) -> str:
    """
    Cleans a WAV file for XTTS:
      - Resamples to 22050 Hz
      - Trims silence (top_db=25)
      - Peak-normalizes to 0.9
    """
    audio, sr = librosa.load(input_path, sr=None)

    if sr != XTTS_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=XTTS_SR)

    # Trim leading/trailing silence
    audio, _ = librosa.effects.trim(audio, top_db=TOP_DB)

    # Peak normalize
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.9

    sf.write(output_path, audio, XTTS_SR)
    return output_path


def best_chunks(input_path: str, n_chunks: int = 3, chunk_sec: int = 10) -> list:
    """
    Selects the N highest-RMS (most energetic/clear) chunks from audio.
    Returns list of paths to chunk WAV files.
    """
    audio, sr = librosa.load(input_path, sr=XTTS_SR)
    chunk_len = int(chunk_sec * sr)
    step = chunk_len // 2

    segments = []
    for start in range(0, max(1, len(audio) - chunk_len), step):
        chunk = audio[start:start + chunk_len]
        if len(chunk) < sr * 3:  # skip chunks shorter than 3s
            continue
        rms = float(librosa.feature.rms(y=chunk).mean())
        segments.append((rms, chunk))

    if not segments:
        segments = [(0.0, audio)]

    segments.sort(key=lambda x: x[0], reverse=True)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    paths = []
    base = os.path.splitext(os.path.basename(input_path))[0]

    for i, (_, chunk) in enumerate(segments[:n_chunks]):
        out = os.path.join(PROCESSED_DIR, f"{base}_chunk{i}.wav")
        sf.write(out, chunk, XTTS_SR)
        paths.append(out)

    return paths


def clone_voice(text: str, speaker_wav: str, output_path: str, language: str = "en") -> str:
    """
    Full pipeline:
      1. Preprocess the uploaded/recorded speaker WAV
      2. Extract best chunks for richer speaker reference
      3. Run XTTS v2 with multi-chunk reference
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Step 1: clean the input
    clean_path = os.path.join(PROCESSED_DIR, "clean_input.wav")
    preprocess_audio(speaker_wav, clean_path)

    # Step 2: extract best chunks (multi-reference = better accuracy)
    chunks = best_chunks(clean_path, n_chunks=3, chunk_sec=10)

    # Fallback: if audio is too short for chunking, use the cleaned file directly
    if not chunks:
        chunks = [clean_path]

    # Step 3: generate
    tts.tts_to_file(
        text=text,
        speaker_wav=chunks,          # list of refs → better voice capture
        language=language,
        file_path=output_path,
        split_sentences=True          # handles long texts more naturally
    )

    return output_path