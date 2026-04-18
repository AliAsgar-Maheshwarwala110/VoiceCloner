# utils.py

import torch
import torchaudio
import numpy as np
import librosa

# ✅ correct import (no k2 issue)
from speechbrain.inference.speaker import EncoderClassifier

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading speaker encoder...")

encoder = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    run_opts={"device": device}
)


def get_embedding(wav_path):
    signal, sr = torchaudio.load(wav_path)

    # mono
    if signal.shape[0] > 1:
        signal = signal.mean(dim=0, keepdim=True)

    # resample to 16k
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        signal = resampler(signal)

    with torch.no_grad():
        emb = encoder.encode_batch(signal.to(device))

    return emb.squeeze().cpu().numpy()


def load_audio_librosa(path):
    y, sr = librosa.load(path, sr=None)
    return y, sr