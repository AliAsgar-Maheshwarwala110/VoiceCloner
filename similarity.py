# similarity.py

import os
os.environ["SB_DISABLE_K2"] = "1"  # Windows fix

import torch
import torchaudio
import torch.nn.functional as F

# Compatible with older speechbrain versions (< 0.5.15)
try:
    from speechbrain.inference.speaker import EncoderClassifier
except ImportError:
    from speechbrain.pretrained import EncoderClassifier

EVAL_SR = 16000
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading speaker encoder...")
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    run_opts={"device": device},
    savedir="pretrained_models/spkrec-ecapa-voxceleb"
)


def load_and_prepare(audio_path: str) -> torch.Tensor:
    signal, fs = torchaudio.load(audio_path)

    if signal.shape[0] > 1:
        signal = torch.mean(signal, dim=0, keepdim=True)

    if fs != EVAL_SR:
        resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=EVAL_SR)
        signal = resampler(signal)

    return signal


def get_embedding(audio_path: str) -> torch.Tensor:
    signal = load_and_prepare(audio_path).to(device)

    with torch.no_grad():
        embedding = classifier.encode_batch(signal)

    embedding = embedding.squeeze()
    embedding = F.normalize(embedding, dim=0)
    return embedding


def cosine_similarity_score(file1: str, file2: str) -> float:
    emb1 = get_embedding(file1)
    emb2 = get_embedding(file2)
    sim = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0), dim=1)
    return float(sim.item())