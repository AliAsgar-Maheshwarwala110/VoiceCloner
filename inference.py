import numpy as np
import librosa
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# SAME MODEL ARCHITECTURE (must match training)
def build_model(input_shape):
    model = Sequential()
    model.add(Dense(256, activation='relu', input_shape=(input_shape,)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    return model

# Load model
input_shape = 10000   # ⚠️ replace with your actual feature size
model = build_model(input_shape)
model.load_weights("model_weights.h5")

# Feature extraction (copy EXACT from notebook)
def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    return mfcc.flatten()

# Padding (same as notebook)
def pad_features(features, max_len=10000):
    if len(features) < max_len:
        return np.pad(features, (0, max_len - len(features)))
    else:
        return features[:max_len]

# Prediction
def predict_audio(file_path):
    features = extract_features(file_path)
    features = pad_features(features)
    features = features.reshape(1, -1)

    prediction = model.predict(features)[0]
    label = np.argmax(prediction)

    print("Prediction:", "REAL" if label == 1 else "FAKE")
    print("Confidence:", prediction[label])

# Test
predict_audio("test.wav")