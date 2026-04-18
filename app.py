# app.py

import streamlit as st
import os
from clone import clone_voice
from similarity import cosine_similarity_score
import matplotlib.pyplot as plt
from audiorecorder import audiorecorder

st.set_page_config(page_title="Voice Cloner", layout="centered")
st.title("🎙️ Voice Cloner + Similarity")

# ---------------------------
# Setup folders
# ---------------------------
os.makedirs("samples", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

input_path = "samples/input.wav"

# ---------------------------
# Input method
# ---------------------------
option = st.radio("Choose input method:", ["Upload Audio", "Record Audio"])

audio_ready = False

# ---------------------------
# Upload audio
# ---------------------------
if option == "Upload Audio":
    uploaded_file = st.file_uploader("Upload WAV file", type=["wav"])

    if uploaded_file:
        with open(input_path, "wb") as f:
            f.write(uploaded_file.read())

        st.audio(input_path)
        audio_ready = True

        st.info(
            "✅ Audio received. It will be automatically preprocessed "
            "(resampled → silence trimmed → normalized) before cloning."
        )

# ---------------------------
# Record audio
# ---------------------------
if option == "Record Audio":
    st.write("Click to start/stop recording:")
    audio = audiorecorder("Start Recording", "Stop Recording")

    if len(audio) > 0:
        st.success("Recording complete!")
        audio.export(input_path, format="wav")
        st.audio(input_path)
        audio_ready = True

        st.info(
            "✅ Recording saved. It will be automatically preprocessed "
            "(resampled → silence trimmed → normalized) before cloning."
        )

# ---------------------------
# Tips for best results
# ---------------------------
with st.expander("💡 Tips for best similarity score"):
    st.markdown("""
    - Use **at least 10–30 seconds** of clean audio for better voice capture
    - Record/upload in a **quiet environment** (low background noise)
    - Speak **clearly and naturally** — avoid whispering or shouting
    - The model automatically picks the **clearest segments** from your audio
    """)

# ---------------------------
# Text input
# ---------------------------
text = st.text_area("Enter text to synthesize", "Hello, this is my cloned voice.")
language = st.selectbox("Language", ["en", "hi", "fr", "de"])

# ---------------------------
# Clone voice
# ---------------------------
if st.button("Clone Voice"):
    if not audio_ready:
        st.error("Please provide audio first!")
    else:
        output_path = "outputs/output.wav"

        with st.spinner("Preprocessing audio and cloning voice..."):
            clone_voice(text, input_path, output_path, language)

        st.subheader("🔊 Cloned Voice")
        st.audio(output_path)

        # ---------------------------
        # Similarity
        # ---------------------------
        with st.spinner("Computing similarity..."):
            sim = cosine_similarity_score(input_path, output_path)

        st.subheader("📊 Cosine Similarity (Speaker Embedding)")
        st.write(f"Similarity Score: **{sim:.4f}**")

        # Plot
        fig, ax = plt.subplots(figsize=(4, 3))
        color = "#2ecc71" if sim > 0.60 else ("#f39c12" if sim > 0.40 else "#e74c3c")
        ax.bar(["Similarity"], [sim], color=color)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Score")
        ax.axhline(y=0.60, color="green", linestyle="--", linewidth=0.8, label="High (0.60)")
        ax.axhline(y=0.40, color="orange", linestyle="--", linewidth=0.8, label="Moderate (0.40)")
        ax.legend(fontsize=8)
        st.pyplot(fig)

        # ---------------------------
        # Interpretation
        # NOTE: ECAPA embeddings cosine scores typically sit between 0.5–1.0
        # for same-speaker pairs, so thresholds are adjusted accordingly.
        # ---------------------------
        if sim > 0.60:
            st.success("🔥 Very high similarity — voice cloned well!")
        elif sim > 0.40:
            st.warning("⚠️ Moderate similarity — try using longer/cleaner audio")
        else:
            st.error("❌ Low similarity — check audio quality or record in a quieter environment")