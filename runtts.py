from TTS.api import TTS

tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")

while True:
    text = input("Enter text: ")
    tts.tts_to_file(
        text=text,
        speaker_wav="voice.wav",
        language="en",
        file_path="output.wav",
        speed=1.2
    )