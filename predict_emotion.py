import os
import pickle
import numpy as np
import soundfile
import librosa
import tkinter as tk
from tkinter import filedialog
import shutil
import tempfile
from playsound import playsound
from feature_extractor import extract_feature

# Load model and scaler
with open("emotion_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# File chooser
def choose_audio_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select a .wav file",
        filetypes=[("WAV files", "*.wav")]
    )
    return file_path

# Play audio
def play_audio(file_path):
    tmp_path = os.path.join(tempfile.gettempdir(), "temp_audio.wav")
    shutil.copy(file_path, tmp_path)
    playsound(tmp_path)

# Main prediction flow
file_path = choose_audio_file()
if file_path:
    print(f"Selected file: {file_path}")
    play_audio(file_path)
    features = extract_feature(file_path)
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)[0]
    print("\nPredicted Emotion:", prediction)
else:
    print("No file selected.")
