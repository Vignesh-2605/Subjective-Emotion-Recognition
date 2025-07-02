import librosa
import soundfile
import os, glob, shutil, tempfile
import numpy as np
import tkinter as tk
from tkinter import filedialog
from playsound import playsound
import time
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Extract features (mfcc, chroma, mel, zcr, contrast, tonnetz)
def extract_feature(file_name, mfcc, chroma, mel, zcr=True, contrast=True, tonnetz=True):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        result = np.array([])
        if chroma or contrast:
            stft = np.abs(librosa.stft(X))
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma_feat = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma_feat))
        if mel:
            mel_spec = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel_spec))
        if zcr:
            zcr_feat = np.mean(librosa.feature.zero_crossing_rate(X).T, axis=0)
            result = np.hstack((result, zcr_feat))
        if contrast:
            contrast_feat = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, contrast_feat))
        if tonnetz:
            tonnetz_feat = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
            result = np.hstack((result, tonnetz_feat))
        return result

# Emotions in the dataset
emotions = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}
observed_emotions = ['calm', 'happy', 'fearful', 'disgust']

# Load the dataset and extract features
def load_data(test_size=0.25):
    x, y = [], []
    for file in glob.glob("C:\\Users\\Vignesh S\\OneDrive\\Documents\\Vignesh S\\Projects\\Speech-data\\Actor_\\.wav"):
        file_name = os.path.basename(file)
        emotion = emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            continue
        feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)
    return train_test_split(np.array(x), y, test_size=test_size, random_state=42)

# Load and prepare data
x_train, x_test, y_train, y_test = load_data(test_size=0.25)

# Normalize features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Print data info
print((x_train.shape[0], x_test.shape[0]))
print(f'Features extracted: {x_train.shape[1]}')

# Build and train model
model = MLPClassifier(
    alpha=0.001,
    batch_size=64,
    epsilon=1e-08,
    hidden_layer_sizes=(512, 256, 128),
    learning_rate='adaptive',
    max_iter=1000
)
model.fit(x_train, y_train)

# Evaluate
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
print("Model trained with {:.2f}% accuracy".format(accuracy * 100))

# File picker GUI
def choose_audio_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select a .wav file",
        filetypes=[("WAV files", "*.wav")]
    )
    root.destroy()
    return file_path

# Play audio via playsound using a short temp path
def play_audio(file_path):
    try:
        tmp_path = os.path.join(tempfile.gettempdir(), "temp_audio.wav")
        shutil.copy(file_path, tmp_path)
        playsound(tmp_path)
    except Exception as e:
        print(f"Audio playback error: {e}")

# Predict emotion from user-chosen file
file_path = choose_audio_file()
if file_path:
    print(f"\nSelected file: {file_path}")
    try:
        print("Playing audio...")
        play_audio(file_path)
        time.sleep(1)
        features = extract_feature(file_path, mfcc=True, chroma=True, mel=True)
        features = features.reshape(1, -1)
        features = scaler.transform(features)
        predicted_emotion = model.predict(features)[0]
        print(f"Predicted Emotion: {predicted_emotion}")
    except Exception as e:
        print(f"Error processing the file: {e}")
else:
    print("No file selected.")
    
