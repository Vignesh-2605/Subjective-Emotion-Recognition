import os, glob, shutil, tempfile
import numpy as np
import librosa
import soundfile
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Emotion mapping
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
# Consider focusing on clearer emotions
observed_emotions = ['calm', 'happy', 'sad', 'angry']

def extract_feature(file_name):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        stft = np.abs(librosa.stft(X))
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
        mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=X).T, axis=0)
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
        return np.hstack((mfccs, chroma, mel, zcr, contrast))

# Load data
def load_data():
    x, y = [], []
    for file in glob.glob(r"C:\Users\Vignesh S\OneDrive\Documents\Vignesh S\Projects\Emotion recognition\Speech-data\Actor_*/*.wav"):
        file_name = os.path.basename(file)
        emotion = emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            continue
        feature = extract_feature(file)
        x.append(feature)
        y.append(emotion)
    return np.array(x), np.array(y)

# Visualize a sample
def plot_sample(file_path):
    X, sr = librosa.load(file_path)
    plt.figure(figsize=(14,6))
    plt.subplot(3,1,1)
    plt.title("Waveform")
    librosa.display.waveshow(X, sr=sr)
    plt.subplot(3,1,2)
    plt.title("Spectrogram")
    plt.specgram(X, Fs=sr)
    plt.subplot(3,1,3)
    plt.title("MFCC")
    mfccs = librosa.feature.mfcc(y=X, sr=sr, n_mfcc=13)
    librosa.display.specshow(mfccs, x_axis='time')
    plt.tight_layout()
    plt.show()

# Main training logic
x, y = load_data()

print("Feature matrix shape:", x.shape)
print("Label vector shape:", y.shape)
print("First 5 labels:", y[:5])

# Visualize the first sample
plot_sample(glob.glob(r"C:\Users\Vignesh S\OneDrive\Documents\Vignesh S\Projects\Emotion recognition\Speech-data\Actor_*/*.wav")[0])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, stratify=y, random_state=42)
print("Train samples:", x_train.shape[0], "Test samples:", x_test.shape[0])

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = MLPClassifier(
    alpha=0.0001,
    batch_size=32,
    epsilon=1e-08,
    hidden_layer_sizes=(512, 256, 128),
    learning_rate='adaptive',
    max_iter=2000
)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print(f"Training completed. Test accuracy: {acc*100:.2f}%")

# Save model and scaler
with open("emotion_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Model and scaler saved successfully.")
