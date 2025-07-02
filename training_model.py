import os
import glob
import numpy as np
import librosa
import soundfile
import pickle
import warnings
import matplotlib.pyplot as plt
import librosa.display
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from feature_extractor import extract_feature  # Import from shared feature_extractor.py

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
# Use all 8 emotions for completeness
observed_emotions = [
    'neutral',
    'calm',
    'happy',
    'sad',
    'angry',
    'fearful',
    'disgust',
    'surprised'
]

# Visualize a sample
def plot_sample(file_path):
    X, sr = librosa.load(file_path)
    plt.figure(figsize=(14, 6))

    # Waveform
    plt.subplot(3, 1, 1)
    plt.title("Waveform")
    librosa.display.waveshow(X, sr=sr)

    # Spectrogram with warning suppression
    plt.subplot(3, 1, 2)
    plt.title("Spectrogram")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plt.specgram(X + 1e-8, Fs=sr)

    # MFCC
    plt.subplot(3, 1, 3)
    plt.title("MFCC")
    mfccs = librosa.feature.mfcc(y=X, sr=sr, n_mfcc=13)
    librosa.display.specshow(mfccs, x_axis='time')

    plt.tight_layout()
    plt.show()

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

# Main training logic
print("Loading data and extracting features...")
x, y = load_data()

print("\nFeature matrix shape:", x.shape)
print("Label vector shape:", y.shape)
print("First 5 labels:", y[:5])

# Visualize the first sample audio file
sample_files = glob.glob(r"C:\Users\Vignesh S\OneDrive\Documents\Vignesh S\Projects\Emotion recognition\Speech-data\Actor_*/*.wav")
if sample_files:
    print("\nGenerating visualizations for the first sample...")
    plot_sample(sample_files[0])

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, stratify=y, random_state=42)
print("\nTrain samples:", x_train.shape[0], "Test samples:", x_test.shape[0])

# Feature scaling
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print("\nFeature vector length:", x_train.shape[1])

# Build and train model
print("\nTraining the model...")
model = MLPClassifier(
    alpha=0.00005,
    batch_size=16,
    epsilon=1e-08,
    hidden_layer_sizes=(1024, 512, 256, 128),
    learning_rate='adaptive',
    max_iter=5000
)
model.fit(x_train, y_train)

# Evaluate
y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)

# Real accuracy for logs
print(f"\nTraining completed. Test accuracy: {acc * 100:.2f}%")

# Save model and scaler
with open("emotion_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("\nModel and scaler saved successfully.")
