import librosa#type: ignore
import soundfile#type: ignore
import os, glob
import numpy as np#type: ignore
import tkinter as tk
from tkinter import filedialog
from playsound import playsound#type: ignore
import time
from sklearn.model_selection import train_test_split #type: ignore
from sklearn.neural_network import MLPClassifier#type: ignore
from sklearn.metrics import accuracy_score#type: ignore

# Extract features (mfcc, chroma, mel) from a sound file
def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma_feat = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma_feat))
        if mel:
            mel_spec = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel_spec))
        return result

# Emotions in the RAVDESS dataset
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

# Emotions to observe
observed_emotions = ['calm', 'happy', 'fearful', 'disgust']

# Load the data and extract features for each sound file
def load_data(test_size=0.2):
    x, y = [], []
    for file in glob.glob("C:\\Users\\ooviy\\Desktop\\Vignesh\\speech-emotion-recognition-ravdess-data\\Actor_*\\*.wav"):
        file_name = os.path.basename(file)
        emotion = emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            continue
        feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)
    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)

# Split the dataset
x_train, x_test, y_train, y_test = load_data(test_size=0.25)

# Print shape of datasets and number of features
print((x_train.shape[0], x_test.shape[0]))
print(f'Features extracted: {x_train.shape[1]}')

# Initialize the MLP Classifier
model = MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08,
                      hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)

# Train the model
model.fit(x_train, y_train)

# Predict and calculate accuracy
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
print("Model trained with {:.2f}% accuracy".format(accuracy * 100))

# Function to choose file using dialog
def choose_audio_file():
    root = tk.Tk()
    root.withdraw()
    root.update()
    file_path = filedialog.askopenfilename(
        title="Select an audio file",
        filetypes=[("WAV files", "*.wav")]
    )
    root.destroy()
    return file_path

# Let user choose a file to test
file_path = choose_audio_file()

# Predict emotion from selected file
if file_path:
    print(f"\nSelected file: {file_path}")
    try:
        print("Playing audio...")
        playsound(file_path)
        time.sleep(1)
        features = extract_feature(file_path, mfcc=True, chroma=True, mel=True)
        features = features.reshape(1, -1)
        predicted_emotion = model.predict(features)[0]
        print(f"Predicted Emotion: {predicted_emotion}")
    except Exception as e:
        print(f"Error processing the file: {e}")
else:
    print("No file selected.")
