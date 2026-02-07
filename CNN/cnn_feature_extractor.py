import numpy as np
import librosa
import cv2
from math_transforms import (
    compute_fft,
    energy_time_domain,
    energy_freq_domain,
    z_transform,
    laplace_system_response
)

IMG_SIZE = 128
SR = 22050

def extract_spectrogram_from_array(y, sr=SR):
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=128, n_fft=2048, hop_length=512
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-9)

    img = cv2.resize(mel_db, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32").reshape(IMG_SIZE, IMG_SIZE, 1)

    return img

def extract_spectrogram(file_path):
    y, sr = librosa.load(file_path, sr=SR)

    # CNN image
    img = extract_spectrogram_from_array(y, sr)

    # Mathematical features
    freqs, mag, phase, X_fft = compute_fft(y, sr)
    E_time = energy_time_domain(y)
    E_freq = energy_freq_domain(X_fft)
    z_val = z_transform(y)
    lap_mean, lap_peak = laplace_system_response()

    math_features = {
        "Time-domain Energy": float(E_time),
        "Frequency-domain Energy": float(E_freq),
        "Energy Ratio (Parseval)": float(E_time / (E_freq + 1e-9)),
        "Z-Transform Value": float(z_val),
        "Laplace Mean Response": float(lap_mean),
        "Laplace Peak Response": float(lap_peak)
    }

    return img, math_features
def augment_audio(y, sr=SR):
    noise = y + 0.005 * np.random.randn(len(y))
    pitch = librosa.effects.pitch_shift(y, sr=sr, n_steps=2)
    stretch = librosa.effects.time_stretch(y, rate=1.1)
    return [noise, pitch, stretch]
