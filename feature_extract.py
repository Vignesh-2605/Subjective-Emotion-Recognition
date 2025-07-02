import numpy as np
import librosa
import soundfile

def extract_feature(file_name):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        stft = np.abs(librosa.stft(X))

        # MFCC
        mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40)
        mfccs_mean = np.mean(mfccs.T, axis=0)

        # MFCC Delta
        mfccs_delta = librosa.feature.delta(mfccs)
        mfccs_delta_mean = np.mean(mfccs_delta.T, axis=0)

        # MFCC Delta-Delta
        mfccs_delta2 = librosa.feature.delta(mfccs, order=2)
        mfccs_delta2_mean = np.mean(mfccs_delta2.T, axis=0)

        # Chroma
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)

        # Mel-spectrogram
        mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)

        # Zero Crossing Rate
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=X).T, axis=0)

        # Spectral Contrast
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)

        # RMS Energy
        rms = np.mean(librosa.feature.rms(y=X).T, axis=0)

        # Pitch (Fundamental frequency)
        pitches, magnitudes = librosa.piptrack(y=X, sr=sample_rate)
        pitch_values = pitches[magnitudes > np.median(magnitudes)]
        if pitch_values.size > 0:
            pitch_mean = np.mean(pitch_values)
            pitch_std = np.std(pitch_values)
        else:
            pitch_mean = 0
            pitch_std = 0

        return np.hstack((
            mfccs_mean,
            mfccs_delta_mean,
            mfccs_delta2_mean,
            chroma,
            mel,
            zcr,
            contrast,
            rms,
            [pitch_mean, pitch_std]
        ))
