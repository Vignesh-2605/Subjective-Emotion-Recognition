import os
import glob
import numpy as np
import tensorflow as tf
import librosa
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from cnn_feature_extractor import (
    extract_spectrogram,
    extract_spectrogram_from_array,
    augment_audio
)

# ---------------- CONFIG ----------------
DATASET_PATH = r"C:\Users\Vignesh S\OneDrive\Documents\Vignesh_S\Projects\Emotion recognition\Speech-data\Actor_*/*.wav"
EMOTIONS = {
    '01':'neutral','02':'calm','03':'happy','04':'sad',
    '05':'angry','06':'fearful','07':'disgust','08':'surprised'
}
SR = 22050
# ----------------------------------------

X = []
y = []

files = glob.glob(DATASET_PATH)
print("Audio files detected:", len(files))

for file in files:
    emotion = EMOTIONS[os.path.basename(file).split("-")[2]]

    # original
    img, _ = extract_spectrogram(file)
    X.append(img)
    y.append(emotion)

    # augmentation
    audio, sr = librosa.load(file, sr=SR)
    augmented = augment_audio(audio, sr)

    for aug in augmented:
        img_aug = extract_spectrogram_from_array(aug, sr)
        X.append(img_aug)
        y.append(emotion)

X = np.array(X, dtype="float32")
y = np.array(y)

print("X shape:", X.shape)
print("y shape:", y.shape)

# Encode labels
le = LabelEncoder()
y_enc = le.fit_transform(y)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.25, stratify=y_enc, random_state=42
)

# ---------------- CNN MODEL ----------------

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(128,128,1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128,(3,3),activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(256,(3,3),activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256,activation='relu'),
    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Dense(8,activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Early stopping
callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=6,
    restore_best_weights=True
)

# Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=40,
    batch_size=32,
    callbacks=[callback]
)

# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print("Final CNN Accuracy:", acc * 100)

# Save
model.save("cnn_emotion_model.h5")
pickle.dump(le, open("label_encoder.pkl","wb"))

print("Model and label encoder saved.")
