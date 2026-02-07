import tensorflow as tf
import pickle
import numpy as np
from cnn_feature_extractor import extract_spectrogram

model = tf.keras.models.load_model("cnn_emotion_model.h5")
le = pickle.load(open("label_encoder.pkl", "rb"))

def predict_emotion(file_path):
    img, math_features = extract_spectrogram(file_path)
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    emotion = le.inverse_transform([np.argmax(pred)])[0]

    return emotion, math_features
