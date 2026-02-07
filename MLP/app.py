import streamlit as st
import pickle
import numpy as np
import os
from feature_extractor import extract_feature
import librosa
import soundfile

# File paths for model and scaler
MODEL_PATH = "emotion_model.pkl"
SCALER_PATH = "scaler.pkl"

# App title
st.set_page_config(page_title="Emotion Recognition from Voice", page_icon="🎤", layout="centered")
st.title("🎤 Emotion Recognition from Voice")

# Sidebar navigation
menu = ["Home", "Train Model", "Predict Emotion"]
choice = st.sidebar.radio("Navigation", menu)

# Home Page
if choice == "Home":
    st.subheader("Welcome to the Emotion Recognition App")
    st.write("""
        This app recognizes human emotions from voice recordings using an MLP model.
        You can:
        - Train the model with your dataset.
        - Upload a `.wav` file to predict the emotion.
    """)
    st.image("https://cdn-icons-png.flaticon.com/512/2541/2541988.png", width=150)

# Train Model Page
elif choice == "Train Model":
    st.subheader("Train the Emotion Recognition Model")
    if st.button("Start Training"):
        with st.spinner("Training in progress... This may take a while."):
            import training_model  # Runs your training script
        st.success("✅ Training completed! Model saved as 'emotion_model.pkl' and scaler as 'scaler.pkl'.")

# Predict Emotion Page
elif choice == "Predict Emotion":
    st.subheader("Predict Emotion from Audio")
    uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")

        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            with open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)
            with open(SCALER_PATH, "rb") as f:
                scaler = pickle.load(f)

            # Save temp file
            temp_path = "temp_audio.wav"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Extract features
            features = extract_feature(temp_path)
            features_scaled = scaler.transform([features])

            # Predict
            prediction = model.predict(features_scaled)[0]
            st.success(f"🎯 Predicted Emotion: **{prediction}**")

        else:
            st.error("❌ Model not found. Please train the model first.")
