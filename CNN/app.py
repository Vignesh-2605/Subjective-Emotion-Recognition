import streamlit as st
import os
from cnn_predict import predict_emotion

st.set_page_config(page_title="CNN Emotion Recognition", page_icon="🎤")
st.title("🎤 CNN Based Speech Emotion Recognition (Applied Mathematics Integrated)")

uploaded_file = st.file_uploader("Upload WAV file", type=["wav"])

if uploaded_file:
    st.audio(uploaded_file)

    temp_path = "temp.wav"
    with open(temp_path,"wb") as f:
        f.write(uploaded_file.getbuffer())

    emotion, math_data = predict_emotion(temp_path)
    st.success(f"Predicted Emotion: {emotion}")
    st.subheader("Mathematical Signal Analysis Output")
    st.json(math_data)

