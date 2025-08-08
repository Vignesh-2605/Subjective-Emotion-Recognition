## 🎤 Emotion Recognition from Voice

A **Streamlit web application** that detects **human emotions from voice recordings** using **audio feature extraction** and an **MLPClassifier**.  
Supports **training** with your dataset and **real-time emotion prediction** from uploaded `.wav` files.


## 📂 Project Structure
📁 Emotion-Recognition
|── 📁 Speech Data # Contains the audio files in .wav format
├── feature_extractor.py # Extracts audio features from .wav files
├── training_model.py # Trains the emotion recognition model
├── predict_emotion.py # Old CLI-based prediction
├── app.py # Streamlit app integrating training & prediction
├── emotion_model.pkl # Saved trained model (generated after training)
├── scaler.pkl # Saved StandardScaler (generated after training)
└── README.md # Project documentation


## ✨ Features
- **Feature Extraction**: MFCC, Delta, Delta-Delta, Chroma, Mel-Spectrogram, ZCR, Spectral Contrast, RMS, Pitch
- **Data Augmentation**: Pitch shifting, time stretching
- **Model Training**: Multi-Layer Perceptron (MLP) using scikit-learn
- **Interactive Web UI**:
  - Train the model in-browser
  - Upload `.wav` file for prediction
  - Play audio and view prediction instantly


## 📦 Installation
1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/emotion-recognition.git
   cd emotion-recognition
2. **Install dependencies**
pip install -r requirements.txt
requirements.txt
- streamlit
- librosa
- soundfile
- scikit-learn
- matplotlib
- seaborn
- playsound
- numpy
▶️ Usage
1️⃣ Train the Model
Place your dataset in .wav format (e.g., RAVDESS dataset) and update the dataset path in training_model.py.
Run the app:
- streamlit run app.py
Go to Train Model page → Click Start Training.
2️⃣ Predict Emotion
Navigate to Predict Emotion page.
Upload a .wav file.
The app will:
-> Play the uploaded audio.
-> Display the predicted emotion.


🛠 Tech Stack
Python 3
Streamlit
Scikit-learn
Librosa
NumPy
Matplotlib
Seaborn

📜 License
This project is licensed under the MIT License.

👨‍💻 Author
Vignesh S
B.Tech AI&ML @ Saveetha School of Engineering
📧 vickymsd3157@gmail.com
