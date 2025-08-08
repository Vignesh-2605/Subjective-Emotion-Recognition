# 🎤 Emotion Recognition from Voice  

A complete **end-to-end audio-based machine learning project** that detects human emotions from voice recordings using advanced **audio feature extraction techniques** and a **Multi-Layer Perceptron (MLP) classifier**.  
The system supports **real-time predictions** and **model training** directly from an interactive **Streamlit dashboard**.  

---

## 📌 Methodology Overview  

### 1. Data Preparation  
- Utilized `.wav` audio files (e.g., RAVDESS dataset).  
- Explored and validated dataset structure.  
- Organized dataset into labeled emotion categories.  

### 2. Feature Engineering  
- Extracted features: **MFCC**, **Delta**, **Delta-Delta**, **Chroma**, **Mel-Spectrogram**, **Zero Crossing Rate**, **Spectral Contrast**, **RMS Energy**, **Pitch**.  
- Applied **data augmentation**:  
  - Pitch shifting (+2 semitones)  
  - Time stretching (1.1×)  

### 3. Train-Test Split  
- Performed stratified split (75% train / 25% test) to ensure balanced emotion distribution.  

### 4. Modeling  
- Trained **MLPClassifier** with multiple hidden layers.  
- Applied **StandardScaler** for feature normalization.  

### 5. Model Optimization  
- Tuned hyperparameters (hidden layers, learning rate, batch size).  
- Ensured high accuracy while avoiding overfitting.  

### 6. Model Evaluation  
- Metrics:  
  - Accuracy  
  - Precision  
  - Recall  
  - F1-Score  
- Visualized:  
  - Confusion Matrix (per emotion class)  
  - Classification Report  

### 7. Feature Importance  
- Analyzed feature contribution to prediction accuracy.  

### 8. Error Analysis  
- Investigated common misclassifications between similar emotions (e.g., calm vs. neutral).  

---

## 🖥️ Streamlit Dashboard  

The interactive dashboard enables:  
✅ **Model training** directly from the UI  
📂 **Audio upload** for real-time emotion prediction  
🎧 **Audio playback** before prediction  
📊 **Clean visualization** of predictions and training results  

---

### ▶️ To run the app:  
```bash
streamlit run app.py
```

---

## 🧠 Models & Persistence  
- All trained models are saved with preprocessing steps:  
  - `emotion_model.pkl` → MLPClassifier model  
  - `scaler.pkl` → StandardScaler instance  
- Metadata includes:  
  - Model parameters  
  - Accuracy score  
  - Timestamped versioning  

---

## 🔍 Prediction on New Audio  
- Upload a `.wav` file via the dashboard.  
- Pipeline applies the **exact same preprocessing** as training.  
- Outputs:  
  - **Predicted Emotion** (e.g., Happy, Angry, Sad)  
  - Confidence score (optional extension)  

---

## 📂 Project Structure  
```
📁 Emotion-Recognition
 ├── 📁 Speech Data           # Contains audio dataset in .wav format
 ├── feature_extractor.py     # Extracts audio features
 ├── training_model.py        # Trains the model
 ├── predict_emotion.py       # Old CLI-based prediction
 ├── app.py                   # Streamlit app
 ├── emotion_model.pkl        # Saved trained model
 ├── scaler.pkl               # Saved scaler
 └── README.md                # Project documentation
```

---

## 🛠️ Tech Stack  
- **Languages & Libraries**: Python, NumPy, Librosa, Scikit-learn  
- **Dashboard**: Streamlit  
- **Visualization**: Matplotlib, Seaborn  
- **Audio Handling**: SoundFile, Playsound  
 

---

## 👨‍💻 Author  
**Vignesh S**  
B.Tech AI&ML @ Saveetha School of Engineering  
📧 **vickymsd3157@gmail.com**  
