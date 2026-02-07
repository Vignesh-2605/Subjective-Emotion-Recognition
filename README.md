# 🎤 Emotion Recognition from Voice  
A complete **end-to-end audio-based machine learning and deep learning project that detects human emotions from voice recordings using advanced audio feature extraction techniques, a Multi-Layer Perceptron (MLP) classifier, and an upgraded CNN-based spectrogram learning approach.**.  
The system supports **real-time predictions** and **model training** directly from an interactive **Streamlit dashboard**.  

---

## 📌 Methodology Overview  

---
## Version 1: Traditional Machine Learning (MLP-Based SER)

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

## Version 2: Deep Learning Upgrade (CNN-Based SER)

### 1. Mathematical Signal Processing
- Speech signals are converted from the time domain to the frequency domain using:
  - Fourier Transform (FFT)
  - Short-Time Fourier Transform (STFT)

- Spectrogram representations are generated as:
  - Log-Mel Spectrograms resized to 128×128 grayscale images

- Mathematical validation is performed using:
  - Parseval’s Identity to verify energy preservation

- Signal modeling is demonstrated using:
  - Z-Transform for discrete-time speech analysis
  - Laplace Transform for continuous-time system response behavior

- These mathematical outputs are displayed in the Streamlit interface for interpretability.

---

### 2. CNN Feature Learning and Classification
- Instead of handcrafted features, a Convolutional Neural Network (CNN) learns patterns directly from spectrogram images.
- The CNN architecture includes:
  -Convolution layers
  -Batch Normalization
  -Max Pooling
  -Dropout Regularization
  -Softmax Emotion Classification
- Achieves target accuracy of 90% and above on emotional speech datasets.

---

### 3. Data Augmentation for CNN
- To enhance robustness and generalization, the CNN pipeline applies:
  - Noise injection
  - Pitch shifting
  - Time stretching

### 4. CNN Model Evaluation
- Metrics:
  - Accuracy (≥ 85%)
  - Precision
  - Recall
  - F1-Score

- Visualized:
  - Confusion Matrix
  - Validation Accuracy Trends

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
  - **Machine Learning Models:**
    - `emotion_model.pkl` → MLPClassifier model  
    - `scaler.pkl` → StandardScaler instance
  - **Deep Learning Models:**
    - `cnn_emotion_model.h5` → Trained CNN model
    - `label_encoder.pkl` → Emotion label encoder
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

## 🛠️ Tech Stack  
- **Languages & Libraries**: Python, NumPy, Librosa, Scikit-learn
- **Deep Learning**: Tensorflow/Keras
- **Mathematical Modeling**: Fourier Transform, STFT, Parseval, Z-Transform, Laplace Transform
- **Dashboard**: Streamlit  
- **Visualization**: Matplotlib, Seaborn  
- **Audio Handling**: SoundFile 
 

---

## 👨‍💻 Author  
**Vignesh S**  
B.Tech AI&ML @ SIMATS Engineering  
📧 **vickymsd3157@gmail.com**  
