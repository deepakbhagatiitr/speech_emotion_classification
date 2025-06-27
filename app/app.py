
import streamlit as st
import librosa
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import os


st.set_page_config(page_title="Audio Emotion Classifier", layout="centered")


emotion_dict = {
    0: "neutral",
    1: "calm",
    2: "happy",
    3: "sad",
    4: "angry",
    5: "fearful",
    6: "disgust"
}


def extract_features(audio, sr):
    mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
    delta_mfccs = np.mean(librosa.feature.delta(mfccs), axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sr).T, axis=0)
    audio_pitch = librosa.effects.pitch_shift(audio, sr=sr, n_steps=1.2)
    mfccs_pitch = np.mean(librosa.feature.mfcc(y=audio_pitch, sr=sr, n_mfcc=40).T, axis=0)
    noise = np.random.normal(0, 0.005, audio.shape)
    audio_noisy = audio + noise
    mfccs_noisy = np.mean(librosa.feature.mfcc(y=audio_noisy, sr=sr, n_mfcc=40).T, axis=0)
    audio_intensity = audio * 1.2
    mfccs_intensity = np.mean(librosa.feature.mfcc(y=audio_intensity, sr=sr, n_mfcc=40).T, axis=0)
    return np.hstack([mfccs, delta_mfccs, chroma, mel, mfccs_pitch, mfccs_noisy, mfccs_intensity])


@st.cache_resource
def load_model_and_scaler():
    model_path = "emotion_classifier.h5"
    scaler_path = "scaler.pkl"
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        st.error("Model or scaler files not found. Please train the model first.")
        st.stop()
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

model, scaler = load_model_and_scaler()


st.title("Audio Emotion Classifier")
st.write("Upload an audio file to predict the emotion.")


uploaded_file = st.file_uploader("Choose an audio file...", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    
    
    audio, sr = librosa.load(uploaded_file, sr=22050)
    features = extract_features(audio, sr)
    features = features.reshape(1, -1)
    features_scaled = scaler.transform(features)
    features_cnn = features_scaled.reshape(features_scaled.shape[0], features_scaled.shape[1], 1)
    
    
    prediction = model.predict(features_cnn)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction) * 100
    
    
    st.success(f"Predicted Emotion: {emotion_dict[predicted_class]} (Confidence: {confidence:.2f}%)")

