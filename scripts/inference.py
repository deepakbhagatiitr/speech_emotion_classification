# inference.py
import librosa
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import os
import tensorflow as tf

# Check GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(f"GPU detected: {gpus[0].name}")
    except RuntimeError as e:
        print(f"Error configuring GPU: {e}")
else:
    print("No GPU detected. Using CPU.")

# Define emotion dictionary (7 classes as per your code)
emotion_dict = {
    0: "neutral",
    1: "calm",
    2: "happy",
    3: "sad",
    4: "angry",
    5: "fearful",
    6: "disgust"
}

# Function to extract features
def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=22050)
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
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Load model and scaler
model_path = "/home/deepak-bhagat/software/speech_emotion_classification/models/emotion_classifier.h5"  # Update path if different
scaler_path = "/home/deepak-bhagat/software/speech_emotion_classification/models/scaler.pkl"  # Update path if different

if not os.path.exists(model_path) or not os.path.exists(scaler_path):
    raise FileNotFoundError("Model or scaler not found. Ensure training has been completed and files are saved.")

with tf.device('/GPU:0'):
    model = load_model(model_path)
scaler = joblib.load(scaler_path)

# Function to predict emotion
def predict_emotion(audio_file_path):
    # Extract features
    features = extract_features(audio_file_path)
    if features is None:
        return "Error processing audio"
    
    # Reshape and scale features
    features = features.reshape(1, -1)
    features_scaled = scaler.transform(features)
    features_cnn = features_scaled.reshape(features_scaled.shape[0], features_scaled.shape[1], 1)
    
    # Predict
    with tf.device('/GPU:0'):
        prediction = model.predict(features_cnn)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction) * 100
    
    return emotion_dict[predicted_class], confidence

# Example usage
if __name__ == "__main__":
    test_audio_path = "/home/deepak-bhagat/software/speech_emotion_classification/dataset/Audio_Song_Actors_01-24/Actor_01/03-02-01-01-01-01-01.wav"  # Replace with your test audio path
    if os.path.exists(test_audio_path):
        emotion, confidence = predict_emotion(test_audio_path)
        print(f"Predicted Emotion: {emotion} (Confidence: {confidence:.2f}%)")
    else:
        print("Test audio file not found. Please provide a valid path.")