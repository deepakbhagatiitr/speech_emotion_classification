# Audio Emotion Classification Project

## Project Overview
This project develops a deep learning model to classify emotions from audio files, specifically using the RAVDESS dataset containing 1440 speech files and 1012 song files from 24 professional actors. The model predicts 7 emotions (neutral, calm, happy, sad, angry, fearful, disgust) and is trained on GPU using a Convolutional Neural Network (CNN) with Mel-Frequency Cepstral Coefficients (MFCCs) and additional audio features. The goal is to achieve an F1 score > 80%, overall accuracy > 80%, and class-wise accuracy > 75% for all emotions.

## Preprocessing & Modeling Approach
- **Data Preprocessing**:
  - Extracts MFCCs, delta MFCCs, chroma, mel spectrograms, pitch-shifted MFCCs, noisy MFCCs, and intensity-adjusted MFCCs from audio files.
  - Applies SpecAugment for data augmentation (time and frequency masking).
  - Normalizes features using StandardScaler and balances classes with SMOTE for sad and fearful emotions.
- **Modeling**:
  - Uses a CNN with 2 Conv1D layers (64, 128 filters), batch normalization, and dropout (0.5, 0.4) to prevent overfitting.
  - Employs a weighted hybrid loss (categorical cross-entropy + focal loss) with class weights to handle imbalance.
  - Trains with Adam optimizer, learning rate scheduling, and early stopping on GPU.

## Accuracy & Metrics
- **Target Metrics**: F1 score > 80%, accuracy > 80%, class-wise accuracy > 75%.
- **Current Performance**: [Update with latest results, e.g., F1: 0.81, Accuracy: 0.81, Neutral: 0.66, Calm: 0.94, Happy: 0.86, Sad: 0.68, Angry: 0.89, Fearful: 0.80, Disgust: 0.75].
- **Evaluation**: Uses confusion matrix, F1 score, and accuracy, plotted with seaborn.

## How to Run the Code
1. **Setup Environment**:
   - Run the notebook on Kaggle with GPU accelerator enabled.
   - Install dependencies: `!pip install scikit-learn==1.2.2 imbalanced-learn==0.10.1 tensorflow`.
2. **Prepare Dataset**:
   - Upload the `audio-data` dataset to Kaggle with `Audio_Speech_Actors_01-24` and `Audio_Song_Actors_01-24` folders.
3. **Execute Notebook**:
   - Run the setup cell first, then the main cell to train the model.
   - Check output for GPU detection, file processing, and validation metrics.
4. **Inference**:
   - Use `inference.py` with a test audio file path after training.
   - Example: `python inference.py` with `test_audio_path` updated.
5. **Web App**:
   - Follow Streamlit setup below to host the app.

## Dependencies
- Python 3.x
- Libraries: numpy, pandas, librosa, scikit-learn, tensorflow, imbalanced-learn, seaborn, matplotlib, joblib