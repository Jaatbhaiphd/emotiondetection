import librosa
import numpy as np
import pickle

with open("app/model.pkl", "rb") as f:
    model = pickle.load(f)

def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=22050)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    return mfccs_scaled.reshape(1, -1)

def predict_emotion(audio_path):
    features = extract_features(audio_path)
    prediction = model.predict(features)
    return prediction[0]