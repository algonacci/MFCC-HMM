import os
import numpy as np
import librosa
import joblib
import matplotlib.pyplot as plt

def hmm_predict(model, mfcc_features):
    return model.score(mfcc_features.T)


def load_trained_models(class_names):
    loaded_hmm_models = {}
    for class_name in class_names:
        model_save_path = f"{class_name}_hmm_model.pkl"
        loaded_hmm_models[class_name] = joblib.load(model_save_path)
        print(
            f"HMM model for class '{class_name}' loaded from {model_save_path}")
    return loaded_hmm_models


def extract_mfcc_delta_delta(audio_signal, samplerate=16000):
    mfcc_features = librosa.feature.mfcc(
        y=audio_signal, sr=samplerate, n_mfcc=13)
    delta_features = librosa.feature.delta(mfcc_features, order=2)
    return np.vstack((mfcc_features, delta_features))

def plot_mfcc(mfcc_features):
    plt.imshow(mfcc_features, origin='lower', aspect='auto')
    plt.ylabel('MFCC Coefficients')
    plt.xlabel('Time')
    plt.title('MFCC')
    plt.colorbar()
    plt.show()


def predict_audio_class(file_path):
    # Load audio file
    audio_signal, sr = librosa.load(file_path, sr=None)

    # Extract MFCC features for the audio signal
    mfcc_features = librosa.feature.mfcc(
        y=audio_signal, sr=sr, n_mfcc=13)
    mfcc_delta_delta_feature = extract_mfcc_delta_delta(
        audio_signal, samplerate=sr)

    print(f'MFCC features shape: {mfcc_delta_delta_feature.shape}')

    # Plot the MFCC
    plot_mfcc(mfcc_features)

    # Predict using HMM models
    scores = {class_name: hmm_predict(
        model, mfcc_delta_delta_feature) for class_name, model in hmm_models.items()}
    predicted_class = max(scores, key=scores.get)

    return predicted_class


if __name__ == '__main__':
    class_names = ['Al-Fatihah', 'Al-Ikhlas']
    hmm_models = load_trained_models(class_names)

    file_path = 'Testing/(fix) Rakin - Al Ikhlas.mp3'
    predicted_class = predict_audio_class(file_path)
    print(f"Predicted class: {predicted_class}")
