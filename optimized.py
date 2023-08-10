import os
import numpy as np
import librosa
from hmmlearn import hmm
from pydub import AudioSegment
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import librosa.display
import joblib


def extract_mfcc_delta_delta(audio_signal, samplerate=16000):
    mfcc_features = librosa.feature.mfcc(
        y=audio_signal, sr=samplerate, n_mfcc=13)
    delta_features = librosa.feature.delta(mfcc_features, order=2)
    return np.vstack((mfcc_features, delta_features))


def train_hmm_model(data, n_components=3, n_iter=1000):
    model = hmm.GaussianHMM(n_components=n_components,
                            n_iter=n_iter, covariance_type='diag', verbose=True)

    # Prepare the lengths parameter: the length of each sequence
    lengths = [x.shape[1] for x in data]

    # Concatenate all sequences along the time axis (axis=1)
    data_concatenated = np.concatenate(data, axis=1)

    # Transpose to fit the model
    model.fit(data_concatenated.T, lengths=lengths)
    return model


def hmm_predict(model, mfcc_features):
    return model.score(mfcc_features.T)  # Transpose the mfcc_features


def combine_predictions(mfcc_prediction, hmm_prediction, threshold=0.0):
    return mfcc_prediction >= threshold and hmm_prediction >= threshold


def load_data(dir_path):
    X = []
    y = []
    class_names = os.listdir(dir_path)

    for class_name in class_names:
        class_dir = os.path.join(dir_path, class_name)
        filenames = os.listdir(class_dir)
        for filename in filenames:
            if filename.endswith(".wav"):
                file_path = os.path.join(class_dir, filename)

                print(f'Processing file: {file_path}')  # Debugging line

                # Load audio file
                audio_signal, sr = librosa.load(file_path, sr=None)

                mfcc_delta_delta_feature = extract_mfcc_delta_delta(
                    audio_signal, samplerate=sr)
                # Debugging line
                print(f'MFCC features shape: {mfcc_delta_delta_feature.shape}')

                X.append(mfcc_delta_delta_feature)
                y.append(class_name)

    return X, y, class_names


def plot_mfcc(mfcc_feature, class_name):
    print(f"MFCC feature shape: {mfcc_feature.shape}")  # Debugging line
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc_feature, x_axis='time')
    plt.colorbar()
    plt.title(f'MFCC (with delta-delta) of class {class_name}')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    X, y, class_names = load_data('Cleaned')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    hmm_models = {}
    for class_index, class_name in enumerate(class_names):
        # Select only the data corresponding to the current class
        class_data = [x for x, label in zip(
            X_train, y_train) if label == class_name]

        # Each sequence (i.e., audio file) is treated as separate; no need for padding or truncating
        hmm_model = train_hmm_model(class_data)
        hmm_models[class_name] = hmm_model

        # Average the MFCC coefficients of each file independently along the time axis (frames)
        averaged_class_data = [
            np.mean(x, axis=1, keepdims=True) for x in class_data]

        # Now we can concatenate them along the time axis (axis=1)
        averaged_class_data_stacked = np.hstack(averaged_class_data)

        # Save the trained HMM model
        model_save_path = f"{class_name}_hmm_model.pkl"
        joblib.dump(hmm_model, model_save_path)
        print(
            f"Trained HMM model for class '{class_name}' saved to {model_save_path}")

        # Plot the average MFCC coefficients
        plot_mfcc(averaged_class_data_stacked, class_name)

    # Load the saved HMM models for predictions
    loaded_hmm_models = {}
    for class_name in class_names:
        model_save_path = f"{class_name}_hmm_model.pkl"
        loaded_hmm_models[class_name] = joblib.load(model_save_path)
        print(
            f"HMM model for class '{class_name}' loaded from {model_save_path}")

    # Predict with HMM models
    results = []
    for i, test_feature in enumerate(X_test):
        scores = {class_name: hmm_predict(
            model, test_feature) for class_name, model in hmm_models.items()}
        predicted_class = max(scores, key=scores.get)
        results.append((y_test[i], predicted_class))

    # Print prediction results
    print("Prediction results:")
    for true_class, predicted_class in results:
        print(f"True class: {true_class}, Predicted class: {predicted_class}")
