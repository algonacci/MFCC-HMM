import numpy as np
import librosa
from hmmlearn import hmm
import os

# Fungsi untuk menghitung MFCC bersama dengan delta-delta dari file audio


def extract_mfcc_delta_delta(audio_signal, samplerate=16000):
    mfcc_features = librosa.feature.mfcc(
        y=audio_signal, sr=samplerate, n_mfcc=13)
    delta_features = librosa.feature.delta(mfcc_features, order=2)
    return np.vstack((mfcc_features, delta_features))

# Fungsi untuk melatih model HMM dengan data pelatihan


def train_hmm_model(data, n_components=3, n_iter=100):
    model = hmm.GaussianHMM(n_components=n_components, n_iter=n_iter)
    model.fit(data)
    return model

# Fungsi untuk melakukan prediksi dengan model HMM


def hmm_predict(model, mfcc_features):
    return model.score(mfcc_features)

# Fungsi untuk menggabungkan hasil prediksi MFCC dan HMM


def combine_predictions(mfcc_prediction, hmm_prediction, threshold=0.0):
    return mfcc_prediction >= threshold and hmm_prediction >= threshold


# Nama folder kelas
class_names = ["Al-Fatihah", "Al-Ikhlas"]

# Daftar data pelatihan dan data uji
training_data = []
testing_data = []

# Data pelatihan
for class_name in class_names:
    class_dir = os.path.join(
        "C:/Users/Socio/Desktop/Bacaan Al-Qur_an/Cleaned", class_name)
    filenames = os.listdir(class_dir)[:-2]
    for filename in filenames:
        if filename.endswith(".wav"):
            file_path = os.path.join(class_dir, filename)
            audio_signal, sr = librosa.load(file_path, sr=None)
            mfcc_delta_delta_feature = extract_mfcc_delta_delta(
                audio_signal, samplerate=sr)
            training_data.append((class_name, mfcc_delta_delta_feature))

# Data uji (mengambil dua data terakhir dari masing-masing kelas)
for class_name in class_names:
    class_dir = os.path.join(
        "C:/Users/Socio/Desktop/Bacaan Al-Qur_an/Cleaned", class_name)
    filenames = os.listdir(class_dir)[-2:]
    for filename in filenames:
        if filename.endswith(".wav"):
            file_path = os.path.join(class_dir, filename)
            audio_signal, sr = librosa.load(file_path, sr=None)
            mfcc_delta_delta_feature = extract_mfcc_delta_delta(
                audio_signal, samplerate=sr)
            testing_data.append((class_name, mfcc_delta_delta_feature))

# Pad or truncate the MFCC features arrays to have the same dimension
max_length = max(len(mfcc_feature[0])
                 for _, mfcc_feature in training_data + testing_data)

for i, (_, mfcc_feature) in enumerate(training_data):
    pad_width = max_length - mfcc_feature.shape[1]
    training_data[i] = (training_data[i][0], np.pad(
        mfcc_feature, pad_width=((0, 0), (0, pad_width))))

for i, (_, mfcc_feature) in enumerate(testing_data):
    pad_width = max_length - mfcc_feature.shape[1]
    testing_data[i] = (testing_data[i][0], np.pad(
        mfcc_feature, pad_width=((0, 0), (0, pad_width))))

# Latih model HMM untuk masing-masing kelas
hmm_models = {}
for class_name in class_names:
    data = [data for label, data in training_data if label == class_name]
    data = np.concatenate(data)
    model = train_hmm_model(data)
    hmm_models[class_name] = model

# Prediksi dengan model HMM
results = []
for class_name, mfcc_feature in testing_data:
    hmm_predictions = {}
    for label, hmm_model in hmm_models.items():
        hmm_predictions[label] = hmm_predict(hmm_model, mfcc_feature)

    final_prediction = combine_predictions(
        hmm_predictions["Al-Fatihah"], hmm_predictions["Al-Ikhlas"], threshold=0.0)
    results.append((class_name, final_prediction))

# Cetak hasil prediksi
print("Hasil prediksi:")
for class_name, prediction in results:
    print(f"Data {class_name}: Prediksi {class_name} = {prediction}")
