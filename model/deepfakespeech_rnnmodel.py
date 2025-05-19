import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import GRU, Dense, Dropout, Masking
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam

# === Constants ===
SAMPLE_RATE = 22050
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048
MAX_FRAMES = 400
BATCH_SIZE = 32
EPOCHS = 20
MODEL_PATH = "deepfake_rnn_model.h5"

# === Paths ===
fake_folder = "E:/dataset/spectrograms_fake"
real_folder = "E:/dataset/spectrograms_real"
audio_folder = r"E:\dataset\fakee"

# === Load and Prepare Training Data ===
X, y = [], []

def load_spectrograms(folder, label):
    for file in os.listdir(folder):
        if file.endswith(".npy"):
            spec = np.load(os.path.join(folder, file))
            X.append(spec.T)
            y.append(label)

load_spectrograms(fake_folder, 1)
load_spectrograms(real_folder, 0)

X = np.array(X, dtype=object)
y = np.array(y)

X_padded = pad_sequences(X, maxlen=MAX_FRAMES, padding='post', dtype='float32')

X_train, X_test, y_train, y_test = train_test_split(
    X_padded, y, test_size=0.2, random_state=42, stratify=y
)

# === Build & Train Model ===
model = Sequential([
    Masking(mask_value=0.0, input_shape=(MAX_FRAMES, 128)),
    GRU(128, return_sequences=True),
    GRU(64, return_sequences=False),
    Dropout(0.4),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("🚀 Training model...")
model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1
)

model.save(MODEL_PATH)
print(f"✅ Model saved to {MODEL_PATH}")

# === Load Model for Prediction ===
model = load_model(MODEL_PATH)

# === Prediction Function ===
def predict_audio(file_path):
    try:
        y_audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        mel_spec = librosa.feature.melspectrogram(y=y_audio, sr=sr, n_fft=N_FFT,
                                                  hop_length=HOP_LENGTH, n_mels=N_MELS)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max).T
        padded = pad_sequences([mel_spec_db], maxlen=MAX_FRAMES, padding='post', dtype='float32')
        prediction = model.predict(padded, verbose=0)[0][0]
        return prediction
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return None

# === Batch Prediction ===
print("\n🔍 Starting batch prediction...")
total_files = 0
predicted_real = 0
predicted_fake = 0

for file_name in os.listdir(audio_folder):
    if file_name.endswith(".mp3") or file_name.endswith(".wav"):
        file_path = os.path.join(audio_folder, file_name)
        prediction = predict_audio(file_path)

        if prediction is not None:
            total_files += 1
            if prediction >= 0.5:
                predicted_fake += 1
                print(f"🛑 {file_name} → Predicted: FAKE (Confidence: {prediction:.2f})")
            else:
                predicted_real += 1
                print(f"✅ {file_name} → Predicted: REAL (Confidence: {1 - prediction:.2f})")

print("\n=== Testing Summary ===")
print(f"Total files tested: {total_files}")
print(f"Predicted Real: {predicted_real}")
print(f"Predicted Fake: {predicted_fake}")
