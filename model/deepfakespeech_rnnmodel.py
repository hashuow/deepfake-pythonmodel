import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Masking
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# === Load Spectrograms ===
fake_folder = "E:/dataset/spectrograms_fake"   # <-- make sure path is correct
real_folder = "E:/dataset/spectrograms_real"    # <-- make sure path is correct

X = []
y = []

# Load fake spectrograms (label: 1)
for file in os.listdir(fake_folder):
    if file.endswith(".npy"):
        spec = np.load(os.path.join(fake_folder, file))
        X.append(spec.T)  # Transpose (time_steps, 128)
        y.append(1)

# Load real spectrograms (label: 0)
for file in os.listdir(real_folder):
    if file.endswith(".npy"):
        spec = np.load(os.path.join(real_folder, file))
        X.append(spec.T)
        y.append(0)

# Convert lists to numpy arrays
X = np.array(X, dtype=object)
y = np.array(y)

print(f"✅ Total samples: {len(X)} (Real: {y.tolist().count(0)}, Fake: {y.tolist().count(1)})")

# === Pad Sequences ===
MAX_FRAMES = 400
X_padded = pad_sequences(X, maxlen=MAX_FRAMES, padding='post', dtype='float32')

print(f"✅ Padded shape: {X_padded.shape}")
print(f"✅ Labels shape: {y.shape}")

# === Split into Training and Test Sets ===
X_train, X_test, y_train, y_test = train_test_split(
    X_padded, y, test_size=0.2, random_state=42, stratify=y
)

print(f"🔹 Train samples: {len(X_train)}")
print(f"🔹 Test samples: {len(X_test)}")

# === Build Improved RNN Model ===
model = Sequential([
    Masking(mask_value=0.0, input_shape=(MAX_FRAMES, 128)),
    GRU(128, return_sequences=True),
    GRU(64, return_sequences=False),
    Dropout(0.4),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

# === Compile the Model ===
model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# === Train the Model ===
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=20,              # 🔥 20 Epochs
    batch_size=32,
    verbose=1
)

print("✅ Training complete!")

# === Save the Model ===
model.save("deepfake_rnn_model.h5")
print("💾 Improved model saved as deepfake_rnn_model.h5")
