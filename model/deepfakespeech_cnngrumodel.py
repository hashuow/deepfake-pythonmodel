# Step-by-step Local Environment Setup for CNN+GRU Model

# 1️⃣ Install required packages in your local terminal:
# pip install pydub torchaudio torch torchvision requests
# brew install ffmpeg
import os
os.environ["PATH"] += os.pathsep + "/opt/homebrew/bin"
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from pydub import AudioSegment
import os
import glob
import random
import torch
import torchaudio
import librosa

from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import requests

AudioSegment.converter = "/opt/homebrew/bin/ffmpeg"
AudioSegment.ffprobe = "/opt/homebrew/bin/ffprobe"



# Paths (Local)
BASE_DIR = os.path.expanduser("~/cnn_gru_project")
REAL_SPEECH_PATH = os.path.join(BASE_DIR, "dataset/real_speech/dev-clean")
FAKE_MP3_PATH = os.path.join(BASE_DIR, "dataset/fake_speech_mp3")
FAKE_FLAC_PATH = os.path.join(BASE_DIR, "dataset/fake_speech_flac")

os.makedirs(FAKE_MP3_PATH, exist_ok=True)
os.makedirs(FAKE_FLAC_PATH, exist_ok=True)


# Load real and fake files
real_files = glob.glob(os.path.join(REAL_SPEECH_PATH, '**/*.flac'), recursive=True)
fake_files = glob.glob(os.path.join(FAKE_FLAC_PATH, '*.flac'))

# Sample files for balanced dataset
real_files = random.sample(real_files, min(len(real_files), 200))
fake_files = random.sample(fake_files, min(len(fake_files), 200))

all_files = real_files + fake_files
all_labels = [0]*len(real_files) + [1]*len(fake_files)

# Dataset Definition
class AudioDataset(Dataset):
    def __init__(self, files, labels):
        self.files = files
        self.labels = labels
        self.mel_transform = torch.nn.Sequential(
            MelSpectrogram(sample_rate=16000, n_mels=128),
            AmplitudeToDB()
        )

    def __getitem__(self, idx):

        audio, sr = librosa.load(self.files[idx], sr=16000)
        waveform = torch.tensor(audio).unsqueeze(0)
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)

        mel = self.mel_transform(waveform)
        if mel.shape[2] < 224:
            mel = torch.nn.functional.pad(mel, (0, 224 - mel.shape[2]))
        else:
            mel = mel[:, :, :224]

        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return mel, label

    def __len__(self):
        return len(self.files)

# Split Data
train_files, val_files, train_labels, val_labels = train_test_split(
    all_files, all_labels, test_size=0.2, stratify=all_labels, random_state=42)

train_dataset = AudioDataset(train_files, train_labels)
val_dataset = AudioDataset(val_files, val_labels)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Model Definition
import torch.nn as nn

class CNN_GRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2,2))
        )
        self.gru = nn.GRU(32 * 32, 64, num_layers=2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 3, 1, 2).reshape(x.size(0), x.size(3), -1)
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

# Training setup
model = CNN_GRU()
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training Loop
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

def train(model, epochs):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for mel, labels in train_loader:
            mel, labels = mel.to(device), labels.to(device).unsqueeze(1)
            outputs = model(mel)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Accuracy: convert logits to predicted class (0 or 1)
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total * 100
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.2f}%")
def evaluate_model(model, val_loader):
    model.eval()
    y_true, y_pred, y_scores = [], [], []

    with torch.no_grad():
        for mel, labels in val_loader:
            mel, labels = mel.to(device), labels.to(device).unsqueeze(1)
            outputs = model(mel)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_scores.extend(probs.cpu().numpy())

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=["Real", "Fake"], digits=4)
    cm = confusion_matrix(y_true, y_pred)

    # EER (Equal Error Rate)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.abs(fnr - fpr))]
    eer = fpr[np.nanargmin(np.abs(fnr - fpr))]

    # Print results
    print("\n✅ Model Evaluation Results:")
    print("--------------------------------------------------")
    print(f"Test Accuracy: {round(acc * 100, 2)}%")
    print(f"Equal Error Rate (EER): {round(eer * 100, 2)}%")
    print("\n🔹 Classification Report:\n", report)
    print("Confusion Matrix (Test Set):")
    print("                | Predicted Real       | Predicted Fake")
    print("------------------------------------------------------------")
    print(f"Actual Real     | {cm[0][0]:<22} | {cm[0][1]:<18}")
    print(f"Actual Fake     | {cm[1][0]:<22} | {cm[1][1]:<18}")
    print("------------------------------------------------------------")

    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

# Run training
train(model, epochs=10)
evaluate_model(model, val_loader)

# Prediction Function
def predict(model, file_path):
    model.eval()

    # Use librosa instead of torchaudio to load the file
    audio, sr = librosa.load(file_path, sr=16000)
    waveform = torch.tensor(audio).unsqueeze(0)  # (1, N)

    mel = train_dataset.mel_transform(waveform)
    if mel.shape[2] < 224:
        mel = torch.nn.functional.pad(mel, (0, 224 - mel.shape[2]))
    else:
        mel = mel[:, :, :224]

    mel = mel.unsqueeze(0).to(device)

    with torch.no_grad():
        output = torch.sigmoid(model(mel)).item()

    prediction = "FAKE" if output > 0.5 else "REAL"
    print(f"File: {file_path}\nPrediction: {prediction} ({output:.2f})")

# Example prediction
predict(model, "/Users/elizabethtom/cnn_gru_project/dataset/fake_speech_flac/aadi.flac")

