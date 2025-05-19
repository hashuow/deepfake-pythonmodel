#%% [Imports]
import torch
import torch.nn as nn
import librosa
import numpy as np
from transformers import Wav2Vec2FeatureExtractor, WavLMModel
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
import os
import time
import matplotlib.pyplot as plt
import warnings
import random
from tqdm import tqdm
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Gain, Shift

# Suppress matplotlib warnings on macOS
import matplotlib
matplotlib.use('Agg')

# Suppress transformers deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.utils.generic")

#%% [Dataset Class]
class FoRDataset(Dataset):
    def __init__(self, data_dir, split, feature_extractor, max_length=32000, augment=False):
        self.data_dir = os.path.join(data_dir, split)
        self.feature_extractor = feature_extractor
        self.max_length = max_length
        self.files = []
        self.labels = []
        self.augment = augment

        # Define augmentation pipeline
        if self.augment:
            self.augmentation = Compose([
                AddGaussianNoise(min_amplitude=0.005, max_amplitude=0.05, p=0.8),
                TimeStretch(min_rate=0.7, max_rate=1.3, p=0.6),
                PitchShift(min_semitones=-5, max_semitones=5, p=0.6),
                Gain(min_gain_db=-6, max_gain_db=6, p=0.5),
                Shift(min_shift=-0.2, max_shift=0.2, p=0.5)
            ])
        else:
            self.augmentation = None

        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Directory not found: {self.data_dir}")

        real_dir = os.path.join(self.data_dir, "real")
        if os.path.exists(real_dir):
            for file in os.listdir(real_dir):
                if file.endswith((".wav", ".mp3")):
                    audio_path = os.path.join(real_dir, file)
                    self.files.append(audio_path)
                    self.labels.append(0)

        fake_dir = os.path.join(self.data_dir, "fake")
        if os.path.exists(fake_dir):
            for file in os.listdir(fake_dir):
                if file.endswith((".wav", ".mp3")):
                    audio_path = os.path.join(fake_dir, file)
                    self.files.append(audio_path)
                    self.labels.append(1)

        if not self.files:
            raise ValueError(f"No WAV or MP3 files found in {self.data_dir}/real or {self.data_dir}/fake")

        # Balance dataset (oversample the minority class)
        bonafide_indices = [i for i, label in enumerate(self.labels) if label == 0]
        spoof_indices = [i for i, label in enumerate(self.labels) if label == 1]
        print(f"{split.capitalize()} - Before balancing - Bonafide: {len(bonafide_indices)}, Spoof: {len(spoof_indices)}")

        if len(bonafide_indices) == 0 or len(spoof_indices) == 0:
            raise ValueError(f"Cannot balance dataset for {split}: Bonafide samples = {len(bonafide_indices)}, Spoof samples = {len(spoof_indices)}.")

        if len(bonafide_indices) > len(spoof_indices):
            oversample_factor = len(bonafide_indices) // len(spoof_indices)
            oversampled_spoof_indices = spoof_indices * oversample_factor
            additional_spoof = len(bonafide_indices) - len(oversampled_spoof_indices)
            oversampled_spoof_indices.extend(random.sample(spoof_indices, additional_spoof))
            oversampled_indices = bonafide_indices + oversampled_spoof_indices
        else:
            oversample_factor = len(spoof_indices) // len(bonafide_indices)
            oversampled_bonafide_indices = bonafide_indices * oversample_factor
            additional_bonafide = len(spoof_indices) - len(oversampled_bonafide_indices)
            oversampled_bonafide_indices.extend(random.sample(bonafide_indices, additional_bonafide))
            oversampled_indices = oversampled_bonafide_indices + spoof_indices

        random.shuffle(oversampled_indices)
        self.files = [self.files[i] for i in oversampled_indices]
        self.labels = [self.labels[i] for i in oversampled_indices]
        print(f"{split.capitalize()} - After balancing - Bonafide: {self.labels.count(0)}, Spoof: {self.labels.count(1)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        audio_path = self.files[idx]
        label = self.labels[idx]
        audio, sr = librosa.load(audio_path, sr=16000)

        # Apply augmentation if enabled
        if self.augmentation:
            audio = self.augmentation(audio, sample_rate=16000)

        if len(audio) > self.max_length:
            audio = audio[:self.max_length]
        else:
            audio = np.pad(audio, (0, self.max_length - len(audio)), "constant")

        inputs = self.feature_extractor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
        return inputs.input_values.squeeze(0), torch.tensor(label, dtype=torch.long)

#%% [Model Class]
class SpoofDetector(nn.Module):
    def __init__(self):
        super(SpoofDetector, self).__init__()
        self.wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base")
        self.dropout = nn.Dropout(0.6)  # Increased dropout to 0.6
        self.classifier = nn.Linear(768, 2)

    def forward(self, input_values):
        outputs = self.wavlm(input_values).last_hidden_state
        pooled = torch.mean(outputs, dim=1)  # Mean pooling
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits

#%% [Focal Loss]
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()

#%% [Evaluation Function]
def evaluate_model(model, test_loader, device="cpu"):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_scores = []

    with torch.no_grad():
        for input_values, labels in tqdm(test_loader, desc="Evaluating"):
            input_values, labels = input_values.to(device), labels.to(device)
            outputs = model(input_values)
            scores = torch.softmax(outputs, dim=1)[:, 1]
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(scores.cpu().numpy())

    test_accuracy = 100 * correct / total
    fpr, tpr, _ = roc_curve(all_labels, all_scores)
    precision, recall, _ = precision_recall_curve(all_labels, all_scores)
    eer = fpr[np.nanargmin(np.abs(tpr - (1 - fpr)))]
    return test_accuracy, all_preds, all_labels, all_scores, fpr, tpr, precision, recall, 100 * eer

#%% [Custom Audio Testing Function]
def test_custom_audio(model, feature_extractor, audio_path, max_length=32000, device="cpu"):
    model.eval()
    audio, sr = librosa.load(audio_path, sr=16000)
    if len(audio) > max_length:
        audio = audio[:max_length]
    else:
        audio = np.pad(audio, (0, max_length - len(audio)), "constant")
    
    inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        outputs = model(input_values)
        scores = torch.softmax(outputs, dim=1)
        confidence = scores.max().item() * 100
        _, predicted = torch.max(outputs.data, 1)
        prediction = "Bonafide" if predicted.item() == 0 else "Spoof"

    print(f"\nCustom Audio Test Result:")
    print(f"File: {audio_path}")
    print(f"Prediction: {prediction}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"Probabilities (Bonafide, Spoof): {scores.tolist()}")

#%% [Plotting Functions]
def plot_roc_curve(fpr, tpr, eer, filename="roc_curve.png"):
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.plot([eer/100, eer/100], [0, 1 - eer/100], color='red', linestyle='--', label=f'EER = {eer:.2f}%')
    plt.plot([0, eer/100], [1 - eer/100, 1 - eer/100], color='red', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(filename)
    plt.close()

def plot_precision_recall_curve(precision, recall, filename="precision_recall_curve.png"):
    pr_auc = auc(recall, precision)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig(filename)
    plt.close()

#%% [Main Function]
def main():
    start_time = time.time()

    base_dir = "/Users/simranpatel/Desktop/cap_proj/code/deepfake-pythonmodel/model"
    os.makedirs(base_dir, exist_ok=True)
    model_save_path = os.path.join(base_dir, "wavlm_for_trained.pth")

    print("Initializing WavLM feature extractor and model...")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --- Load FOR Dataset ---
    print("\n=== Loading FOR Dataset ===")
    for_data_dir = "/Users/simranpatel/Desktop/cap_proj/code/dataset/archive/for-2sec/for-2seconds/"

    # Training split with augmentation
    train_dataset = FoRDataset(data_dir=for_data_dir, split="training", feature_extractor=feature_extractor, max_length=32000, augment=True)
    print(f"Training dataset loaded with {len(train_dataset)} samples after balancing.")

    # Validation split (no augmentation)
    val_dataset = FoRDataset(data_dir=for_data_dir, split="validation", feature_extractor=feature_extractor, max_length=32000, augment=False)
    print(f"Validation dataset loaded with {len(val_dataset)} samples after balancing.")

    # Test split (no augmentation, limit to 500 samples)
    test_dataset = FoRDataset(data_dir=for_data_dir, split="testing", feature_extractor=feature_extractor, max_length=32000, augment=False)
    test_indices = list(range(len(test_dataset)))[:500]  # Limit to 500 samples
    test_dataset.files = [test_dataset.files[i] for i in test_indices]
    test_dataset.labels = [test_dataset.labels[i] for i in test_indices]
    print(f"Test dataset loaded with {len(test_dataset)} samples after limiting.")

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)

    # --- Initialize Model ---
    model = SpoofDetector()
    model.to(device)

    # --- Training Setup ---
    criterion = FocalLoss(alpha=0.75, gamma=2.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    num_epochs = 5
    patience = 2
    best_val_acc = 0.0
    patience_counter = 0

    # --- Training Loop with Validation ---
    print("\n=== Training Model ===")
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for input_values, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            input_values, labels = input_values.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(input_values)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.2f}%")

        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        with torch.no_grad():
            for input_values, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                input_values, labels = input_values.to(device), labels.to(device)
                outputs = model(input_values)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total
        val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%")

        # Early stopping and model saving
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved best model with validation accuracy {best_val_acc:.2f}% at {model_save_path}")
        else:
            patience_counter += 1
            print(f"No improvement in validation accuracy. Patience counter: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # --- Load Best Model for Evaluation ---
    print("\n=== Loading Best Model for Evaluation ===")
    model.load_state_dict(torch.load(model_save_path))
    model.to(device)

    # --- Evaluate Model on Test Set ---
    print("\n=== Evaluating Model on Test Set ===")
    test_accuracy, test_preds, test_labels, test_scores, fpr, tpr, precision, recall, eer = evaluate_model(model, test_loader, device=device)

    print("\nModel Evaluation Results (FOR Test Set):")
    print("-" * 50)
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    print(f"Equal Error Rate (EER): {eer:.2f}%")

    cm = confusion_matrix(test_labels, test_preds)
    print("\nConfusion Matrix (FOR Test Set):")
    print(f"{'':<15} | {'Predicted Bonafide':<20} | {'Predicted Spoof':<20}")
    print("-" * 60)
    print(f"{'Actual Bonafide':<15} | {cm[0,0]:<20} | {cm[0,1]:<20}")
    print(f"{'Actual Spoof':<15} | {cm[1,0]:<20} | {cm[1,1]:<20}")
    print("-" * 60)

    plt.figure(figsize=(6, 6))
    plt.matshow(cm, cmap="Blues")
    plt.title("Confusion Matrix (FOR Test Set)", pad=20)
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0, 1], ["Bonafide", "Spoof"])
    plt.yticks([0, 1], ["Bonafide", "Spoof"])
    for (i, j), val in np.ndenumerate(cm):
        plt.text(j, i, f"{val}", ha="center", va="center", color="black")
    plt.savefig(os.path.join(base_dir, "confusion_matrix_for.png"))
    plt.close()

    plot_roc_curve(fpr, tpr, eer, filename=os.path.join(base_dir, "roc_curve_for.png"))
    plot_precision_recall_curve(precision, recall, filename=os.path.join(base_dir, "precision_recall_curve_for.png"))

    # --- Test on Custom Audio Files ---
    print("\n=== Testing on Custom Audio Files ===")
    test_files = [
        "/Users/simranpatel/Downloads/Record (online-voice-recorder.com).mp3",
        "/Users/simranpatel/Downloads/file1032.wav_16k.wav_norm.wav_mono.wav_silence.wav_2sec.wav",
        "/Users/simranpatel/Desktop/cap_proj/code/dataset/ASVspoof2021_LA_eval/LA_E_5656373.flac"
    ]

    for test_file in test_files:
        if os.path.exists(test_file):
            test_custom_audio(model, feature_extractor, test_file, max_length=32000, device=device)
        else:
            print(f"Error: File '{test_file}' not found. Skipping custom audio test.")

    total_time = time.time() - start_time
    print(f"\nTotal Execution Time: {total_time:.2f} seconds (~{total_time/60:.2f} minutes)")

#%% [Entry Point]
if __name__ == "__main__":
    main()