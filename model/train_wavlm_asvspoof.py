#%% [Imports]
import torch
import torch.nn as nn
import librosa
import numpy as np
from transformers import Wav2Vec2FeatureExtractor, WavLMModel
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_curve, precision_recall_curve, auc
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
class ASVspoofDataset(Dataset):
    def __init__(self, trial_file, key_file, audio_dir, feature_extractor, max_length=32000, augment=False):
        self.audio_dir = audio_dir
        self.feature_extractor = feature_extractor
        self.max_length = max_length
        self.files = []
        self.labels = []
        self.augment = augment

        # Define augmentation pipeline
        if self.augment:
            self.augmentation = Compose([
                AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
                TimeStretch(min_rate=0.8, max_rate=1.2, p=0.5),
                PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
                Gain(min_gain_db=-6, max_gain_db=6, p=0.3),
                Shift(min_shift=-0.4, max_shift=0.4, p=0.5),
            ])
        else:
            self.augmentation = None

        # Load key file to map audio IDs to labels
        key_dict = {}
        print(f"Reading key file: {key_file}")
        if not os.path.exists(key_file):
            raise FileNotFoundError(f"Key file not found at: {key_file}")
        with open(key_file, "r") as kf:
            for line in kf:
                parts = line.strip().split()
                if len(parts) >= 6:  # Ensure there are enough columns
                    audio_id = parts[1]  # Second column: e.g., DF_E_2000011
                    label_str = parts[5]  # Sixth column: bonafide or spoof
                    key_dict[audio_id] = 0 if label_str == 'bonafide' else 1

        # Load trial file to get list of audio IDs
        print(f"Reading trial file: {trial_file}")
        if not os.path.exists(trial_file):
            raise FileNotFoundError(f"Trial file not found at: {trial_file}")
        with open(trial_file, "r") as f:
            for line in f:
                audio_id = line.strip()
                if audio_id in key_dict:
                    audio_path = os.path.join(audio_dir, f"{audio_id}.wav")
                    if os.path.exists(audio_path):
                        print(f"Found audio file: {audio_path}")  # Debugging output
                        self.files.append(audio_path)
                        self.labels.append(key_dict[audio_id])
                    else:
                        print(f"File not found: {audio_path}")  # Debugging output

        if not self.files:
            raise ValueError(f"No audio files found in {audio_dir}")

        print(f"Full dataset loaded with {len(self.files)} samples before balancing.")
        print(f"Before balancing - Bonafide: {self.labels.count(0)}, Spoof: {self.labels.count(1)}")

        # Undersample the majority class
        bonafide_indices = [i for i, label in enumerate(self.labels) if label == 0]
        spoof_indices = [i for i, label in enumerate(self.labels) if label == 1]
        num_bonafide = len(bonafide_indices)
        num_spoof = len(spoof_indices)

        if num_bonafide < num_spoof:
            spoof_indices = random.sample(spoof_indices, num_bonafide)
        else:
            bonafide_indices = random.sample(bonafide_indices, num_spoof)

        balanced_indices = bonafide_indices + spoof_indices
        random.shuffle(balanced_indices)
        self.files = [self.files[i] for i in balanced_indices]
        self.labels = [self.labels[i] for i in balanced_indices]

        print(f"After undersampling - Total samples: {len(self.files)}, Bonafide: {self.labels.count(0)}, Spoof: {self.labels.count(1)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        audio_path = self.files[idx]
        label = self.labels[idx]
        audio, sr = librosa.load(audio_path, sr=16000)

        if self.augment and self.augmentation:
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
        # Freeze WavLM layers to reduce training time
        for param in self.wavlm.parameters():
            param.requires_grad = False
        self.dropout = nn.Dropout(0.4)
        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )

    def forward(self, input_values):
        outputs = self.wavlm(input_values).last_hidden_state
        pooled = torch.mean(outputs, dim=1)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits

#%% [Focal Loss]
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0):
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
    try:
        audio, sr = librosa.load(audio_path, sr=16000)
    except Exception as e:
        print(f"Error loading {audio_path}: {e}")
        return

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
    model_save_path = os.path.join(base_dir, "wavlm_asvspoof_with_key_file.pth")

    print("Initializing WavLM feature extractor and model...")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --- Load ASVspoof Dataset ---
    print("\n=== Loading ASVspoof Training Dataset ===")
    trial_file = "/Users/simranpatel/Desktop/cap_proj/code/dataset/ASVspoof2021_DF_eval/ASVspoof2021.DF.cm.eval.trl.txt"
    key_file = "/Users/simranpatel/Desktop/cap_proj/code/dataset/ASVspoof2021_DF_eval/keys/DF/CM/trial_metadata.txt"
    audio_dir = "/Users/simranpatel/Desktop/cap_proj/code/dataset/ASVspoof2021_DF_eval/wav"

    # Split dataset into train and test (80% train, 20% test)
    full_dataset = ASVspoofDataset(trial_file, key_file, audio_dir, feature_extractor, max_length=32000, augment=True)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    test_dataset.dataset.augment = False  # Disable augmentation for test set
    print(f"Training set size: {len(train_dataset)}, Test set size: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)

    # --- Initialize Model ---
    model = SpoofDetector()
    model.to(device)

    # --- Training Setup ---
    criterion = FocalLoss(alpha=0.5, gamma=2.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, verbose=True)
    num_epochs = 10
    patience = 3
    best_test_acc = 0.0
    patience_counter = 0

    # --- Training Loop with Validation ---
    print("\n=== Training Model ===")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (input_values, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            input_values, labels = input_values.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(input_values)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if (batch_idx + 1) % 100 == 0:
                batch_acc = 100 * correct / total
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {running_loss/(batch_idx+1):.4f}, Batch Accuracy: {batch_acc:.2f}%")

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")

        # Evaluate on test set
        test_accuracy, test_preds, test_labels, test_scores, fpr, tpr, precision, recall, eer = evaluate_model(model, test_loader, device=device)
        print(f"Epoch {epoch+1}/{num_epochs}, Test Accuracy: {test_accuracy:.2f}%, EER: {eer:.2f}%")

        # Early stopping and model saving
        if test_accuracy > best_test_acc:
            best_test_acc = test_accuracy
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved best model with test accuracy {best_test_acc:.2f}% at {model_save_path}")
        else:
            patience_counter += 1
            print(f"No improvement in test accuracy. Patience counter: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        scheduler.step(test_accuracy)

    # --- Load Best Model for Final Evaluation ---
    print("\n=== Evaluating Model on Test Set ===")
    model.load_state_dict(torch.load(model_save_path))
    test_accuracy, test_preds, test_labels, test_scores, fpr, tpr, precision, recall, eer = evaluate_model(model, test_loader, device=device)

    print("\nModel Evaluation Results (ASVspoof Test Set):")
    print("-" * 50)
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    print(f"Equal Error Rate (EER): {eer:.2f}%")

    cm = np.zeros((2, 2), dtype=int)
    for true, pred in zip(test_labels, test_preds):
        cm[true, pred] += 1
    print("\nConfusion Matrix (ASVspoof Test Set):")
    print(f"{'':<15} | {'Predicted Bonafide':<20} | {'Predicted Spoof':<20}")
    print("-" * 60)
    print(f"{'Actual Bonafide':<15} | {cm[0,0]:<20} | {cm[0,1]:<20}")
    print(f"{'Actual Spoof':<15} | {cm[1,0]:<20} | {cm[1,1]:<20}")
    print("-" * 60)

    plt.figure(figsize=(6, 6))
    plt.matshow(cm, cmap="Blues")
    plt.title("Confusion Matrix (ASVspoof Test Set)", pad=20)
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0, 1], ["Bonafide", "Spoof"])
    plt.yticks([0, 1], ["Bonafide", "Spoof"])
    for (i, j), val in np.ndenumerate(cm):
        plt.text(j, i, f"{val}", ha="center", va="center", color="black")
    plt.savefig(os.path.join(base_dir, "confusion_matrix_asvspoof.png"))
    plt.close()

    plot_roc_curve(fpr, tpr, eer, filename=os.path.join(base_dir, "roc_curve_asvspoof.png"))
    plot_precision_recall_curve(precision, recall, filename=os.path.join(base_dir, "precision_recall_curve_asvspoof.png"))

    # --- Test on Custom Audio Files ---
    print("\n=== Testing on Custom Audio Files ===")
    test_files = [
        "/Users/simranpatel/Downloads/Record (online-voice-recorder.com).mp3",
        "/Users/simranpatel/Downloads/file1032.wav_16k.wav_norm.wav_mono.wav_silence.wav_2sec.wav",
        "/Users/simranpatel/Downloads/HumeAI_voice-preview_simran1.wav",
        "/Users/simranpatel/Downloads/record1.mp3"
    ]

    for test_file in test_files:
        test_custom_audio(model, feature_extractor, test_file, max_length=32000, device=device)

    total_time = time.time() - start_time
    print(f"\nTotal Execution Time: {total_time:.2f} seconds (~{total_time/60:.2f} minutes)")

#%% [Entry Point]
if __name__ == "__main__":
    main()