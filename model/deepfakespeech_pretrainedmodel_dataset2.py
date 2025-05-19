#%% [Imports]
import torch
import librosa
import numpy as np
from transformers import Wav2Vec2FeatureExtractor, WavLMModel
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
import os
import time
import matplotlib.pyplot as plt
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Gain, ClippingDistortion
import warnings
import random
from sklearn.model_selection import train_test_split

# Suppress matplotlib warnings on macOS
import matplotlib
matplotlib.use('Agg')

# Suppress transformers deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.utils.generic")

#%% [Dataset Class]
#%% [Dataset Class]
class FoRDataset(Dataset):
    def __init__(self, data_dir, feature_extractor, max_length=32000, files=None, labels=None, split=""):
        self.data_dir = data_dir
        self.feature_extractor = feature_extractor
        self.max_length = max_length
        self.files = files if files is not None else []
        self.labels = labels if labels is not None else []
        self.split = split
        self.augment = Compose([
            AddGaussianNoise(min_amplitude=0.005, max_amplitude=0.05, p=0.8),
            TimeStretch(min_rate=0.7, max_rate=1.3, p=0.6),
            PitchShift(min_semitones=-5, max_semitones=5, p=0.6),
            Gain(min_gain_db=-6, max_gain_db=6, p=0.5),
            ClippingDistortion(min_percentile_threshold=0, max_percentile_threshold=20, p=0.5)
        ])

        if not self.files:  # Load files only if not provided
            if not os.path.exists(self.data_dir):
                raise FileNotFoundError(f"Directory not found: {self.data_dir}")

            real_dir = os.path.join(self.data_dir, "real")
            if os.path.exists(real_dir):
                for file in os.listdir(real_dir):
                    if file.endswith((".wav", ".mp3", ".flac")):  # Added .flac for ASVspoof dataset
                        audio_path = os.path.join(real_dir, file)
                        self.files.append(audio_path)
                        self.labels.append(0)  # Bonafide

            fake_dir = os.path.join(self.data_dir, "fake")
            if os.path.exists(fake_dir):
                for file in os.listdir(fake_dir):
                    if file.endswith((".wav", ".mp3", ".flac")):  # Added .flac for ASVspoof dataset
                        audio_path = os.path.join(fake_dir, file)
                        self.files.append(audio_path)
                        self.labels.append(1)  # Spoof

            if not self.files:
                raise ValueError(f"No WAV, MP3, or FLAC files found in {self.data_dir}/real or {self.data_dir}/fake")

            # Balance the dataset (1:1 ratio)
            bonafide_indices = [i for i, label in enumerate(self.labels) if label == 0]
            spoof_indices = [i for i, label in enumerate(self.labels) if label == 1]
            print(f"Dataset - Before balancing - Bonafide: {len(bonafide_indices)}, Spoof: {len(spoof_indices)}")

            if len(bonafide_indices) == 0 or len(spoof_indices) == 0:
                raise ValueError(f"Cannot balance dataset: Bonafide samples = {len(bonafide_indices)}, Spoof samples = {len(spoof_indices)}.")

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
            print(f"Dataset - After balancing - Bonafide: {self.labels.count(0)}, Spoof: {self.labels.count(1)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        audio_path = self.files[idx]
        label = self.labels[idx]
        audio, sr = librosa.load(audio_path, sr=16000)  # librosa supports .flac natively
        if "train" in self.split.lower():
            audio = self.augment(samples=audio, sample_rate=16000)
            if len(audio) > self.max_length:
                start = random.randint(0, len(audio) - self.max_length)
                audio = audio[start:start + self.max_length]
        if len(audio) > self.max_length:
            audio = audio[:self.max_length]
        else:
            audio = np.pad(audio, (0, self.max_length - len(audio)), "constant")
        inputs = self.feature_extractor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
        return inputs.input_values.squeeze(0), torch.tensor(label, dtype=torch.long)
    
#%% [Model Class]
class SpoofDetector(torch.nn.Module):
    def __init__(self):
        super(SpoofDetector, self).__init__()
        self.wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base")
        self.dropout = torch.nn.Dropout(0.6)  # Increased dropout to prevent overfitting
        self.batch_norm = torch.nn.BatchNorm1d(768)
        self.classifier = torch.nn.Linear(768, 2)

    def forward(self, input_values):
        outputs = self.wavlm(input_values).last_hidden_state
        pooled = torch.mean(outputs, dim=1)
        pooled = self.batch_norm(pooled)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits

#%% [Training Function]
def train_model(model, train_loader, val_loader, num_epochs=5, device="cpu", save_path="final_spoof_retrained_newdataset.pth"):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-3)  # Lower LR, higher weight decay
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1)
    class_weights = torch.tensor([1.0, 1.0]).to(device)  # Balanced class weights to prevent overfitting
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    train_accuracies = []
    val_accuracies = []
    train_losses = []
    best_val_accuracy = 0.0
    patience = 2  # Reduced patience to prevent overfitting
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        start_time = time.time()

        for batch_idx, batch in enumerate(train_loader):
            input_values, labels = batch
            input_values, labels = input_values.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(input_values)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if batch_idx % 100 == 0:
                train_losses.append(loss.item())
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        train_accuracies.append(train_accuracy)

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for input_values, labels in val_loader:
                input_values, labels = input_values.to(device), labels.to(device)
                outputs = model(input_values)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_accuracy = 100 * val_correct / val_total
        val_accuracies.append(val_accuracy)

        scheduler.step(val_accuracy)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path} with validation accuracy: {best_val_accuracy:.2f}%")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"Patience counter: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Accuracy: {val_accuracy:.2f}%, Time: {epoch_time:.2f}s")

    torch.save(model.state_dict(), save_path)
    print(f"Saved final model to {save_path} after training completed.")

    return train_accuracies, val_accuracies, train_losses

#%% [Evaluation Function]
def evaluate_model(model, test_loader, device="cpu"):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_scores = []

    with torch.no_grad():
        for input_values, labels in test_loader:
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
def plot_roc_curve(fpr, tpr, eer):
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
    plt.savefig("roc_curve_newdataset.png")
    plt.close()

def plot_precision_recall_curve(precision, recall):
    pr_auc = auc(recall, precision)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig("precision_recall_curve_newdataset.png")
    plt.close()

def plot_training_loss(train_losses):
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, color='green', lw=2, label='Training Loss')
    plt.xlabel('Batch (every 100 batches)')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend(loc="upper right")
    plt.savefig("training_loss_newdataset.png")
    plt.close()

#%% [Main Function]
def main():
    start_time = time.time()

    base_dir = "/Users/simranpatel/Desktop/cap_proj/code/deepfake-pythonmodel/model"
    os.makedirs(base_dir, exist_ok=True)
    load_path = os.path.join(base_dir, "final_spoof_pretrained.pth")  # Path to load existing weights
    save_path = os.path.join(base_dir, "final_spoof_retrained_newdataset.pth")  # New path for retrained model

    print("Initializing WavLM feature extractor and model...")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base")
    model = SpoofDetector()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load the pre-trained model
    if os.path.exists(load_path):
        print(f"Loading pre-trained model from {load_path}...")
        model.load_state_dict(torch.load(load_path))
    else:
        raise FileNotFoundError(f"Pre-trained model not found at {load_path}. Please train the model first.")

    model.to(device)

    # Freeze WavLM layers except the last two to prevent overfitting
    for name, param in model.wavlm.named_parameters():
        if "layer.10" in name or "layer.11" in name:  # Unfreeze last two layers
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Specify the new dataset path here
    new_data_dir = "/Users/simranpatel/Desktop/cap_proj/code/dataset/drive-download-20250409T055040Z-001"  # Replace with the actual path to your ASVspoof LA dataset
    old_data_dir = "/Users/simranpatel/Desktop/cap_proj/code/dataset/archive/for-2sec/for-2seconds/testing"  # Keep the old dataset for testing

    print("Loading ASVspoof LA dataset (single folder with real and fake subfolders)...")
    # Load all data from the ASVspoof LA dataset
    full_dataset = FoRDataset(data_dir=new_data_dir, feature_extractor=feature_extractor, max_length=32000)

    # Split the dataset into training (70%), validation (15%), and test (15%)
    train_files, temp_files, train_labels, temp_labels = train_test_split(
        full_dataset.files, full_dataset.labels, test_size=0.3, stratify=full_dataset.labels, random_state=42
    )
    val_files, test_files_new, val_labels, test_labels_new = train_test_split(
        temp_files, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
    )

    # Create datasets for training, validation, and test (ASVspoof LA dataset)
    train_dataset = FoRDataset(
        data_dir=new_data_dir, feature_extractor=feature_extractor, max_length=32000,
        files=train_files, labels=train_labels, split="train"
    )
    val_dataset = FoRDataset(
        data_dir=new_data_dir, feature_extractor=feature_extractor, max_length=32000,
        files=val_files, labels=val_labels, split="val"
    )
    test_dataset_new = FoRDataset(
        data_dir=new_data_dir, feature_extractor=feature_extractor, max_length=32000,
        files=test_files_new, labels=test_labels_new, split="test_new"
    )

    # Load test dataset from the old dataset for final evaluation
    test_dataset_original = FoRDataset(
        data_dir=old_data_dir, feature_extractor=feature_extractor, max_length=32000, split="testing"
    )

    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Test set size (ASVspoof LA): {len(test_dataset_new)}")
    print(f"Test set size (original dataset): {len(test_dataset_original)}")

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)
    test_loader_new = DataLoader(test_dataset_new, batch_size=1, shuffle=False, num_workers=2)
    test_loader_original = DataLoader(test_dataset_original, batch_size=1, shuffle=False, num_workers=2)

    print("Starting retraining on ASVspoof LA dataset to create a new model...")
    train_accuracies, val_accuracies, train_losses = train_model(
        model, train_loader, val_loader, num_epochs=5, device=device, save_path=save_path
    )

    print("Loading the best retrained model for evaluation...")
    if os.path.exists(save_path):
        model = SpoofDetector()
        model.load_state_dict(torch.load(save_path))
        model.to(device)

        # Evaluate on the test set from the ASVspoof LA dataset
        print("Evaluating on test set (from ASVspoof LA dataset)...")
        test_accuracy_new, test_preds_new, test_labels_new, test_scores_new, fpr_new, tpr_new, precision_new, recall_new, eer_new = evaluate_model(model, test_loader_new, device=device)

        print("\nModel Evaluation Results (ASVspoof LA Test Set):")
        print("-" * 50)
        print(f"Training Accuracy: {train_accuracies[-1]:.2f}%")
        print(f"Validation Accuracy: {val_accuracies[-1]:.2f}%")
        print(f"Test Accuracy (ASVspoof LA): {test_accuracy_new:.2f}%")
        print(f"Equal Error Rate (EER, ASVspoof LA): {eer_new:.2f}%")

        cm_new = confusion_matrix(test_labels_new, test_preds_new)
        print("\nConfusion Matrix (ASVspoof LA Test Set):")
        print(f"{'':<15} | {'Predicted Bonafide':<20} | {'Predicted Spoof':<20}")
        print("-" * 60)
        print(f"{'Actual Bonafide':<15} | {cm_new[0,0]:<20} | {cm_new[0,1]:<20}")
        print(f"{'Actual Spoof':<15} | {cm_new[1,0]:<20} | {cm_new[1,1]:<20}")
        print("-" * 60)

        plt.figure(figsize=(6, 6))
        plt.matshow(cm_new, cmap="Blues")
        plt.title("Confusion Matrix (ASVspoof LA Test Set)", pad=20)
        plt.colorbar()
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.xticks([0, 1], ["Bonafide", "Spoof"])
        plt.yticks([0, 1], ["Bonafide", "Spoof"])
        for (i, j), val in np.ndenumerate(cm_new):
            plt.text(j, i, f"{val}", ha="center", va="center", color="black")
        plt.savefig(os.path.join(base_dir, "confusion_matrix_asvspoof_la_test.png"))
        plt.close()

        # Evaluate on the test set from the original dataset
        print("\nEvaluating on test set (from original dataset)...")
        test_accuracy_original, test_preds_original, test_labels_original, test_scores_original, fpr_original, tpr_original, precision_original, recall_original, eer_original = evaluate_model(model, test_loader_original, device=device)

        print("\nModel Evaluation Results (Original Dataset Test Set):")
        print("-" * 50)
        print(f"Test Accuracy (Original Dataset): {test_accuracy_original:.2f}%")
        print(f"Equal Error Rate (EER, Original Dataset): {eer_original:.2f}%")

        cm_original = confusion_matrix(test_labels_original, test_preds_original)
        print("\nConfusion Matrix (Original Dataset Test Set):")
        print(f"{'':<15} | {'Predicted Bonafide':<20} | {'Predicted Spoof':<20}")
        print("-" * 60)
        print(f"{'Actual Bonafide':<15} | {cm_original[0,0]:<20} | {cm_original[0,1]:<20}")
        print(f"{'Actual Spoof':<15} | {cm_original[1,0]:<20} | {cm_original[1,1]:<20}")
        print("-" * 60)

        plt.figure(figsize=(6, 6))
        plt.matshow(cm_original, cmap="Blues")
        plt.title("Confusion Matrix (Original Dataset Test Set)", pad=20)
        plt.colorbar()
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.xticks([0, 1], ["Bonafide", "Spoof"])
        plt.yticks([0, 1], ["Bonafide", "Spoof"])
        for (i, j), val in np.ndenumerate(cm_original):
            plt.text(j, i, f"{val}", ha="center", va="center", color="black")
        plt.savefig(os.path.join(base_dir, "confusion_matrix_original_test.png"))
        plt.close()

        # Use metrics from the original test set for plots
        plot_roc_curve(fpr_original, tpr_original, eer_original)
        plot_precision_recall_curve(precision_original, recall_original)
        plot_training_loss(train_losses)

        test_files = [
            "/Users/simranpatel/Downloads/Record (online-voice-recorder.com).mp3",
            "/Users/simranpatel/Downloads/file1032.wav_16k.wav_norm.wav_mono.wav_silence.wav_2sec.wav"
        ]

        for test_file in test_files:
            if os.path.exists(test_file):
                test_custom_audio(model, feature_extractor, test_file, max_length=32000, device=device)
            else:
                print(f"Error: File '{test_file}' not found. Skipping custom audio test.")
    else:
        print(f"Model file not found at {save_path}. Skipping evaluation.")

    total_time = time.time() - start_time
    print(f"\nTotal Execution Time: {total_time:.2f} seconds (~{total_time/60:.2f} minutes)")

#%% [Entry Point]
if __name__ == "__main__":
    main()