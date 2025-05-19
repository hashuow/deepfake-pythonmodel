#%% [Imports]
import torch
import librosa
import numpy as np
from transformers import Wav2Vec2FeatureExtractor, WavLMModel, Wav2Vec2Model
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
from sklearn.model_selection import train_test_split
import os
import time
import matplotlib.pyplot as plt
import warnings
import random
from tqdm import tqdm

# Suppress matplotlib warnings on macOS
import matplotlib
matplotlib.use('Agg')

# Suppress transformers deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.utils.generic")

#%% [Dataset Classes]
class ASVspoofDFDataset(Dataset):
    def __init__(self, data_dir, trial_file, key_file, feature_extractor, max_length=32000):
        self.data_dir = data_dir
        self.feature_extractor = feature_extractor
        self.max_length = max_length
        self.files = []
        self.labels = []

        key_dict = {}
        key_path = os.path.join(data_dir, key_file)
        if not os.path.exists(key_path):
            raise FileNotFoundError(f"Key file not found at: {key_path}")
        with open(key_path, "r") as kf:
            for line in kf:
                parts = line.strip().split()
                if len(parts) >= 6:
                    file_id = parts[1]
                    label_str = parts[5]
                    key_dict[file_id] = 0 if label_str == "bonafide" else 1

        trial_path = os.path.join(data_dir, trial_file)
        if not os.path.exists(trial_path):
            raise FileNotFoundError(f"Trial file not found at: {trial_path}")
        print(f"Reading trial file: {trial_path}")
        with open(trial_path, "r") as f:
            for line in f:
                file_id = line.strip()
                if file_id in key_dict:
                    audio_path = os.path.join(data_dir, "flac", f"{file_id}.flac")
                    if os.path.exists(audio_path):
                        self.files.append(audio_path)
                        self.labels.append(key_dict[file_id])

        # Oversample bonafide samples
        bonafide_indices = [i for i, label in enumerate(self.labels) if label == 0]
        spoof_indices = [i for i, label in enumerate(self.labels) if label == 1]
        oversampled_indices = spoof_indices + bonafide_indices * 10
        self.files = [self.files[i] for i in oversampled_indices]
        self.labels = [self.labels[i] for i in oversampled_indices]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        audio_path = self.files[idx]
        label = self.labels[idx]
        audio, sr = librosa.load(audio_path, sr=16000)
        if len(audio) > self.max_length:
            audio = audio[:self.max_length]
        else:
            audio = np.pad(audio, (0, self.max_length - len(audio)), "constant")
        inputs = self.feature_extractor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
        return inputs.input_values.squeeze(0), torch.tensor(label, dtype=torch.long)

class FoRDataset(Dataset):
    def __init__(self, data_dir, split, feature_extractor, max_length=32000, files=None, labels=None):
        self.data_dir = os.path.join(data_dir, split) if split else data_dir
        self.feature_extractor = feature_extractor
        self.max_length = max_length
        self.files = files if files is not None else []
        self.labels = labels if labels is not None else []
        self.split = split

        if not self.files:
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
        if len(audio) > self.max_length:
            audio = audio[:self.max_length]
        else:
            audio = np.pad(audio, (0, self.max_length - len(audio)), "constant")
        inputs = self.feature_extractor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
        return inputs.input_values.squeeze(0), torch.tensor(label, dtype=torch.long)

#%% [Model Classes]
class SpoofDetectorASVspoof(torch.nn.Module):
    def __init__(self):
        super(SpoofDetectorASVspoof, self).__init__()
        self.wavlm = Wav2Vec2Model.from_pretrained("microsoft/wavlm-base")
        self.classifier = torch.nn.Linear(768, 2)

    def forward(self, input_values):
        outputs = self.wavlm(input_values).last_hidden_state
        pooled = torch.mean(outputs, dim=1)
        logits = self.classifier(pooled)
        return logits

class SpoofDetectorFOR(torch.nn.Module):
    def __init__(self):
        super(SpoofDetectorFOR, self).__init__()
        self.wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base")
        self.dropout = torch.nn.Dropout(0.5)
        self.batch_norm = torch.nn.BatchNorm1d(768)
        self.classifier = torch.nn.Linear(768, 2)

    def forward(self, input_values):
        outputs = self.wavlm(input_values).last_hidden_state
        pooled = torch.mean(outputs, dim=1)
        pooled = self.batch_norm(pooled)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits

#%% [Evaluation Function for Individual Models]
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

#%% [Evaluation Function with Ensemble]
def evaluate_ensemble(model_for, model_asvspoof, test_loader, device="cpu", w_for=0.5, w_asvspoof=0.5):
    model_for.eval()
    model_asvspoof.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_scores = []

    with torch.no_grad():
        for input_values, labels in tqdm(test_loader, desc="Evaluating Ensemble"):
            input_values, labels = input_values.to(device), labels.to(device)

            # Get predictions from FOR model
            outputs_for = model_for(input_values)
            scores_for = torch.softmax(outputs_for, dim=1)

            # Get predictions from ASVspoof model
            outputs_asvspoof = model_asvspoof(input_values)
            scores_asvspoof = torch.softmax(outputs_asvspoof, dim=1)

            # Combine probabilities (weighted average)
            combined_scores = w_for * scores_for + w_asvspoof * scores_asvspoof
            scores = combined_scores[:, 1]  # Probability of Spoof class
            _, predicted = torch.max(combined_scores, dim=1)

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

#%% [Custom Audio Testing Function with Ensemble]
def test_custom_audio_ensemble(model_for, model_asvspoof, feature_extractor, audio_path, max_length=32000, device="cpu", w_for=0.5, w_asvspoof=0.5):
    model_for.eval()
    model_asvspoof.eval()
    audio, sr = librosa.load(audio_path, sr=16000)
    if len(audio) > max_length:
        audio = audio[:max_length]
    else:
        audio = np.pad(audio, (0, max_length - len(audio)), "constant")
    
    inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        # Get predictions from FOR model
        outputs_for = model_for(input_values)
        scores_for = torch.softmax(outputs_for, dim=1)

        # Get predictions from ASVspoof model
        outputs_asvspoof = model_asvspoof(input_values)
        scores_asvspoof = torch.softmax(outputs_asvspoof, dim=1)

        # Combine probabilities
        combined_scores = w_for * scores_for + w_asvspoof * scores_asvspoof
        confidence = combined_scores.max().item() * 100
        _, predicted = torch.max(combined_scores, dim=1)
        prediction = "Bonafide" if predicted.item() == 0 else "Spoof"

    print(f"\nCustom Audio Test Result:")
    print(f"File: {audio_path}")
    print(f"Prediction: {prediction}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"Probabilities (Bonafide, Spoof): {combined_scores.tolist()}")

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
    model_path_for = os.path.join(base_dir, "final_spoof_pretrained.pth")
    model_path_asvspoof = os.path.join(base_dir, "best_spoof_detector.pth")

    print("Initializing WavLM feature extractor...")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --- Load Datasets for Evaluation ---
    print("\n=== Loading ASVspoof DF Dataset ===")
    asvspoof_df_data_dir = "/Users/simranpatel/Desktop/cap_proj/code/dataset/ASVspoof2021_DF_eval/"
    trial_file = "ASVspoof2021.DF.cm.eval.trl.txt"
    key_file = "keys/DF/CM/trial_metadata.txt"

    asvspoof_df_dataset = ASVspoofDFDataset(
        data_dir=asvspoof_df_data_dir, trial_file=trial_file, key_file=key_file, feature_extractor=feature_extractor, max_length=32000
    )
    print(f"ASVspoof DF dataset loaded with {len(asvspoof_df_dataset)} samples after oversampling.")

    labels_df = [label for _, label in asvspoof_df_dataset]
    _, temp_indices_df = train_test_split(
        range(len(asvspoof_df_dataset)), train_size=0.7, random_state=42, stratify=labels_df
    )
    temp_labels_df = [labels_df[i] for i in temp_indices_df]
    _, test_indices_df = train_test_split(
        temp_indices_df, train_size=0.5, random_state=42, stratify=temp_labels_df
    )
    test_indices_df = test_indices_df[:1000]  # Reduce test set size
    test_dataset_df = Subset(asvspoof_df_dataset, test_indices_df)
    test_loader_df = DataLoader(test_dataset_df, batch_size=1, shuffle=False, num_workers=2)
    print(f"ASVspoof DF - Test set size: {len(test_dataset_df)}")

    print("\n=== Loading FOR Dataset ===")
    for_data_dir = "/Users/simranpatel/Desktop/cap_proj/code/dataset/archive/for-2sec/for-2seconds/"
    test_dataset_for = FoRDataset(data_dir=for_data_dir, split="testing", feature_extractor=feature_extractor, max_length=32000)
    test_indices_for = list(range(len(test_dataset_for)))[:500]  # Reduce test set size
    test_dataset_for = Subset(test_dataset_for, test_indices_for)
    test_loader_for = DataLoader(test_dataset_for, batch_size=1, shuffle=False, num_workers=2)
    print(f"FOR - Test set size: {len(test_dataset_for)}")

    # --- Load Models ---
    print("\n=== Loading Trained Models ===")
    print(f"Loading FOR-trained model from {model_path_for}...")
    model_for = SpoofDetectorFOR()
    if not os.path.exists(model_path_for):
        raise FileNotFoundError(f"FOR model file not found at: {model_path_for}")
    model_for.load_state_dict(torch.load(model_path_for))
    model_for.to(device)

    print(f"Loading ASVspoof DF-trained model from {model_path_asvspoof}...")
    model_asvspoof = SpoofDetectorASVspoof()
    if not os.path.exists(model_path_asvspoof):
        raise FileNotFoundError(f"ASVspoof DF model file not found at: {model_path_asvspoof}")
    model_asvspoof.load_state_dict(torch.load(model_path_asvspoof))
    model_asvspoof.to(device)

    # --- Evaluate Individual Models ---
    # Evaluate model_for on ASVspoof DF
    print("\n=== Evaluating FOR Model on ASVspoof DF Test Set ===")
    test_accuracy_df_for, test_preds_df_for, test_labels_df_for, test_scores_df_for, fpr_df_for, tpr_df_for, precision_df_for, recall_df_for, eer_df_for = evaluate_model(model_for, test_loader_df, device=device)
    print(f"FOR Model - ASVspoof DF Test Accuracy: {test_accuracy_df_for:.2f}%")
    print(f"FOR Model - ASVspoof DF EER: {eer_df_for:.2f}%")

    # Evaluate model_asvspoof on ASVspoof DF
    print("\n=== Evaluating ASVspoof Model on ASVspoof DF Test Set ===")
    test_accuracy_df_asvspoof, test_preds_df_asvspoof, test_labels_df_asvspoof, test_scores_df_asvspoof, fpr_df_asvspoof, tpr_df_asvspoof, precision_df_asvspoof, recall_df_asvspoof, eer_df_asvspoof = evaluate_model(model_asvspoof, test_loader_df, device=device)
    print(f"ASVspoof Model - ASVspoof DF Test Accuracy: {test_accuracy_df_asvspoof:.2f}%")
    print(f"ASVspoof Model - ASVspoof DF EER: {eer_df_asvspoof:.2f}%")

    # Evaluate model_for on FOR
    print("\n=== Evaluating FOR Model on FOR Test Set ===")
    test_accuracy_for_for, test_preds_for_for, test_labels_for_for, test_scores_for_for, fpr_for_for, tpr_for_for, precision_for_for, recall_for_for, eer_for_for = evaluate_model(model_for, test_loader_for, device=device)
    print(f"FOR Model - FOR Test Accuracy: {test_accuracy_for_for:.2f}%")
    print(f"FOR Model - FOR EER: {eer_for_for:.2f}%")

    # Evaluate model_asvspoof on FOR
    print("\n=== Evaluating ASVspoof Model on FOR Test Set ===")
    test_accuracy_for_asvspoof, test_preds_for_asvspoof, test_labels_for_asvspoof, test_scores_for_asvspoof, fpr_for_asvspoof, tpr_for_asvspoof, precision_for_asvspoof, recall_for_asvspoof, eer_for_asvspoof = evaluate_model(model_asvspoof, test_loader_for, device=device)
    print(f"ASVspoof Model - FOR Test Accuracy: {test_accuracy_for_asvspoof:.2f}%")
    print(f"ASVspoof Model - FOR EER: {eer_for_asvspoof:.2f}%")

    # --- Compute Ensemble Weights ---
    # Use inverse EER as weights (lower EER = higher weight)
    # For ASVspoof DF test set
    total_eer_df = eer_df_for + eer_df_asvspoof
    w_for_df = (1 - eer_df_for / total_eer_df) if total_eer_df > 0 else 0.5
    w_asvspoof_df = (1 - eer_df_asvspoof / total_eer_df) if total_eer_df > 0 else 0.5
    print(f"\nEnsemble Weights for ASVspoof DF Test Set: w_for={w_for_df:.3f}, w_asvspoof={w_asvspoof_df:.3f}")

    # For FOR test set
    total_eer_for = eer_for_for + eer_for_asvspoof
    w_for_for = (1 - eer_for_for / total_eer_for) if total_eer_for > 0 else 0.5
    w_asvspoof_for = (1 - eer_for_asvspoof / total_eer_for) if total_eer_for > 0 else 0.5
    print(f"Ensemble Weights for FOR Test Set: w_for={w_for_for:.3f}, w_asvspoof={w_asvspoof_for:.3f}")

    # --- Evaluate the Ensemble ---
    print("\n=== Evaluating the Ensemble on ASVspoof DF Test Set ===")
    test_accuracy_df, test_preds_df, test_labels_df, test_scores_df, fpr_df, tpr_df, precision_df, recall_df, eer_df = evaluate_ensemble(model_for, model_asvspoof, test_loader_df, device=device, w_for=w_for_df, w_asvspoof=w_asvspoof_df)

    print("\nModel Evaluation Results (ASVspoof DF Test Set):")
    print("-" * 50)
    print(f"Test Accuracy: {test_accuracy_df:.2f}%")
    print(f"Equal Error Rate (EER): {eer_df:.2f}%")

    cm_df = confusion_matrix(test_labels_df, test_preds_df)
    print("\nConfusion Matrix (ASVspoof DF Test Set):")
    print(f"{'':<15} | {'Predicted Bonafide':<20} | {'Predicted Spoof':<20}")
    print("-" * 60)
    print(f"{'Actual Bonafide':<15} | {cm_df[0,0]:<20} | {cm_df[0,1]:<20}")
    print(f"{'Actual Spoof':<15} | {cm_df[1,0]:<20} | {cm_df[1,1]:<20}")
    print("-" * 60)

    plt.figure(figsize=(6, 6))
    plt.matshow(cm_df, cmap="Blues")
    plt.title("Confusion Matrix (ASVspoof DF Test Set)", pad=20)
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0, 1], ["Bonafide", "Spoof"])
    plt.yticks([0, 1], ["Bonafide", "Spoof"])
    for (i, j), val in np.ndenumerate(cm_df):
        plt.text(j, i, f"{val}", ha="center", va="center", color="black")
    plt.savefig(os.path.join(base_dir, "confusion_matrix_asvspoof_df_ensemble.png"))
    plt.close()

    plot_roc_curve(fpr_df, tpr_df, eer_df, filename=os.path.join(base_dir, "roc_curve_asvspoof_df_ensemble.png"))
    plot_precision_recall_curve(precision_df, recall_df, filename=os.path.join(base_dir, "precision_recall_curve_asvspoof_df_ensemble.png"))

    print("\n=== Evaluating the Ensemble on FOR Test Set ===")
    test_accuracy_for, test_preds_for, test_labels_for, test_scores_for, fpr_for, tpr_for, precision_for, recall_for, eer_for = evaluate_ensemble(model_for, model_asvspoof, test_loader_for, device=device, w_for=w_for_for, w_asvspoof=w_asvspoof_for)

    print("\nModel Evaluation Results (FOR Test Set):")
    print("-" * 50)
    print(f"Test Accuracy: {test_accuracy_for:.2f}%")
    print(f"Equal Error Rate (EER): {eer_for:.2f}%")

    cm_for = confusion_matrix(test_labels_for, test_preds_for)
    print("\nConfusion Matrix (FOR Test Set):")
    print(f"{'':<15} | {'Predicted Bonafide':<20} | {'Predicted Spoof':<20}")
    print("-" * 60)
    print(f"{'Actual Bonafide':<15} | {cm_for[0,0]:<20} | {cm_for[0,1]:<20}")
    print(f"{'Actual Spoof':<15} | {cm_for[1,0]:<20} | {cm_for[1,1]:<20}")
    print("-" * 60)

    plt.figure(figsize=(6, 6))
    plt.matshow(cm_for, cmap="Blues")
    plt.title("Confusion Matrix (FOR Test Set)", pad=20)
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0, 1], ["Bonafide", "Spoof"])
    plt.yticks([0, 1], ["Bonafide", "Spoof"])
    for (i, j), val in np.ndenumerate(cm_for):
        plt.text(j, i, f"{val}", ha="center", va="center", color="black")
    plt.savefig(os.path.join(base_dir, "confusion_matrix_for_ensemble.png"))
    plt.close()

    plot_roc_curve(fpr_for, tpr_for, eer_for, filename=os.path.join(base_dir, "roc_curve_for_ensemble.png"))
    plot_precision_recall_curve(precision_for, recall_for, filename=os.path.join(base_dir, "precision_recall_curve_for_ensemble.png"))

    print("\n=== Testing on Custom Audio Files ===")
    test_files = [
        "/Users/simranpatel/Downloads/Record (online-voice-recorder.com).mp3",
        "/Users/simranpatel/Downloads/file1032.wav_16k.wav_norm.wav_mono.wav_silence.wav_2sec.wav",
        "/Users/simranpatel/Desktop/cap_proj/code/dataset/ASVspoof2021_LA_eval/LA_E_5656373.flac"
    ]

    # Use average weights for custom audio (since we don't know which dataset they resemble)
    w_for_custom = (w_for_df + w_for_for) / 2
    w_asvspoof_custom = (w_asvspoof_df + w_asvspoof_for) / 2
    print(f"Ensemble Weights for Custom Audio: w_for={w_for_custom:.3f}, w_asvspoof={w_asvspoof_custom:.3f}")

    for test_file in test_files:
        if os.path.exists(test_file):
            test_custom_audio_ensemble(model_for, model_asvspoof, feature_extractor, test_file, max_length=32000, device=device, w_for=w_for_custom, w_asvspoof=w_asvspoof_custom)
        else:
            print(f"Error: File '{test_file}' not found. Skipping custom audio test.")

    total_time = time.time() - start_time
    print(f"\nTotal Execution Time: {total_time:.2f} seconds (~{total_time/60:.2f} minutes)")

#%% [Entry Point]
if __name__ == "__main__":
    main()