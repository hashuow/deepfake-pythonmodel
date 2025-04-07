import torch
import librosa
import numpy as np
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
import os
import time
import matplotlib.pyplot as plt
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift
import warnings

# Suppress matplotlib warnings on macOS
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid CATransaction warnings

# Suppress transformers deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.utils.generic")

class ASVspoofDFDataset(Dataset):
    def __init__(self, data_dir, trial_file, key_file, feature_extractor, max_length=8000):
        self.data_dir = data_dir
        self.feature_extractor = feature_extractor
        self.max_length = max_length
        self.files = []
        self.labels = []
        # Define audio augmentations
        self.augment = Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
            TimeStretch(min_rate=0.8, max_rate=1.2, p=0.5),
            PitchShift(min_semitones=-4, max_semitones=4, p=0.5)
        ])

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
        oversampled_indices = spoof_indices + bonafide_indices * 10  # Oversample bonafide 10x
        self.files = [self.files[i] for i in oversampled_indices]
        self.labels = [self.labels[i] for i in oversampled_indices]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        audio_path = self.files[idx]
        label = self.labels[idx]
        audio, sr = librosa.load(audio_path, sr=16000)
        # Apply augmentation
        audio = self.augment(samples=audio, sample_rate=16000)
        if len(audio) > self.max_length:
            audio = audio[:self.max_length]
        else:
            audio = np.pad(audio, (0, self.max_length - len(audio)), "constant")
        inputs = self.feature_extractor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
        return inputs.input_values.squeeze(0), torch.tensor(label, dtype=torch.long)

class SpoofDetector(torch.nn.Module):
    def __init__(self):
        super(SpoofDetector, self).__init__()
        self.wavlm = Wav2Vec2Model.from_pretrained("microsoft/wavlm-base")
        self.classifier = torch.nn.Linear(768, 2)

    def forward(self, input_values):
        outputs = self.wavlm(input_values).last_hidden_state
        pooled = torch.mean(outputs, dim=1)
        logits = self.classifier(pooled)
        return logits

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = torch.nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()

def train_model(model, train_loader, val_loader, num_epochs=10, device="cpu", save_path="best_spoof_detector.pth"):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1)
    criterion = FocalLoss(alpha=0.75, gamma=2.0)
    train_accuracies = []
    val_accuracies = []
    train_losses = []
    best_val_accuracy = 0.0
    patience = 3
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

        # Update learning rate scheduler
        scheduler.step(val_accuracy)

        # Save best model
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

    # Save the final model state after training completes
    final_save_path = save_path.replace("best_", "final_")
    torch.save(model.state_dict(), final_save_path)
    print(f"Saved final model to {final_save_path} after training completed.")

    return train_accuracies, val_accuracies, train_losses

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

def test_custom_audio(model, feature_extractor, audio_path, max_length=8000, device="cpu"):
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
        _, predicted = torch.max(outputs.data, 1)
        prediction = "Bonafide" if predicted.item() == 0 else "Spoof"
        confidence = torch.softmax(outputs, dim=1).max().item() * 100

    print(f"\nCustom Audio Test Result:")
    print(f"File: {audio_path}")
    print(f"Prediction: {prediction}")
    print(f"Confidence: {confidence:.2f}%")

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
    plt.savefig("roc_curve.png")
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
    plt.savefig("precision_recall_curve.png")
    plt.close()

def plot_training_loss(train_losses):
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, color='green', lw=2, label='Training Loss')
    plt.xlabel('Batch (every 100 batches)')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend(loc="upper right")
    plt.savefig("training_loss.png")
    plt.close()

def main():
    start_time = time.time()

    # Define save paths
    base_dir = "/Users/simranpatel/Desktop/cap_proj/code/deepfake-pythonmodel/model/"  # Adjust for local
    # For Colab, uncomment the following line and comment the above
    # base_dir = "/content/drive/MyDrive/cap_proj/code/deepfake-pythonmodel/model/"
    os.makedirs(base_dir, exist_ok=True)  # Ensure directory exists
    best_model_path = os.path.join(base_dir, "best_spoof_detector.pth")

    # Count .flac files
    flac_dir = "/Users/simranpatel/Desktop/cap_proj/code/dataset/ASVspoof2021_DF_eval/flac/"
    # For Colab: flac_dir = "/content/drive/MyDrive/cap_proj/code/dataset/ASVspoof2021_DF_eval/flac/"
    flac_count = len([f for f in os.listdir(flac_dir) if f.endswith('.flac')])
    print(f"Number of .flac files: {flac_count}")

    print("Initializing WavLM feature extractor and model...")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base")
    model = SpoofDetector()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    data_dir = "/Users/simranpatel/Desktop/cap_proj/code/dataset/ASVspoof2021_DF_eval/"
    # For Colab: data_dir = "/content/drive/MyDrive/cap_proj/code/dataset/ASVspoof2021_DF_eval/"
    trial_file = "ASVspoof2021.DF.cm.eval.trl.txt"
    key_file = "keys/DF/CM/trial_metadata.txt"

    print("Loading dataset...")
    dataset = ASVspoofDFDataset(data_dir=data_dir, trial_file=trial_file, key_file=key_file, feature_extractor=feature_extractor)
    print(f"Dataset loaded with {len(dataset)} samples after oversampling.")

    bonafide_count = sum(1 for _, label in dataset if label == 0)
    spoof_count = sum(1 for _, label in dataset if label == 1)
    print(f"Bonafide samples: {bonafide_count}, Spoof samples: {spoof_count}")
    class_weight = spoof_count / bonafide_count if bonafide_count > 0 else 1.0
    print(f"Class weight (spoof/bonafide): {class_weight:.2f}")

    labels = [label for _, label in dataset]
    # 70/15/15 split
    train_indices, temp_indices = train_test_split(
        range(len(dataset)), train_size=0.7, random_state=42, stratify=labels
    )
    temp_labels = [labels[i] for i in temp_indices]
    val_indices, test_indices = train_test_split(
        temp_indices, train_size=0.5, random_state=42, stratify=temp_labels
    )

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    print(f"Training set size: {len(train_dataset)}, Validation set size: {len(val_dataset)}, Test set size: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

    print("Starting training...")
    train_accuracies, val_accuracies, train_losses = train_model(
        model, train_loader, val_loader, num_epochs=10, device=device, save_path=best_model_path
    )

    print("Loading the best model for evaluation...")
    model.load_state_dict(torch.load(best_model_path))
    model.to(device)

    print("Evaluating on test set...")
    test_accuracy, test_preds, test_labels, test_scores, fpr, tpr, precision, recall, eer = evaluate_model(model, test_loader, device=device)

    print("\nModel Evaluation Results:")
    print("-" * 50)
    print(f"Training Accuracy: {train_accuracies[-1]:.2f}%")
    print(f"Validation Accuracy: {val_accuracies[-1]:.2f}%")
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    print(f"Equal Error Rate (EER): {eer:.2f}%")

    cm = confusion_matrix(test_labels, test_preds)
    print("\nConfusion Matrix (Test Set):")
    print(f"{'':<15} | {'Predicted Bonafide':<20} | {'Predicted Spoof':<20}")
    print("-" * 60)
    print(f"{'Actual Bonafide':<15} | {cm[0,0]:<20} | {cm[0,1]:<20}")
    print(f"{'Actual Spoof':<15} | {cm[1,0]:<20} | {cm[1,1]:<20}")
    print("-" * 60)

    plt.figure(figsize=(6, 6))
    plt.matshow(cm, cmap="Blues")
    plt.title("Confusion Matrix (Test Set)", pad=20)
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0, 1], ["Bonafide", "Spoof"])
    plt.yticks([0, 1], ["Bonafide", "Spoof"])
    for (i, j), val in np.ndenumerate(cm):
        plt.text(j, i, f"{val}", ha="center", va="center", color="black")
    plt.savefig(os.path.join(base_dir, "confusion_matrix.png"))
    plt.close()

    plot_roc_curve(fpr, tpr, eer)
    plot_precision_recall_curve(precision, recall)
    plot_training_loss(train_losses)

    custom_audio_path = '/Users/simranpatel/Desktop/cap_proj/code/dataset/ASVspoof2021_LA_eval/LA_E_5656373.flac'
    if os.path.exists(custom_audio_path):
        test_custom_audio(model, feature_extractor, custom_audio_path, max_length=8000, device=device)
    else:
        print(f"Error: File '{custom_audio_path}' not found. Skipping custom audio test.")

    total_time = time.time() - start_time
    print(f"\nTotal Execution Time: {total_time:.2f} seconds (~{total_time/60:.2f} minutes)")

if __name__ == "__main__":
    main()