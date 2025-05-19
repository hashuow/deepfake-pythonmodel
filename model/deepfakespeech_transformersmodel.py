# === System & Utility ===
import os                    # For file and folder operations
from tqdm import tqdm        # For progress bars during processing

# === Data Handling ===
import pandas as pd          # For handling CSV files (like metadata)
import numpy as np           # For numerical operations

# === Audio Processing ===
import librosa               # For loading and trimming audio
import soundfile as sf       # For saving trimmed audio (optional)

# === Deep Learning: PyTorch & Transformers ===
import torch                 # Core PyTorch library
from torch.utils.data import Dataset  # For creating custom dataset loaders

# Hugging Face Transformers - handles the Wav2Vec2 model
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ForSequenceClassification,
    TrainingArguments,
    Trainer
)

# === Model Evaluation ===
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve
)

# === Visualization ===
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
import torch



# === Step 2: Define Paths ===

# Where your dataset and CSV will live
DATASET_DIR = "dataset_asv"
PROCESSED_CSV = os.path.join(DATASET_DIR, "processed_data", "preprocessed_data.csv")
REAL_DIR = os.path.join(DATASET_DIR, "real_speech")
FAKE_DIR = os.path.join(DATASET_DIR, "fake_speech")
MODEL_DIR = "./results"
MAX_AUDIO_LENGTH = 5 * 16000  # 5 seconds x 16000 samples/second

os.makedirs(os.path.join(DATASET_DIR, "processed_data"), exist_ok=True)

data = []

for label, folder in [("real", REAL_DIR), ("fake", FAKE_DIR)]:
    for fname in os.listdir(folder):
        if fname.endswith(".flac") or fname.endswith(".wav") or fname.endswith(".mp3"):
            full_path = os.path.join(folder, fname)
            data.append({"filename": full_path, "label": label})

df = pd.DataFrame(data)
df.to_csv(PROCESSED_CSV, index=False)
print(f"✅ Preprocessed CSV saved at: {PROCESSED_CSV}")


# Load the metadata
df = pd.read_csv(PROCESSED_CSV)
df.dropna(inplace=True)

# Encode 'real' as 0, 'fake' as 1
df["encoded_label"] = df["label"].map({"real": 0, "fake": 1})

# Split into training and test sets
from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["encoded_label"], random_state=42)


class AudioDataset(Dataset):
    def __init__(self, dataframe, processor):
        self.data = dataframe
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        audio_path = row["filename"]
        label = row["encoded_label"]

        audio, sr = librosa.load(audio_path, sr=16000)

        # Trim or pad audio to 5 seconds
        if len(audio) < MAX_AUDIO_LENGTH:
            audio = np.pad(audio, (0, MAX_AUDIO_LENGTH - len(audio)), mode='constant')
        else:
            audio = audio[:MAX_AUDIO_LENGTH]

        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
        item = {k: v.squeeze(0) for k, v in inputs.items()}
        item["labels"] = torch.tensor(label)
        return item

from torch.utils.data import Dataset

class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, processor):
        self.data = dataframe.reset_index(drop=True)
        self.processor = processor
        self.max_length = 5 * 16000  # 5 seconds

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        audio_path = row["filename"]
        label = row["encoded_label"]

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"File not found: {audio_path}")

        # Load and preprocess audio
        audio, sr = librosa.load(audio_path, sr=16000)

        if len(audio) < self.max_length:
            audio = np.pad(audio, (0, self.max_length - len(audio)), mode="constant")
        else:
            audio = audio[:self.max_length]

        # Process audio using Wav2Vec2 processor
        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)

        # Return as model input
        item = {key: val.squeeze(0) for key, val in inputs.items()}
        item["labels"] = torch.tensor(label)

        return item



# === Step 6: Load Processor and Model ===
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "facebook/wav2vec2-base",
    num_labels=2  # 0: real, 1: fake
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("✅ Processor and model loaded. Using device:", device)


# === Step 7: Train the Model ===

from transformers import TrainingArguments, Trainer

# ✅ 1. Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",       # Evaluate every epoch
    save_strategy="epoch",             # Save model after every epoch
    num_train_epochs=3,                # You can increase this later
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=3e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    report_to="none"  # Disable wandb
)

# ✅ 2. Create datasets
train_dataset = AudioDataset(train_df, processor)
test_dataset = AudioDataset(test_df, processor)

# ✅ 3. Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# ✅ 4. Train the model
trainer.train()

# ✅ 5. Save the trained model
trainer.save_model(MODEL_DIR)
print("✅ Model trained and saved to:", MODEL_DIR)


# === Step 8: Evaluation and Prediction ===

# Load trained model again (optional if already in memory)
model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
model.eval()

# Reload full dataset (if needed)
df = pd.read_csv(PROCESSED_CSV)
df.dropna(inplace=True)
df["encoded_label"] = df["label"].map({"real": 0, "fake": 1})

# === Predict function ===
def predict_audio(file_path):
    audio, sr = librosa.load(file_path, sr=16000)
    if len(audio) < MAX_AUDIO_LENGTH:
        audio = np.pad(audio, (0, MAX_AUDIO_LENGTH - len(audio)), mode="constant")
    else:
        audio = audio[:MAX_AUDIO_LENGTH]

    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        pred = np.argmax(probs)

    return pred, probs[1]  # 1 = fake

# === Evaluate on test set ===
y_true, y_pred, y_scores = [], [], []

for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
    if not os.path.exists(row["filename"]):
        continue
    true_label = row["encoded_label"]
    pred, score = predict_audio(row["filename"])
    y_true.append(true_label)
    y_pred.append(pred)
    y_scores.append(score)

# === Accuracy & Classification
from sklearn.metrics import roc_curve

acc = accuracy_score(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=["Real", "Fake"], digits=4)
cm = confusion_matrix(y_true, y_pred)

# Equal Error Rate (EER)
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
fnr = 1 - tpr
eer_threshold = thresholds[np.nanargmin(np.abs(fnr - fpr))]
eer = fpr[np.nanargmin(np.abs(fnr - fpr))]

# === Print Evaluation
print("\n✅ Model Evaluation Results:")
print("--------------------------------------------------")
print(f"Test Accuracy: {round(acc * 100, 2)}%")
print(f"Equal Error Rate (EER): {round(eer * 100, 2)}%")
print("\n🔹 Classification Report:\n", report)
print("Confusion Matrix (Test Set):")
print("                | Predicted Bonafide   | Predicted Spoof")
print("------------------------------------------------------------")
print(f"Actual Bonafide | {cm[0][0]:<22} | {cm[0][1]:<18}")
print(f"Actual Spoof    | {cm[1][0]:<22} | {cm[1][1]:<18}")
print("------------------------------------------------------------")

# === Optional: Visualize confusion matrix
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# === Step 9: Predict a custom audio file
custom_file = "LA_T_1000137.flac"  # Change to your test file (.wav, .mp3, .flac)
if os.path.exists(custom_file):
    label, prob = predict_audio(custom_file)
    print(f"\n🔍 Prediction for {custom_file}: {'Real' if label == 0 else 'Fake'} (Spoof Score: {round(prob, 3)})")
else:
    print(f"\n⚠️ File '{custom_file}' not found.")

