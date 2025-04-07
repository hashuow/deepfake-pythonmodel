import torch
import librosa
import numpy as np
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
import os

# Define the SpoofDetector model class (same as in your training script)
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

# Load the trained model
def load_model(model_path, device="cpu"):
    print(f"Loading model from {model_path}...")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base")
    model = SpoofDetector()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded successfully.")
    return model, feature_extractor

# Preprocess the audio file
def preprocess_audio(audio_path, feature_extractor, max_length=8000):
    audio, sr = librosa.load(audio_path, sr=16000)
    if len(audio) > max_length:
        audio = audio[:max_length]
    else:
        audio = np.pad(audio, (0, max_length - len(audio)), "constant")
    inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    return inputs.input_values

# Run prediction on the audio file
def predict(model, feature_extractor, audio_path, device="cpu"):
    input_values = preprocess_audio(audio_path, feature_extractor, max_length=8000)
    input_values = input_values.to(device)
    with torch.no_grad():
        outputs = model(input_values)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, dim=1)
        prediction = "Bonafide" if predicted.item() == 0 else "Spoof"
        confidence = confidence.item() * 100
    return prediction, confidence

def main():
    # Define paths and device
    model_path = "/Users/simranpatel/Desktop/cap_proj/code/deepfake-pythonmodel/model/best_spoof_detector.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load the model and feature extractor
    model, feature_extractor = load_model(model_path, device)

    # Get audio file path from user
    audio_path = input("Enter the path to the audio file (e.g., '/Users/simranpatel/Desktop/cap_proj/code/dataset/ASVspoof2021_DF_eval/flac/DF_E_1000048.flac'): ")
    if not os.path.exists(audio_path):
        print(f"Error: Audio file '{audio_path}' not found.")
        return

    # Run prediction
    try:
        prediction, confidence = predict(model, feature_extractor, audio_path, device)
    except Exception as e:
        print(f"Error processing the audio file: {str(e)}")
        return

    # Display results
    print("\nPrediction Results:")
    print(f"File: {audio_path}")
    print(f"Prediction: {prediction}")
    print(f"Confidence: {confidence:.2f}%")

if __name__ == "__main__":
    main()