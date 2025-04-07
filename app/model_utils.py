import torch
import librosa
import numpy as np
from transformers import Wav2Vec2FeatureExtractor
from api_model import SpoofDetector

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load feature extractor and model
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base")
model = SpoofDetector()
model.load_state_dict(torch.load("/Users/simranpatel/Desktop/cap_proj/code/deepfake-pythonmodel/model/best_spoof_detector.pth", map_location=device))
model.to(device)
model.eval()

def predict_audio(file_path):
    max_length = 8000
    audio, sr = librosa.load(file_path, sr=16000)

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

    return {
        "result": prediction,
        "confidence": f"{confidence:.2f}%",
        "real": True if prediction == "Bonafide" else False
    }
