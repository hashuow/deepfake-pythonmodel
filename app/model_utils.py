import torch
import librosa
import numpy as np
from transformers import Wav2Vec2FeatureExtractor, WavLMModel
from api_model import SpoofDetector

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load feature extractor and model
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
    "microsoft/wavlm-base-plus",
    return_attention_mask=False
)
model = SpoofDetector()
try:
    model.load_state_dict(
        torch.load(
            "/Users/simranpatel/Desktop/cap_proj/code/deepfake-pythonmodel/model/wavlm_finetuned_for.pth",
            map_location=device,
            weights_only=True
        )
    )
    print("Model weights loaded successfully.")
except Exception as e:
    raise RuntimeError(f"Failed to load model weights: {str(e)}")
model.to(device)
model.eval()

def predict_audio(file_path):
    # Load audio at 16kHz with error handling
    try:
        audio, sr = librosa.load(file_path, sr=16000)
    except Exception as e:
        return {"error": f"Error loading audio: {str(e)}"}

    # Use max_length consistent with training (2 seconds at 16kHz) or keep 10 seconds
    max_length = 32000  # Changed from 160000 to match training data
    if len(audio) > max_length:
        audio = audio[:max_length]
    else:
        audio = np.pad(audio, (0, max_length - len(audio)), "constant")

    # Preprocess audio with feature extractor
    inputs = feature_extractor(
        audio,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
        max_length=max_length,
        truncation=True
    )
    input_values = inputs.input_values.to(device)

    # Make prediction
    with torch.no_grad():
        outputs = model(input_values)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, dim=1)
        prediction = "Bonafide" if predicted.item() == 0 else "Spoof"
        confidence = confidence.item() * 100

    return {
        "result": prediction,
        "confidence": f"{confidence:.2f}%",
        "real": True if prediction == "Bonafide" else False
    }