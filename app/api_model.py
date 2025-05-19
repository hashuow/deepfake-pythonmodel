import torch
import torch.nn as nn
from transformers import WavLMModel

class SpoofDetector(nn.Module):
    def __init__(self):
        super(SpoofDetector, self).__init__()
        self.wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base-plus")  # Match feature extractor
        self.dropout = nn.Dropout(0.4)  # Matches fine-tuning script
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