import torch
from transformers import Wav2Vec2Model

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
