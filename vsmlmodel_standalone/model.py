import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class DeepfakeDetector(nn.Module):
    def __init__(self, num_frames=20, hidden_size=256, num_layers=2):
        super(DeepfakeDetector, self).__init__()
        # Load pretrained ResNet18
        self.cnn = resnet18(weights=ResNet18_Weights.DEFAULT)
        # Remove the last fully connected layer
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])  # Output: 512 features

        # LSTM for temporal modeling
        self.lstm = nn.LSTM(input_size=512, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=0.5)

        # Fully connected layer for classification
        self.fc = nn.Linear(hidden_size, 1)  # Binary classification

    def forward(self, x):
        # x: (batch, seq_len, C, H, W)
        batch_size, seq_len, C, H, W = x.size()
        x = x.view(batch_size * seq_len, C, H, W)  # Flatten for CNN
        features = self.cnn(x)  # (batch*seq, 512, 1, 1)
        features = features.view(batch_size, seq_len, -1)  # (batch, seq, 512)

        # LSTM
        lstm_out, _ = self.lstm(features)  # (batch, seq, hidden)
        # Take the last output
        final_out = lstm_out[:, -1, :]  # (batch, hidden)

        # Classification
        out = self.fc(final_out)  # (batch, 1)
        return out

if __name__ == "__main__":
    model = DeepfakeDetector()
    print(model)
