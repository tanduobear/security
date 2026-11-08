import torch
import torch.nn as nn

class CNN1D(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: (B, F) -> (B, 1, F)
        x = x.unsqueeze(1)
        return self.net(x).squeeze(-1)