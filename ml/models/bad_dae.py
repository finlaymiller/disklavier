import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F


class BadAutoEncoder(nn.Module):
    def __init__(self, **args):
        super(BadAutoEncoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.MaxPool2d(2, 2),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 8, kernel_size=3, stride=2),
            nn.SiLU(),
            nn.ConvTranspose2d(8, 16, kernel_size=3, stride=2),
            nn.SiLU(),
            nn.ConvTranspose2d(16, 32, kernel_size=2, stride=2),
            nn.SiLU(),
            nn.ConvTranspose2d(32, 64, kernel_size=2, stride=2),
            nn.SiLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.SiLU(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x
