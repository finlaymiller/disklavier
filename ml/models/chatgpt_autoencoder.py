import torch
import torch.nn as nn


class ChatGPTAutoencoder(nn.Module):
    def __init__(self):
        super(ChatGPTAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(
                1, 16, kernel_size=3, stride=2, padding=1
            ),  # output: [16, 70, 206]
            nn.SiLU(),
            nn.Conv2d(
                16, 32, kernel_size=3, stride=2, padding=1
            ),  # output: [32, 35, 103]
            nn.SiLU(),
            nn.Conv2d(
                32, 64, kernel_size=3, stride=2, padding=1
            ),  # output: [64, 18, 52]
            nn.SiLU(),
            nn.Conv2d(
                64, 128, kernel_size=3, stride=2, padding=1
            ),  # output: [128, 9, 26]
            nn.SiLU(),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # output: [64, 18, 52]
            nn.SiLU(),
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # output: [32, 35, 103]
            nn.SiLU(),
            nn.ConvTranspose2d(
                32, 16, kernel_size=3, stride=2, padding=1
            ),  # output: [16, 70, 206]
            nn.SiLU(),
            nn.ConvTranspose2d(
                16, 1, kernel_size=3, stride=2#, padding=1
            ),  # output: [1, 140, 412]
            nn.Sigmoid(),  # Ensuring output is within [0, 1]
        )

    def forward(self, x):
        # print("Input:", x.shape)
        # for layer in self.encoder:
        #     x = layer(x)
        #     print("Encoder:", x.shape)
        # for layer in self.decoder:
        #     x = layer(x)
        #     print("Decoder:", x.shape)
        x = self.encoder(x)
        x = self.decoder(x)
        return x
