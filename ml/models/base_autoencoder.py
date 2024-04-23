import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F


class BaseAutoEncoder(nn.Module):
    def __init__(self):
        super(BaseAutoEncoder, self).__init__()

        # Encoder layers
        self.enc1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.enc3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.enc4 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Decoder layers
        self.dec1 = nn.ConvTranspose2d(8, 8, kernel_size=3, stride=2)
        self.dec2 = nn.ConvTranspose2d(8, 16, kernel_size=3, stride=2)
        self.dec3 = nn.ConvTranspose2d(16, 32, kernel_size=2, stride=2)
        self.dec4 = nn.ConvTranspose2d(32, 64, kernel_size=2, stride=2)
        self.out = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.pool(F.silu(self.enc1(x)))
        x = self.pool(F.silu(self.enc2(x)))
        x = self.pool(F.silu(self.enc3(x)))
        x = self.pool(F.silu(self.enc4(x)))

        # Decoder
        x = F.silu(self.dec1(x))
        x = F.silu(self.dec2(x))
        x = F.silu(self.dec3(x))
        x = F.silu(self.dec4(x))
        x = F.silu(self.out(x))

        return x


class Autoencoder(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        loss=nn.MSELoss,
        lr: float = 1e-3,
        num_input_channels: int = 1,
        width: int = 28,
        height: int = 28,
        scheduler_params: dict | None = None,
    ):
        """
        :param encoder: the encoder
        :param decoder: the decoder
        :param emb_channels: is the number of dimensions in the quantized embedding space
        :param z_channels: is the number of channels in the embedding space
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.loss = loss()
        self.lr = lr
        self.scheduler_params = scheduler_params
        # Example input array needed for visualizing the graph of the network
        self.example_input_array = torch.zeros(2, num_input_channels, width, height)

    def forward(self, x):
        """The forward function takes in an image and returns the reconstructed image."""
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def get_reconstruction_loss(self, x):
        """Given a batch of images, this function returns the reconstruction loss (MSE in our case)."""
        # x, _ = batch  # We do not need the labels
        x_hat = self.forward(x)
        loss = self.loss(x, x_hat, reduction="none")
        # loss = loss.sum(dim=[1, 2, 3]).mean(dim=[0])
        return loss

    def configure_optimizers(self):
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        # Using a scheduler is optional but can be helpful.
        # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, mode="min", factor=0.2, patience=20, min_lr=5e-5
        # )

        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": scheduler,
        #     "monitor": "val_loss",
        # }

    def get_embedding(self, x):
        return self.encoder(x)


class Encoder(nn.Module):
    def __init__(self, params):
        """Encoder

        Args:
           num_input_channels : Number of input channels of the image. For CIFAR, this parameter is 3
           base_channel_size : Number of channels we use in the first convolutional layers. Deeper layers might use a duplicate of it.
           latent_dim : Dimensionality of latent representation z
           act_fn : Activation function used throughout the encoder network
        """
        super().__init__()
        self.enc1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.enc3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.enc4 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # c_hid = base_channel_size
        # self.net = nn.Sequential(
        #     nn.Conv2d(num_input_channels, 8 * c_hid, kernel_size=3, padding=1),
        #     act_fn(),
        #     nn.MaxPool2d(2, 2),
        #     nn.Conv2d(8 * c_hid, 4 * c_hid, kernel_size=3, padding=1),
        #     act_fn(),
        #     nn.MaxPool2d(2, 2),
        #     nn.Conv2d(4 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
        #     act_fn(),
        #     nn.MaxPool2d(2, 2),
        #     nn.Conv2d(2 * c_hid, c_hid, kernel_size=3, padding=1),
        #     act_fn(),
        #     nn.MaxPool2d(2, 2),
        # )

    def forward(self, x):
        x = self.pool(F.silu(self.enc1(x)))
        x = self.pool(F.silu(self.enc2(x)))
        x = self.pool(F.silu(self.enc3(x)))
        x = self.pool(F.silu(self.enc4(x)))

        return x


class Decoder(nn.Module):
    def __init__(self, params):
        """Decoder

        Args:
           num_input_channels : Number of channels of the image to reconstruct. For CIFAR, this parameter is 3
           base_channel_size : Number of channels we use in the last convolutional layers. Early layers might use a duplicate of it.
           latent_dim : Dimensionality of latent representation z
           act_fn : Activation function used throughout the decoder network
        """
        super().__init__()
        self.dec1 = nn.ConvTranspose2d(8, 8, kernel_size=3, stride=2)
        self.dec2 = nn.ConvTranspose2d(8, 16, kernel_size=3, stride=2)
        self.dec3 = nn.ConvTranspose2d(16, 32, kernel_size=2, stride=2)
        self.dec4 = nn.ConvTranspose2d(32, 64, kernel_size=2, stride=2)
        self.out = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        # c_hid = base_channel_size
        # self.linear = nn.Sequential(nn.Linear(latent_dim, 2 * 16 * c_hid), act_fn())
        # self.net = nn.Sequential(
        #     nn.ConvTranspose2d(
        #         2 * c_hid,
        #         2 * c_hid,
        #         kernel_size=3,
        #         output_padding=1,
        #         padding=1,
        #         stride=2,
        #     ),  # 4x4 => 8x8
        #     act_fn(),
        #     nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
        #     act_fn(),
        #     nn.ConvTranspose2d(
        #         2 * c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2
        #     ),  # 8x8 => 16x16
        #     act_fn(),
        #     nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
        #     act_fn(),
        #     nn.ConvTranspose2d(
        #         c_hid,
        #         num_input_channels,
        #         kernel_size=3,
        #         output_padding=1,
        #         padding=1,
        #         stride=2,
        #     ),  # 16x16 => 32x32
        #     nn.Tanh(),  # The input images is scaled between -1 and 1, hence the output has to be bounded as well
        # )

    def forward(self, x):
        x = F.silu(self.dec1(x))
        x = F.silu(self.dec2(x))
        x = F.silu(self.dec3(x))
        x = F.silu(self.dec4(x))
        x = F.silu(self.out(x))
        # x = self.linear(x)
        # x = x.reshape(x.shape[0], -1, 4, 4)
        # x = self.net(x)
        return x
