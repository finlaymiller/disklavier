import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

import os
from datetime import datetime
from utils import console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from utils.image import format_image, plot_images
from ml.data_tools.augmentation import noise

from typing import List


class Trainer:
    model: torch.nn.Module
    train_dl: DataLoader
    val_dl: DataLoader
    optimizer: Optimizer
    train_loss: List = []
    p = "[aquamarine1]trainer[/aquamarine1]:"

    def __init__(
        self,
        model: torch.nn.Module,
        model_name: str,
        loader: DataLoader,
        device: torch.device,
        params,
    ) -> None:
        console.log(f"{self.p} initializing model: '{model_name}'")

        self.model_name = model_name
        self.model = model
        self.train_dl = loader
        self.params = params
        self.device = device
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.params.learning_rate
        )

        self.log_dir = os.path.join(
            self.params.log_dir,
            f"{datetime.now().strftime('%y-%m-%d_%H%M%S')}{'_' if self.params.log_tag else ''}{self.params.log_tag}",
        )

        total_parameters = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        console.log(
            f"{self.p} loaded model '{self.model_name}' with {total_parameters} parameters"
        )

    def train(self) -> None:
        """Train the model"""

        console.log(
            f"{self.p} training for {self.params.epochs} epochs on {len(self.train_dl) * self.params.batch_size} images"
        )
        writer = SummaryWriter(self.log_dir)

        with Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            train_task = progress.add_task("[green3]dataset", total=len(self.train_dl))
            epoch_task = progress.add_task("[cyan3]epochs", total=self.params.epochs)

            self.train_loss = []
            for epoch in range(self.params.epochs):
                running_loss = 0.0
                for i, (_, images) in enumerate(self.train_dl):
                    # train step
                    images = images.to(self.device)
                    self.optimizer.zero_grad()
                    outputs = self.model(images)
                    loss = self.loss_fn(outputs, images)
                    # backpropagation
                    loss.backward()
                    # update the parameters
                    self.optimizer.step()
                    # update loss tracking
                    running_loss += loss.item()

                    # log to tensorboard
                    global_step = epoch * len(self.train_dl) + i
                    writer.add_scalar("training/loss", loss.item(), global_step)
                    for p_name, param in self.model.named_parameters():
                        writer.add_histogram(
                            f"weights/{p_name}", param.data, global_step
                        )
                        if param.requires_grad:
                            writer.add_histogram(
                                f"gradients/{p_name}.grad", param.grad, global_step
                            )

                    progress.update(train_task, advance=self.params.batch_size)
                # track loss
                loss = running_loss / len(self.train_dl)
                self.train_loss.append(loss)

                progress.update(epoch_task, advance=1)

            console.log(f"{self.p}done training! loss is", self.train_loss)

    def test_reconstruction(
        self, image, label: str, output_dir: str, run_file=None, overfit: bool = False
    ) -> None:
        """"""
        console.log(
            f"{self.p}testing {self.model_name} image reconstruction on {label}",
            image.shape,
        )
        noisy_image = image  # noise(image)
        noisy_image = noisy_image.to(self.device)

        output = self.model(noisy_image)

        images = [
            format_image(image),
            noisy_image.cpu().data,
            output.cpu().data,
        ]
        titles = [
            f"{label} (epochs={self.params.epochs})",
            f"noisy ({self.params.noise_factor}% noise)",
            f"reconstructed (loss={self.train_loss[-1]:.03f})",
        ]

        plot_images(images, titles, main_title=f"{run_file}", output_dir=output_dir)
