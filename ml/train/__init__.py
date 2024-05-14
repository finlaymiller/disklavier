import os
import time
import numpy as np
from datetime import datetime
from utils import console
from rich.progress import Progress

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from utils.image import format_image, plot_images
from ml.data_tools.augmentation import noise


class Trainer:
    model: torch.nn.Module
    train_dl: DataLoader
    val_dl: DataLoader
    optimizer: Optimizer
    train_loss = {}
    p = "[aquamarine1]trainer[/aquamarine1]:"

    def __init__(
        self,
        model: torch.nn.Module,
        model_name: str,
        loader: DataLoader,
        device: torch.device,
        params,
        plot_dir: str,
    ) -> None:
        console.log(f"{self.p} initializing model: '{model_name}'")

        self.model_name = model_name
        self.model = model
        self.train_dl = loader
        self.params = params
        self.device = device
        self.plot_dir = plot_dir
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.params.learning_rate
        )

        self.log_dir = os.path.join(
            self.params.log_dir,
            f"{datetime.now().strftime('%y-%m-%d_%H%M%S')}{'_' if self.params.log_tag else ''}{self.params.log_tag}",
        )
        # if self.params.ckpt_path:
        #     self.ckpt_dir = self.params.ckpt_path
        # else:
        self.ckpt_dir = os.path.join("ml", "outputs", "checkpoints", model_name)

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

        with Progress() as progress:
            epoch_task = progress.add_task("[cyan3]epochs", total=self.params.epochs)
            train_task = progress.add_task("[green3]dataset", total=len(self.train_dl))

            self.train_loss = {}
            for epoch in range(self.params.epochs):
                running_loss = 0.0
                for i, (names, images) in enumerate(self.train_dl):
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

                    progress.update(train_task, advance=1)
                # track loss
                loss = running_loss / len(self.train_dl)
                self.train_loss[f"epoch {epoch}"] = loss

                filename = f"{datetime.now().strftime('%y%m%d-%H%M%S')}_epoch-{epoch+1}"
                self.test_reconstruction(
                    images[0], names[0], self.plot_dir, f"epoch-{epoch}", True, filename
                )

                # checkpoint
                ckpt_file = os.path.join(
                    self.ckpt_dir,
                    filename,
                )
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "loss": loss,
                    },
                    ckpt_file,
                )

                # update progress bars
                progress.update(epoch_task, advance=1)
                progress.update(train_task, completed=0)

        console.log(
            f"{self.p} done training! loss is",
            self.train_loss,
        )

    def test_reconstruction(
        self,
        image: torch.Tensor,
        label: str,
        output_dir: str,
        run_file=None,
        overfit: bool = False,
        plot_name=None,
    ) -> None:
        """"""
        # console.log(f"{self.p}testing '{self.model_name}' image reconstruction on '{label}'")
        # noisy_image = image  # noise(image)
        # noisy_image = noisy_image.to(self.device)

        clean_image = torch.from_numpy(np.expand_dims(image.cpu(), 0))
        test_image = image.to(self.device, copy=True)
        output = self.model(test_image)

        # images = [
        #     format_image(image),
        #     noisy_image.cpu().data,
        #     output.cpu().data,
        # ]
        images = [
            clean_image,
            test_image.cpu().data,
            output.cpu().data,
        ]
        titles = [
            f"{label} (epochs={self.params.epochs})",
            f"noisy ({self.params.noise_factor}% noise)",
            f"reconstructed (loss={list(self.train_loss.items())[-1][1]:.3e})",
        ]

        if not plot_name:
            plot_name = f"{datetime.now().strftime('%y%m%d-%H%M%S')}"

        plot_images(
            images,
            titles,
            main_title=f"{run_file}",
            outfile_name=plot_name,
            output_dir=output_dir,
        )
