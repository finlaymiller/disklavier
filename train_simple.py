import os
import sys
import pretty_midi
from argparse import ArgumentParser
from omegaconf import OmegaConf

import torch
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard.writer import SummaryWriter


from ml.models import *
from utils import console
from utils.ml import init_weights


class MidiDataset(Dataset):
    def __init__(self, midi_folder):
        self.midi_folder = midi_folder
        self.midi_files = [
            os.path.join(midi_folder, f)
            for f in os.listdir(midi_folder)
            if f.endswith(".mid")
        ]

    def __len__(self):
        return len(self.midi_files)

    def __getitem__(self, idx):
        midi_file = self.midi_files[idx]
        piano_roll = self.midi_to_piano_roll(midi_file)
        return torch.tensor(piano_roll, dtype=torch.float32)

    def midi_to_piano_roll(self, midi_file):
        return pretty_midi.PrettyMIDI(midi_file).get_piano_roll()


def train(model, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    # writer = SummaryWriter(log_dir)

    for batch_idx, data in enumerate(train_loader):
        data = data.view(data.size(0), -1)  # Flatten the input
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = model.loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 10 == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}"
                f" ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}"
            )
    print(
        f"====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}"
    )


def main(args, params):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    console.log(f"using device: '{device}'")

    # model
    try:
        model = vae_models["VanillaVAE"](in_channels=3, latent_dim=128)
    except ImportError as e:
        console.log(e)
        sys.exit(1)

    console.log(f"successfully loaded model '{params.model.name}':\n", model)

    model.apply(lambda m: init_weights(m, 0.01, 0.1))
    model.to(device)

    # filesystem
    plot_dir = os.path.join(args.output_dir, "plots")
    ckpt_dir = os.path.join(args.output_dir, "checkpoints", params.model.name)

    if not os.path.exists(params.trainer.log_dir):
        console.log(f"creating new log folder as '{params.trainer.log_dir}'")
        os.makedirs(params.trainer.log_dir)
    if not os.path.exists(args.output_dir):
        console.log(f"creating new output folder as '{args.output_dir}'")
        os.makedirs(args.output_dir)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    # data
    dataset = MidiDataset(args.data_dir)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=0.0)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    for epoch in range(1, 11):
        train(model, train_loader, optimizer, epoch)


if __name__ == "__main__":
    # load arguments and parameters
    parser = ArgumentParser(description="train a model on MIDI data")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="directory in which to store model checkpoints and training logs",
    )
    parser.add_argument(
        "--data_dir", default=None, help="directory of custom training data"
    )
    parser.add_argument(
        "--param_file", default=None, help="path to parameter file, in .yaml"
    )
    parser.add_argument("--device", default="cuda:0", help="which device to run on")
    args = parser.parse_args()
    params = OmegaConf.load(args.param_file)

    console.log("running with arguments:\n", args)
    console.log("running with parameters:\n", params)

    main(args, params)
