import os
import sys
from glob import iglob
import torch
from torch.utils.data import DataLoader
from datetime import datetime

from argparse import ArgumentParser
from omegaconf import OmegaConf

from ml.data_tools.augmenter import DataAugmenter
from ml.train import Trainer
from ml.models import *

from utils import console
from utils.ml import init_weights

p = "[white]main[/white]   :"


def main(args, params):
    # model setup
    try:
        model = vae_models[params.model.name](**params.model)
        # model = load_model(params.model.name, params.model.path)
    except ImportError as e:
        console.log(f"{p} {e}")
        sys.exit(1)

    console.log(f"{p} successfully loaded model '{params.model.name}':\n", model)
    model.apply(
        lambda m: init_weights(m, params.model.w_init_min, params.model.w_init_max)
    )

    if os.path.exists(os.path.join(args.output_dir, "checkpoints")) and params.model.load_ckpt:
        checkpoints = iglob(
            os.path.join(os.path.join(args.output_dir, "checkpoints"), "*")
        )
        model_checkpoint = torch.load(max(checkpoints, key=os.path.getctime))

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(args.device)
    console.log(f"{p} using device: '{device}'")
    model.to(device)

    # filesystem setup
    plot_dir = os.path.join(args.output_dir, "plots")
    ckpt_dir = os.path.join(args.output_dir, "checkpoints", params.model.name)

    if not os.path.exists(params.trainer.log_dir):
        console.log(f"{p} creating new log folder as '{params.trainer.log_dir}'")
        os.makedirs(params.trainer.log_dir)
    if not os.path.exists(args.output_dir):
        console.log(f"{p} creating new output folder as '{args.output_dir}'")
        os.makedirs(args.output_dir)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    # dataset setup
    augmenter = DataAugmenter(args.data_dir, params.augmenter)
    dataset = augmenter.augment()
    loader = DataLoader(
        dataset,
        batch_size=params.loader.batch_size,
        shuffle=params.loader.shuffle,
        num_workers=params.loader.num_workers,
    )

    # train
    trainer = Trainer(
        model, params.model.name, loader, device, params.trainer, plot_dir
    )
    trainer.train()
    test_label, test_image = augmenter.get_clean()
    trainer.test_reconstruction(
        test_image,
        test_label,
        plot_dir,
        args.param_file,
        params.augmenter.overfit_num > 0,
    )


if __name__ == "__main__":
    # load arguments and parameters
    parser = ArgumentParser(description="train a model on MIDI data")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="directory in which to store model checkpoints and training logs",
    )
    parser.add_argument(
        "--data_dir", default=None, help="directory of custom training data, in .npz"
    )
    parser.add_argument(
        "--param_file", default=None, help="path to parameter file, in .yaml"
    )
    parser.add_argument("--device", default="cuda:0", help="which device to run on")
    args = parser.parse_args()
    params = OmegaConf.load(args.param_file)

    console.log(f"{p} running with arguments:\n", args)
    console.log(f"{p} running with parameters:\n", params)

    main(args, params)
