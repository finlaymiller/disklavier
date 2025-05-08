import os
import sys
from omegaconf import OmegaConf
from argparse import ArgumentParser
from PySide6.QtWidgets import QApplication

from utils import console
from widgets.main_window import MainWindow
from ml.specdiff.model import SpectrogramDiffusion, DEFAULT_CONFIG


tag = "[white]main[/white]  :"
config = DEFAULT_CONFIG
config["device"] = "mps"
config["encoder_weights_path"] = (
    "/Users/finlay/Documents/Programming/disklavier/src/ml/specdiff/note_encoder.bin"
)
model = SpectrogramDiffusion(
    new_config=config,
    fix_time=False,
    verbose=True,
)
model.embed("/Users/finlay/Documents/Programming/disklavier/data/datasets/test/velocitytweaks-060-05_c4vel100_t00s00.mid")

def main(args, params):
    global model
    app = QApplication(sys.argv)
    main_window = MainWindow(args, params, model)
    main_window.args = args
    main_window.show()
    sys.exit(app.exec())


def load_args(args):
    # load/build arguments and parameters
    parser = ArgumentParser(description="Argparser description")
    parser.add_argument(
        "-d", "--dataset", type=str, default="20250420", help="name of the dataset"
    )
    parser.add_argument(
        "--dataset_path", type=str, default=None, help="path to MIDI files"
    )
    parser.add_argument(
        "-p",
        "--params",
        type=str,
        default="max",
        help="name of parameter file (must be located at 'params/[NAME].yaml')",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="data/outputs/logs",
        help="directory in which to store outputs (logs, recordings, etc...)",
    )
    parser.add_argument(
        "-t",
        "--tables",
        type=str,
        default=None,
        help="directory in which precomputed tables are stored",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default=None,
        help="option to override seeker metric in yaml file",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        help="option to override seeker mode in yaml file",
    )
    parser.add_argument(
        "-b",
        "--bpm",
        type=int,
        help="bpm to record and play at, in bpm",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="enable verbose output",
    )
    parser.add_argument(
        "-r",
        "--replay",
        action="store_true",
        help="run again using last seed file",
    )
    args = parser.parse_args()

    # first load template parameters
    template_params = OmegaConf.load("params/template.yaml")

    # then load specified parameters and merge with template
    # (specified parameters override template parameters)
    specific_params = OmegaConf.load(f"params/{args.params}.yaml")
    params = OmegaConf.merge(template_params, specific_params)

    # handle overrides
    if args.dataset_path == None:
        args.dataset_path = f"data/datasets/{args.dataset}/augmented"
    if args.tables == None:
        args.tables = f"data/tables/{args.dataset}"
    if args.metric != None:
        console.log(f"{tag} overriding seeker metric to '{args.metric}'")
        params.seeker.metric = args.metric
    if args.mode != None:
        console.log(f"{tag} overriding seeker mode to '{args.mode}'")
        params.seeker.mode = args.mode

    # copy these so that they only have to be specified once
    params.scheduler.n_beats_per_segment = params.n_beats_per_segment
    params.metronome.n_beats_per_segment = params.n_beats_per_segment

    if args.replay:
        # get path to last seed file
        entries = os.listdir(args.output)
        folders = [
            entry
            for entry in entries
            if os.path.isdir(os.path.join(args.output, entry))
        ]
        folders.sort()
        last_folder = folders[-1]
        console.log(f"{tag} last run is in folder '{last_folder}'")

        last_timestamp, _, last_initialization, _ = last_folder.split("_")
        pf_last_playlist = os.path.join(
            args.output, last_folder, f"playlist_{last_timestamp}.csv"
        )
        pf_last_seed = None
        with open(pf_last_playlist, newline="") as csvfile:
            import csv

            first_row = next(csv.DictReader(csvfile), None)
            pf_last_seed = (
                first_row["file path"]
                if first_row and "file path" in first_row
                else None
            )

        if pf_last_seed is None:
            raise FileNotFoundError("couldn't load seed file path")

        params.initialization = last_initialization
        params.kickstart_path = pf_last_seed

    return args, params


if __name__ == "__main__":
    args, params = load_args(sys.argv[1:])

    console.log(f"{tag} loading with arguments:\n\t{args}")
    console.log(f"{tag} loading with parameters:\n\t{params}")

    if not os.path.exists(args.tables):
        console.log(
            f"{tag} [red bold]ERROR[/red bold]: table directory not found, exiting..."
        )
        raise FileNotFoundError("table directory not found")
    
    main(args, params)
