import os
from datetime import datetime
from argparse import ArgumentParser

from rich import print
from rich.pretty import pprint
from rich.progress import track

from dataset.dataset import segment_midi

if __name__ == "__main__":
    # load args
    parser = ArgumentParser(description="Argparser description")
    parser.add_argument("--data_dir", default=None, help="location of MIDI files")
    parser.add_argument(
        "--output_dir", default=None, help="location to write segments to"
    )
    parser.add_argument(
        "-m",
        "--store_metrics",
        default=f"metrics-{datetime.now().strftime('%y%m%d-%H%M%S')}.json",
        help="file to write segment metrics to (must be JSON)",
    )
    parser.add_argument(
        "-n" "--num_beats",
        type=int,
        default=8,
        help="number of beats each segment should have",
    )
    parser.add_argument(
        "-t",
        "--strip_tempo",
        action="store_true",
        help="strip all tempo messages from files",
    )
    parser.add_argument(
        "-s",
        "--do_shift",
        type=int,
        default=12,
        help="generate a segment for a number of semitone shifts",
    )
    parser.add_argument(
        "-l",
        "--limit",
        type=int,
        default=None,
        help="stop after a certain number of files",
    )
    args = parser.parse_args()
    pprint(args)

    # set up filesystem
    if not os.path.exists(args.data_dir):
        print(f"no data dir found at {args.data_dir}")
        exit()
    if os.path.exists(args.output_dir):
        i = 0
        for i, file in enumerate(os.listdir(args.output_dir)):
            os.remove(os.path.join(args.output_dir, file))
            i += 1
        print(f"cleaned {i} files out of output folder: '{args.output_dir}'")
    else:
        print(f"creating new output folder: '{args.output_dir}'")
        os.mkdir(args.output_dir)

    graveyard = os.path.join("outputs", "graveyard")
    if os.path.exists(graveyard):
        i = 0
        for i, file in enumerate(os.listdir(graveyard)):
            os.remove(os.path.join(graveyard, file))
            i += 1
        print(f"cleaned {i} files out of graveyard: '{graveyard}'")
    else:
        print(f"creating new graveyard: '{graveyard}'")
        os.mkdir(graveyard)

    if args.limit is None:
        dataset = os.listdir(args.data_dir)
    else:
        dataset = os.listdir(args.data_dir)[: args.limit]

    # segment files
    num_files = 0
    for filename in track(dataset, description="generating segments"):
        if filename.endswith(".mid") or filename.endswith(".midi"):
            num_files += segment_midi(os.path.join(args.data_dir, filename), args)

    total_segs = len(os.listdir(args.output_dir))
    print(f"[green]segmentation complete, {num_files} files generated")
