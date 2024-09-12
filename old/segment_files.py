import os
from shutil import copy2
import zipfile
from datetime import datetime
from argparse import ArgumentParser

from rich import print
from rich.pretty import pprint
from rich.progress import (
    Progress,
    SpinnerColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
)

from dataset.dataset import augment_midi, segment_midi


def main(args):
    # set up filesystem
    if not os.path.exists(args.data_dir):
        print(f"no data dir found at {args.data_dir}")
        raise IsADirectoryError

    p_path = os.path.join(args.data_dir, "play")
    t_path = os.path.join(args.data_dir, "train")
    u_path = os.path.join(args.data_dir, "unsegmented")

    for dir in [p_path, t_path, u_path]:
        if not os.path.exists(dir):
            os.mkdir(dir)
            print(f"created new folder: '{dir}'")

    if args.limit is None:
        tracks = os.listdir(args.data_dir)
    else:
        tracks = os.listdir(args.data_dir)[: args.limit]

    p = Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
        MofNCompleteColumn(),
        refresh_per_second=1,
    )
    task_s = p.add_task("segmenting", total=len(tracks))

    # segment files
    segment_paths = []
    augment_paths = []
    with p:
        for filename in tracks:
            if filename.endswith(".mid"):
                if args.segment:
                    # segment
                    new_segments = segment_midi(
                        os.path.join(args.data_dir, filename),
                        p_path,
                    )
                    segment_paths.extend(new_segments)
                else:
                    copy2(
                        os.path.join(args.data_dir, filename),
                        os.path.join(p_path, filename),
                    )
                    new_segments = [os.path.join(args.data_dir, filename)]

                # augment
                if args.augment:
                    augment_paths.extend(
                        augment_midi(p, filename[:-4], new_segments, t_path)
                    )

                # move
                os.rename(
                    os.path.join(args.data_dir, filename),
                    os.path.join(u_path, filename),
                )

                p.update(task_s, advance=1)

    zip_path = os.path.join("data", "datasets", f"{args.dataset_name}_segmented.zip")
    print(f"compressing to zipfile '{zip_path}'")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(args.data_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, args.data_dir)
                zipf.write(file_path, arcname)

    print(
        f"[green bold]segmentation complete, {len(segment_paths)} play files generated and {len(augment_paths)} train files generated"
    )


if __name__ == "__main__":
    # load args
    parser = ArgumentParser(description="Argparser description")
    parser.add_argument("--data_dir", default=None, help="location of MIDI files")
    parser.add_argument("--dataset_name", default=None, help="name of dataset")
    parser.add_argument(
        "--store_metrics",
        default=f"metrics-{datetime.now().strftime('%y%m%d-%H%M%S')}.json",
        help="file to write segment metrics to (must be JSON)",
    )
    parser.add_argument(
        "--num_beats",
        type=int,
        default=8,
        help="number of beats each segment should have, not including the leading and trailing sections of each segment",
    )
    parser.add_argument(
        "-t",
        "--strip_tempo",
        action="store_true",
        help="strip all tempo messages from files",
    )
    parser.add_argument(
        "-s",
        "--segment",
        action="store_true",
        help="generate a segment for a number of semitone shifts",
    )
    parser.add_argument(
        "-a",
        "--augment",
        action="store_true",
        help="augment dataset and store files",
    )
    parser.add_argument(
        "-l",
        "--limit",
        type=int,
        default=None,
        help="stop after a certain number of files",
    )
    parser.add_argument(
        "-r",
        "--redis",
        action="store_true",
        default=False,
        help="upload files to redis",
    )
    args = parser.parse_args()

    pprint(args)

    main(args)
