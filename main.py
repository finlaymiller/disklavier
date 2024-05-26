import os
import time

# import logging
# import logging.config
from datetime import datetime
from argparse import ArgumentParser
from omegaconf import OmegaConf
from threading import Thread, Event, enumerate
from rich.prompt import Confirm, IntPrompt

from overseer.overseer import Overseer
from utils import console, tick


p = "[white]main[/white]  :"


def check_tempo(tempo: int = 60) -> int:
    console.log(f"{p} running at {tempo}bpm")
    console.log(f"{p} how does this sound?")
    stop_event = Event()

    metro_thread = Thread(target=tick, args=(tempo, stop_event))
    metro_thread.start()

    time.sleep(60 / tempo * 4)

    stop_event.set()
    metro_thread.join()

    if Confirm.ask(f"{p} was that bpm ok?", default=True):
        return int(tempo)
    else:
        while True:
            new_tempo = IntPrompt.ask(
                f"{p} enter a new tempo between [b]20[/b] and [b]200[/b] [60]",
                default=60,
            )
            if new_tempo >= 20 and new_tempo <= 200:
                return check_tempo(new_tempo)
            console.log(f"{p} [prompt.invalid]tempo must be between 20 and 200")


if __name__ == "__main__":
    # load args and params
    parser = ArgumentParser(description="Argparser description")
    parser.add_argument("--data_dir", default=None, help="location of MIDI files")
    parser.add_argument(
        "--param_file", default=None, help="path to parameter file, in .yaml"
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="directory in which to store outputs (logs, recordings, etc...)",
    )
    parser.add_argument(
        "--log_config",
        default=None,
        help="where the logging config file is found",
    )
    parser.add_argument(
        "-f",
        "--force_rebuild",
        action="store_true",
        help="whether to rebuild similarity metrics",
    )
    parser.add_argument(
        "-i",
        "--random_init",
        action="store_true",
        help="dont wait for user input, just start playing",
    )
    parser.add_argument(
        "-p",
        "--plot",
        action="store_true",
        help="plot pitch histogram and midi file pairs as we go",
    )
    parser.add_argument(
        "-t",
        "--tick",
        action="store_true",
        help="metronome during playback",
    )
    parser.add_argument(
        "-v",
        "--velocity",
        type=float,
        default=1.0,
        help="scale note velocities [0.1, 2.0]",
    )
    parser.add_argument(
        "--tempo",
        type=int,
        help="tempo to record and play at, in bpm",
    )
    parser.add_argument(
        "-c",
        "--commands",
        action="store_true",
        help="enable keyboard commands",
    )
    parser.add_argument(
        "-k",
        "--kickstart",
        type=str,
        help="use provided midi file as prompt",
    )
    args = parser.parse_args()
    params = OmegaConf.load(args.param_file)

    console.log(f"{p} loading with args:\n\t{args}")
    console.log(f"{p} loading with params:\n\t{params}")

    # filesystem setup
    log_dir = os.path.join(args.output_dir, "logs")
    record_dir = os.path.join(args.output_dir, "records")
    plot_dir = os.path.join(args.output_dir, "plots", f"{datetime.now().strftime('%y%m%d-%H%M')}")
    playlist_dir = os.path.join(
        "data", "playlist", f"{datetime.now().strftime("%y%m%d-%H%M")}-{params.seeker.property}"
    )

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    if not os.path.exists(log_dir):
        console.log(f"{p} creating new logging folder: '{log_dir}'")
        os.mkdir(log_dir)
    if not os.path.exists(record_dir):
        console.log(f"{p} creating new recordings folder: '{record_dir}'")
        os.mkdir(record_dir)
    if not os.path.exists(os.path.join(args.output_dir, "plots")):
        os.mkdir(os.path.join(args.output_dir, "plots"))
    if not os.path.exists(plot_dir):
        console.log(f"{p} creating new plots folder: '{plot_dir}'")
        os.mkdir(plot_dir)
    if not os.path.exists(os.path.join("data", "playlist")):
        os.mkdir(os.path.join("data", "playlist"))
    if not os.path.exists(playlist_dir):
        console.log(f"{p} creating new playlist folder: '{playlist_dir}'")
        os.mkdir(playlist_dir)
    console.log(f"{p}[green bold] filesystem set up complete")

    if args.tempo:
        # playback_tempo = check_tempo(args.tempo)
        playback_tempo = args.tempo

    if args.velocity < 0.1 or args.velocity > 2.0:
        console.log(f"{p}[red] velocity argument is out of bounds\n{args.velocity} must be between [0.1, 2.0]")

    # run!
    overseer = Overseer(
        params,
        args,
        playlist_dir,
        record_dir,
        plot_dir,
        playback_tempo,
    )
    console.log(f"{p} overseer setup complete")
    overseer.run()

    # run complete
    console.save_text(
        os.path.join(log_dir, f"{datetime.now().strftime('%y-%m-%d_%H%M%S')}.log")
    )
    console.log(f"{p}[green bold] session complete, exiting")

    console.log(f"Current threads: {enumerate()}")
