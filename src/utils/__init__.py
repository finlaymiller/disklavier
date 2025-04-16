import os
import re
import csv
import time
from threading import Event
from rich.console import Console

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
import pygame

console = Console(record=True, log_time_format="%m-%d %H:%M:%S.%f")


def tick(
    bpm: int,
    stop_event: Event,
    p: str = "[cyan]metro[/cyan] : ",
    do_tick: bool = True,
    do_print: bool = True,
    pf_tick: str = "data/m_tick.wav",
):
    pygame.mixer.init()
    tick_sound = pygame.mixer.Sound(pf_tick)
    start_time = time.time()
    last_beat = start_time

    while not stop_event.is_set():
        beat = time.time()
        if beat - last_beat >= 60.0 / bpm:
            if do_print:
                console.log(
                    f"{p} [grey50]tick! [bright_black]({beat - last_beat:.02f}s)"
                )
            if do_tick:
                tick_sound.play()
            last_beat = beat

        time.sleep(0.01)
    return


def get_transformations(filename: str) -> tuple[str, dict[str, int]]:
    """
    Parses the filename, strips out the transformation section, and returns the modified filename
    and the transformations as a dictionary. Transformations are formatted as `_t{transpose}s{shift}`,
    where `transpose` and `shift` are integers on the range [0, 11] and [0, 7], respectively.
    They are both zero padded to two digits. The function works with filenames both with and without
    the .mid extension.

    Parameters
    ----------
    filename : str
        The input filename with the transformation section, with or without .mid extension.

    Returns
    -------
    tuple
        A tuple containing the modified filename and the transformations dictionary.
    """
    pattern = r"_(t\d+s\d+)(?:\.mid)?$"  # regex to match the transformation section, extension optional
    match = re.search(pattern, filename)
    if not match:
        raise ValueError(
            "Filename format is incorrect or does not contain transformations."
        )
    transformation_str = match.group(1)
    transpose = int(transformation_str[1:3])
    shift = int(transformation_str[4:6])

    return (
        filename.replace(f"_{transformation_str}", ""),
        {"transpose": transpose, "shift": shift},
    )


def basename(filename: str) -> str:
    return os.path.splitext(os.path.basename(filename))[0]


def write_log(filename: str, *args):
    with open(filename, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(args)
