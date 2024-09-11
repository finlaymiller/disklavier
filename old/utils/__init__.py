import re
import time
import numpy as np
import simpleaudio as sa
from threading import Event
from rich.console import Console


console = Console(record=True, log_time_format="%m-%d %H:%M:%S.%f")


def tick(
    tick_file: str = "data/m_tick.wav",
    p: str = "[cyan]metro[/cyan] :",
    do_print: bool = True,
):
    if do_print:
        console.log(f"{p} [grey50]tick!")

    sa.WaveObject.from_wave_file(tick_file).play()

    return


def tick_loop(
    bpm: int, stop_event: Event, p: str = "[cyan]metro[/cyan] : ", do_print=True
):
    start_time = time.time()
    last_beat = start_time

    while not stop_event.is_set():
        beat = time.time()
        if beat - last_beat >= 60.0 / bpm:
            if do_print:
                console.log(
                    f"{p} [grey50]tick! [bright_black]({beat - last_beat:.02f}s)"
                )
            sa.WaveObject.from_wave_file("data/m_tick.wav").play()
            last_beat = beat
    return


def shift_array(arr, up=0, down=0):
    """Shift array vertically within bounds"""
    if up > 0:
        arr = np.roll(arr, -up, axis=0)
        arr[-up:] = 0
    elif down > 0:
        arr = np.roll(arr, down, axis=0)
        arr[:down] = 0
    return arr


def get_tempo(filename: str) -> int:
    return int(filename.split('-')[1])

def extract_transformations(filename: str):
    """CHATGPT UNTESTESTED
    Parses the filename, strips out the transformation section, and returns the modified filename
    and the transformations as a dictionary.

    Args:
        filename (str): The input filename with the transformation section.

    Returns:
        list: A list containing the modified filename and the transformations dictionary.
    """
    # regex pattern to match the transformation section
    pattern = r'_(t\d+s\d+)\.mid$'
    
    # search for the transformation section in the filename
    match = re.search(pattern, filename)
    if not match:
        raise ValueError("Filename format is incorrect or does not contain transformations.")

    # extract the transformation section
    transformation_str = match.group(1)

    # parse the transformation values
    transpose = int(transformation_str[1:3])
    shift = int(transformation_str[4:6])

    # create the transformations dictionary
    transformations = {'transpose': transpose, 'shift': shift}

    # create the modified filename
    modified_filename = filename.replace(f'_{transformation_str}', '')

    return [modified_filename, transformations]
