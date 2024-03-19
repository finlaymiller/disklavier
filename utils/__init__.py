import time
import wave
import random
import numpy as np
from pathlib import Path
import simpleaudio as sa
from threading import Event
from rich.console import Console


console = Console(record=True, log_time_format="%m-%d %H:%M:%S")


def tick(bpm: int, stop_event: Event, p: str = "[cyan]metro[/cyan] : ", do_print=True):
    """
    Plays a metronome tick at the specified BPM

    Args:
    bpm: Beats per minute, defines the tempo of the metronome.
    """
    tick = sa.WaveObject.from_wave_file("data/tick.wav")
    tick_len = 0
    with wave.open("data/tick.wav", "rb") as wave_file:
        frames = wave_file.getnframes()
        rate = wave_file.getframerate()
        tick_len = frames / float(rate)

    while not stop_event.is_set():
        if do_print:
            console.log(f"{p} [grey50]tick!")
        play_obj = tick.play()
        play_obj.wait_done()
        time.sleep(60.0 / bpm - tick_len)
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
