import time
import wave
import numpy as np
import simpleaudio as sa
from threading import Event, Thread
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
    """
    Plays a metronome tick at the specified BPM

    Args:
    bpm: Beats per minute, defines the tempo of the metronome.
    """
    start_time = time.time()
    last_beat = start_time

    while not stop_event.is_set():
        beat = time.time()
        if beat - last_beat >= 60.0 / bpm:
            if do_print:
                console.log(f"{p} [grey50]tick!")
            thread_t = Thread(
                target=sa.WaveObject.from_wave_file("data/m_tick.wav").play,
                args=(),
                name="",
            )
            thread_t.start()
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
