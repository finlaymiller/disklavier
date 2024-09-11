import time
import simpleaudio as sa
from threading import Event, Thread

from utils import console


class Metronome:
    p = "[cyan]metro[/cyan] :"
    tick_file = "data/m_kick.wav"

    def __init__(
        self,
        params,
        kill_event: Event,
        ready_event: Event,
        go_event: Event,
    ):
        self.params = params
        self.killed = kill_event
        self.ready = ready_event
        self.go = go_event

        self.tick_rate = 60.0 / self.params.tempo

        if self.params.tick_file:
            self.tick_file = self.params.tick_file

    def tick(self) -> None:
        console.log(f"{self.p} ticking every {self.tick_rate:.01f} seconds")

        beats = self.params.beats_per_seg
        start_time = time.time()
        last_beat = start_time

        while not self.killed.is_set():
            beat = time.time()
            if beat - last_beat >= self.tick_rate:
                if beats // self.params.beats_per_seg:
                    if self.params.do_tick:
                        console.log(
                            f"{self.p} beat {beats} [grey50]({beat - last_beat:.05f}s)[/grey50]\t[green]go!"
                        )
                    self.go.set()
                    beats = 0
                elif beats // (self.params.beats_per_seg - 1):
                    if self.params.do_tick:
                        console.log(
                            f"{self.p} beat {beats} [grey50]({beat - last_beat:.05f}s)[/grey50]\t[dark_orange]ready?"
                        )
                    self.ready.set()
                else:
                    if self.params.do_tick:
                        console.log(
                            f"{self.p} beat {beats} [grey50]({beat - last_beat:.05f}s)"
                        )

                if self.params.do_tick:
                    thread_t = Thread(
                        target=sa.WaveObject.from_wave_file(self.tick_file).play,
                        args=(),
                        name="",
                    )
                    thread_t.start()

                last_beat = beat
                beats += 1

        console.log(f"{self.p} metronome shutting down")
