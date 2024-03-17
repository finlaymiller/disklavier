import os
import mido
from mido import MidiFile
from threading import Thread, Event
from queue import Queue

from utils import console, tick


class Player:
    p = "[blue]play[/blue]  :"
    is_playing = False

    def __init__(
        self,
        params,
        record_dir: str,
        do_tick: bool,
        kill_event: Event,
        playback_event: Event,
        filename_queue: Queue,
        progress_queue: Queue,
    ) -> None:
        self.params = params
        self.record_dir = record_dir
        self.do_tick = do_tick
        self.kill_event = kill_event
        self.get_next = playback_event
        self.file_queue = filename_queue
        self.playback_progress = progress_queue
        self.out_port = mido.open_output(self.params.out_port)  # type: ignore

    def playback_loop(self, seed_file_path: str, fh):
        """"""
        self.playing_file_path = seed_file_path
        (next_file_path, similarity) = self.file_queue.get(block=True)
        next_file = os.path.basename(next_file_path)
        first_loop = True

        while next_file_path is not None and not self.kill_event.is_set():
            self.playing_file = os.path.basename(self.playing_file_path)
            if not first_loop:  # bad code!
                next_file_path, similarity = self.file_queue.get()
                next_file = os.path.basename(next_file_path)
            self.get_next.set()

            console.log(
                f"{self.p} playing '{self.playing_file}' -- {similarity:.3f} --> '{next_file}'"
            )

            self.play_midi_file(self.playing_file_path)
            self.playing_file_path = next_file_path

            first_loop = False

        console.log(f"{self.p} [orange]shutting down")

    def play_midi_file(self, midi_path: str) -> None:
        """"""
        midi = MidiFile(midi_path)
        file_tempo = int(os.path.basename(midi_path).split("-")[1])
        found_tempo = -1
        printed_msg = False

        for track in midi.tracks:
            for msg in track:
                if msg.type == "set_tempo":
                    found_tempo = msg.tempo
        console.log(
            f"{self.p} default tempo is {file_tempo}, playback tempo is {round(mido.tempo2bpm(found_tempo)):01d}"
        )

        if self.do_tick:
            stop_tick = Event()
            tick_thread = Thread(
                target=tick,
                args=(int(mido.tempo2bpm(found_tempo)), stop_tick, self.p, False),
            )
        for msg in midi.play():
            if self.do_tick and not tick_thread.is_alive():
                tick_thread.start()
            self.out_port.send(msg)
            self.playback_progress.put(msg.time)  # type: ignore

            if self.kill_event.isSet() and not printed_msg:
                console.log(f"{self.p} [yellow]finishing playback of the current file")
                printed_msg = True

        if self.do_tick:
            stop_tick.set()
            tick_thread.join()
