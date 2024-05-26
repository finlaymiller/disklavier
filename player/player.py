import os
import mido
from mido import MidiFile, Message
from threading import Thread, Event
from queue import Queue
import time
import simpleaudio as sa

from utils import console, tick


class Player:
    p = "[blue]player[/blue]:"
    playing_file_path = ""
    playing_file = ""
    next_file_path = ""
    volume = 1.0  # volume scaling factor
    last_volume = 1.0  # memory

    def __init__(
        self,
        params,
        kill_event: Event,
        get_next_event: Event,
        waiting_event: Event,
        play_event: Event,
        filename_queue: Queue,
        command_queue: Queue,
        note_queue: Queue,
    ) -> None:
        self.params = params
        self.killed = kill_event
        self.get_next = get_next_event
        self.waiting = waiting_event
        self.play = play_event
        self.file_queue = filename_queue
        self.commands = command_queue
        self.notes = note_queue

        # multiple player override
        if self.params.p:
            self.p = self.params.p

    def play_loop(self):
        self.get_next.set()

        while not self.killed.is_set():
            # get next file from queue
            console.log(f"{self.p} waiting for file")
            while self.file_queue.empty():
                time.sleep(0.01)
                if self.killed.is_set():
                    console.log(f"{self.p}[bold orange1] shutting down")
                    return
            self.waiting.clear()
            self.playing_file_path, similarity = self.file_queue.get()
            self.playing_file = os.path.basename(self.playing_file_path)
            self.get_next.set()

            # print progress
            file_tempo = int(os.path.basename(self.playing_file).split("-")[1])
            found_tempo = -1
            for track in MidiFile(self.playing_file_path).tracks:
                for msg in track:
                    if msg.type == "set_tempo":
                        found_tempo = msg.tempo
            console.log(
                f"{self.p} playing '{self.playing_file}' ({file_tempo}BPM --> {round(mido.tempo2bpm(found_tempo)):01d}BPM) sim = {similarity:.03f}"
            )

            # play file
            self.play_midi(self.playing_file_path)
            self.waiting.set()

        console.log(f"{self.p}[bold orange1] shutting down")

    def play_midi(self, midi_path: str) -> None:
        midi = MidiFile(midi_path)
        # midi.print_tracks()

        # open file and wait for play event
        with mido.open_output(self.params.out_port) as outport:  # type: ignore
            console.log(f"{self.p} waiting to play")
            while not self.play.is_set():
                time.sleep(0.00001)
                if self.killed.is_set():
                    console.log(f"{self.p}[bold orange1] shutting down")
                    return

            self.play.clear()

            # play file
            last_beat = time.time()
            for msg in midi.play(meta_messages=True):
                if not msg.is_meta:
                    if msg.type == "note_on":  # type: ignore
                        self.notes.put_nowait(msg)
                    outport.send(msg)
                else:
                    if msg.type == "set_tempo":  # type: ignore
                        console.log(f"{self.p} playing at {mido.tempo2bpm(msg.tempo):.01f} BPM\t{msg}")  # type: ignore
                    if msg.type == "text" and self.params.do_tick:  # type: ignore
                        beat = time.time()
                        console.log(f"{self.p} {msg.text} [grey50]({beat - last_beat:.05f}s)")  # type: ignore
                        last_beat = beat
                        tick(self.params.tick, self.p, False)

                # end active notes and return if killed
                if self.killed.is_set():
                    with mido.open_output(self.params.out_port) as outport:  # type: ignore
                        for note in range(128):
                            msg = Message("note_off", note=note, velocity=0, channel=0)
                            outport.send(msg)
                    return
